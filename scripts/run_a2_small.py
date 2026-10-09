"""Small, controlled A + A <-> A2 NVT trajectory with CSV outputs.

Run from the repository root:
    python -m scripts.run_a2_small --output-dir results/a2_smoke

This demonstration uses artificial kinetics and shortened relaxation by
*default*. It is not the published rs@md NPT benchmark.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from reactive_md.config import SimConfig
from reactive_md.forcefield import build_forcefield
from reactive_md.md_a2_driver import run_md_nvt_a2
from reactive_md.reaction import SystemState
from reactive_md.reactions.a2 import A2Reaction


KB_REAL = 0.0019872041  # kcal/(mol K)
EPSILON = 1.9572 / 4.184  # kcal/mol
SIGMA = 3.4  # angstrom
# Paper: (1/2) * 25000 kJ/mol/nm^2 * (r-r0)^2.
# JAX-MD/OPLS bond form: k_jax * (r-r0)^2, r in angstrom.
K_BOND_JAX = 0.5 * 25000.0 / 100.0 / 4.184


def make_positions():
    """16 A + 16 B on a dilute periodic lattice; one accessible A pair."""
    box = np.array([19.2, 19.2, 19.2], dtype=np.float32)
    positions = []
    for z in (4.8, 14.4):
        for y in (2.4, 7.2, 12.0, 16.8):
            for x in (2.4, 7.2, 12.0, 16.8):
                positions.append((x, y, z))
    positions = np.asarray(positions, dtype=np.float32)
    # Atom 0 is at (2.4, 2.4, 4.8); atom 1 is shifted to 2.65 A
    # away. The association cutoff is 2.7 A.
    positions[1, 0] = positions[0, 0] + 2.65
    return jnp.asarray(positions), jnp.asarray(box)


def make_system(n_atoms):
    empty_four = (
        np.empty((0, 4), dtype=np.int32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.float32),
    )
    return SystemState(
        bonds=(
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ),
        angles=(
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ),
        torsions=empty_four,
        impropers=empty_four,
        charges=np.zeros(n_atoms, dtype=np.float32),
        sigmas=np.full(n_atoms, SIGMA, dtype=np.float32),
        epsilons=np.full(n_atoms, EPSILON, dtype=np.float32),
        # All atoms start as separate molecules. The A2 trial builder
        # handles the bonded topology after association.
        molecule_id=np.arange(n_atoms, dtype=np.int32),
        pf6_reacted=jnp.zeros((0,), dtype=jnp.bool_),
    )


def make_forcefield(R, box, sys):
    return build_forcefield(
        R=R,
        box=box,
        bond_idx=np.asarray(sys.bonds[0]),
        k_b=np.asarray(sys.bonds[1]),
        r0=np.asarray(sys.bonds[2]),
        angle_idx=np.asarray(sys.angles[0]),
        k_theta=np.asarray(sys.angles[1]),
        theta0=np.asarray(sys.angles[2]),
        torsions=tuple(np.asarray(x) for x in sys.torsions),
        impropers=tuple(np.asarray(x) for x in sys.impropers),
        charges=np.asarray(sys.charges),
        sigmas=np.asarray(sys.sigmas),
        epsilons=np.asarray(sys.epsilons),
        molecule_id=np.asarray(sys.molecule_id),
        r_cut=9.0,
        dr_threshold=0.5,
    )


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/a2_smoke"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--check-every", type=int, default=1)
    parser.add_argument("--dt-ps", type=float, default=0.00025)
    parser.add_argument("--association-rate-ps", type=float, default=4000.0)
    parser.add_argument("--dissociation-rate-ps", type=float, default=0.0)
    parser.add_argument("--relaxation-time-ps", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.steps <= 0 or args.check_every <= 0 or args.dt_ps <= 0:
        parser.error("steps, check-every and dt-ps must be positive")
    if args.relaxation_time_ps <= 0:
        parser.error("relaxation-time-ps must be positive")
    interval = min(args.steps, args.check_every) * args.dt_ps
    for name, rate in (
        ("association-rate-ps", args.association_rate_ps),
        ("dissociation-rate-ps", args.dissociation_rate_ps),
    ):
        if not np.isfinite(rate) or rate < 0 or rate * interval > 1.0:
            parser.error(f"{name} must satisfy 0 <= rate * check_interval <= 1")

    R, box = make_positions()
    disp_fn, shift_fn = space.periodic(box)
    system = make_system(len(R))
    ff = make_forcefield(R, box, system)
    reaction = A2Reaction(
        a_indices=np.arange(16, dtype=np.int32),
        k_bond_jax=K_BOND_JAX,
        r0_bond=1.5,
        association_cutoff=2.7,
    )
    cfg = SimConfig(
        steps=args.steps,
        check_every=args.check_every,
        max_events=100000,
        dt=args.dt_ps,
        temperature_k=188.0,
        kb_real=KB_REAL,
        tau_T=0.5,
    )
    result = run_md_nvt_a2(
        jax.random.PRNGKey(args.seed),
        cfg=cfg,
        init_positions=R,
        masses=np.ones(len(R)),#jnp.full((len(R),), 39.948, dtype=jnp.float32),
        box=box,
        shift_fn=shift_fn,
        ff=ff,
        sys=system,
        reaction=reaction,
        association_rate_ps=args.association_rate_ps,
        dissociation_rate_ps=args.dissociation_rate_ps,
        rng=np.random.default_rng(args.seed),
        relaxation_time_ps=args.relaxation_time_ps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "a2_trajectory.csv"
    event_path = args.output_dir / "a2_events.csv"
    sample_rows = [asdict(sample) for sample in result.samples]
    write_csv(samples_path, sample_rows, list(sample_rows[0]))
    time_by_step = {sample.step: sample.elapsed_time_ps for sample in result.samples}
    event_rows = [
        {
            "step": rec.step,
            "elapsed_time_ps": time_by_step[rec.step],
            "reaction_type": rec.event.reaction_type,
            "i": rec.event.i,
            "j": rec.event.j,
            "distance_angstrom": rec.event.distance,
            "probability": rec.event.probability,
        }
        for rec in result.event_history
    ]
    write_csv(
        event_path,
        event_rows,
        ["step", "elapsed_time_ps", "reaction_type", "i", "j",
         "distance_angstrom", "probability"],
    )
    last = result.samples[-1]
    print(f"Particles: {len(R)} (16 A, 16 B)")
    print(f"MD steps: {result.steps_done}")
    print(f"Accepted events: {result.accepted_events} "
          f"({last.n_association} association, {last.n_dissociation} dissociation)")
    print(f"Final A2 molecules: {last.n_a2}")
    print(f"Final PE: {last.potential_energy_kcal_mol:.6f} kcal/mol")
    print(f"Final KE: {last.kinetic_energy_kcal_mol:.6f} kcal/mol")
    print(f"Trajectory: {samples_path}")
    print(f"Events: {event_path}")


if __name__ == "__main__":
    main()

