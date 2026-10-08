"""Controlled A/A2 NVT dissociation and reversible-rate trajectory experiments.

Place at scripts/run_a2_reversible.py and run from the repository root.

Examples:
  python -m scripts.run_a2_reversible --mode dissociation
  python -m scripts.run_a2_reversible --mode roundtrip
  python -m scripts.run_a2_reversible --mode reversible --steps 200

Uses a dimensionless numerical mass of 1 in the existing integrator units.
Rate and time labels follow the driver's current convention; this is NOT
an NPT or quantitatively calibrated reproduction of the published benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from reactive_md.config import SimConfig
from reactive_md.md_a2_driver import run_md_nvt_a2
from reactive_md.reactions.a2 import A2Reaction

# Reuse the tested 32-particle lattice, topology builder, and CSV writer.
from .run_a2_small import (
    KB_REAL,
    K_BOND_JAX,
    make_forcefield,
    make_positions,
    make_system,
    write_csv,
)


def make_initial_system(n_atoms: int, *, bonded: bool):
    """Create the A2 topology explicitly for the forced dissociation run."""
    system = make_system(n_atoms)
    if not bonded:
        return system
    # Keep unique IDs: the explicit bond list handles the bonded LJ
    # exclusion, and distinct IDs allow LJ to return after dissociation.
    molecule_id = np.arange(n_atoms, dtype=np.int32)
    return replace(
        system,
        bonds=(
            np.array([[0, 1]], dtype=np.int32),
            np.array([K_BOND_JAX], dtype=np.float32),
            np.array([1.5], dtype=np.float32),
        ),
        molecule_id=molecule_id,
    )


def simulate_case(
    *,
    mode: str,
    steps: int,
    seed: int = 0,
    dt: float = 0.00025,
    check_every: int = 1,
    relaxation_time: float = 0.002,
    association_rate: float | None = None,
    dissociation_rate: float | None = None,
):
    """Return an A2MDRunResult; no files are written by this function."""
    if mode not in {"dissociation", "roundtrip", "reversible"}:
        raise ValueError(f"Unknown mode: {mode}")
    if steps <= 0 or check_every <= 0 or dt <= 0 or relaxation_time <= 0:
        raise ValueError("steps, check_every, dt, relaxation_time must be positive")

    defaults = {
        "dissociation": (0.0, 1.0 / dt),
        "roundtrip": (1.0 / dt, 1.0 / dt),
        "reversible": (0.25 / dt, 0.25 / dt),
    }
    default_assoc, default_dissoc = defaults[mode]
    ka = default_assoc if association_rate is None else float(association_rate)
    kd = default_dissoc if dissociation_rate is None else float(dissociation_rate)
    interval = min(steps, check_every) * dt
    for name, rate in (("association_rate", ka), ("dissociation_rate", kd)):
        if not np.isfinite(rate) or rate < 0 or rate * interval > 1:
            raise ValueError(f"{name} * reactive_interval must be in [0, 1]")

    R, box = make_positions()
    # A bonded pair is placed at 3.0 A for dissociation: the restored LJ
    # interaction is substantially less repulsive than at 1.5 A.
    if mode == "dissociation":
        R = R.at[1, 0].set(R[0, 0] + 3.0)
    _, shift_fn = space.periodic(box)
    sys = make_initial_system(len(R), bonded=(mode == "dissociation"))
    ff = make_forcefield(R, box, sys)
    reaction = A2Reaction(
        a_indices=np.arange(16, dtype=np.int32),
        k_bond_jax=K_BOND_JAX,
        r0_bond=1.5,
        association_cutoff=2.7,
    )
    cfg = SimConfig(
        steps=steps,
        check_every=check_every,
        max_events=100000,
        dt=dt,
        temperature_k=188.0,
        kb_real=KB_REAL,
        tau_T=0.5,
    )
    result = run_md_nvt_a2(
        jax.random.PRNGKey(seed),
        cfg=cfg,
        init_positions=R,
        masses=jnp.ones((len(R),), dtype=jnp.float32),
        box=box,
        shift_fn=shift_fn,
        ff=ff,
        sys=sys,
        reaction=reaction,
        association_rate_ps=ka,
        dissociation_rate_ps=kd,
        rng=np.random.default_rng(seed),
        relaxation_time_ps=relaxation_time,
    )
    return result, R, box


def check_topology_and_lj(result, *, expect_no_bonds: bool):
    """Check bonded topology and that free A atoms can interact by LJ.

    In this forcefield, same-molecule exclusions may persist even after
    removing a bond. An unbonded pair must not retain a shared molecule ID.
    """
    bonds = np.asarray(result.sys.bonds[0]).reshape(-1, 2)
    ids = np.asarray(result.sys.molecule_id)
    if expect_no_bonds:
        if len(bonds) != 0:
            raise AssertionError(f"Expected no A2 bonds, got {bonds.tolist()}")
        if ids[0] == ids[1]:
            raise AssertionError(
                "Dissociated atoms 0 and 1 still share molecule_id. "
                "The nonbonded exclusion may persist, preventing LJ restoration. "
                "Check A2 topology updates before interpreting this trajectory."
            )
    else:
        if len(bonds) != 1:
            raise AssertionError(f"Expected one A2 bond, got {bonds.tolist()}")


def save_result(result, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "a2_trajectory.csv"
    events_path = output_dir / "a2_events.csv"
    sample_rows = [asdict(sample) for sample in result.samples]
    write_csv(samples_path, sample_rows, list(sample_rows[0]))
    times = {sample.step: sample.elapsed_time_ps for sample in result.samples}
    event_rows = [
        {
            "step": rec.step,
            "elapsed_time_ps": times[rec.step],
            "reaction_type": rec.event.reaction_type,
            "i": rec.event.i,
            "j": rec.event.j,
            "distance_angstrom": rec.event.distance,
            "probability": rec.event.probability,
        }
        for rec in result.event_history
    ]
    fields = ["step", "elapsed_time_ps", "reaction_type", "i", "j",
              "distance_angstrom", "probability"]
    write_csv(events_path, event_rows, fields)
    return samples_path, events_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dissociation", "roundtrip", "reversible"],
                        default="dissociation")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--check-every", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.00025)
    parser.add_argument("--relaxation-time", type=float, default=0.002)
    parser.add_argument("--association-rate", type=float, default=None)
    parser.add_argument("--dissociation-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    steps = args.steps if args.steps is not None else {
        "dissociation": 1, "roundtrip": 2, "reversible": 200
    }[args.mode]
    result, _, _ = simulate_case(
        mode=args.mode,
        steps=steps,
        seed=args.seed,
        dt=args.dt,
        check_every=args.check_every,
        relaxation_time=args.relaxation_time,
        association_rate=args.association_rate,
        dissociation_rate=args.dissociation_rate,
    )
    last = result.samples[-1]
    print(f"Mode: {args.mode}")
    print(f"MD steps: {result.steps_done}")
    print(f"Events: {result.accepted_events} "
          f"({last.n_association} association, {last.n_dissociation} dissociation)")
    print(f"Final A2: {last.n_a2}")
    print(f"Final PE: {last.potential_energy_kcal_mol:.6f} kcal/mol")
    print(f"Final KE: {last.kinetic_energy_kcal_mol:.6f} kcal/mol")
    print("Event sequence:", [r.event.reaction_type for r in result.event_history])

    if args.mode == "dissociation":
        if last.n_dissociation != 1 or last.n_a2 != 0:
            raise AssertionError("Forced dissociation did not remove the A2 bond")
        check_topology_and_lj(result, expect_no_bonds=True)
    elif args.mode == "roundtrip":
        kinds = [r.event.reaction_type for r in result.event_history]
        if kinds != ["association", "dissociation"] or last.n_a2 != 0:
            raise AssertionError(f"Expected association -> dissociation; got {kinds}")
        check_topology_and_lj(result, expect_no_bonds=True)
    else:
        if last.n_association == 0 or last.n_dissociation == 0:
            print("NOTE: Not both directions occurred; extend steps or change seed/rates.")

    output_dir = args.output_dir or Path("results") / f"a2_{args.mode}"
    samples_path, events_path = save_result(result, output_dir)
    print(f"Trajectory: {samples_path}")
    print(f"Events: {events_path}")


if __name__ == "__main__":
    main()

