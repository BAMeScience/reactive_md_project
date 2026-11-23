"""Full-size 200 A + 824 B reactive NVT driver, rs@md PEL I.

The reaction algorithms and MD integrator are imported unchanged from the
Phase-7 reactive_md project. Both modes use identical initialization and
force-field construction. This is NOT the published NPT/PEL II benchmark.

From a flat working directory containing this script, with reactive_md on
PYTHONPATH and run_a2_small.py available:

    python run_a2_1024_rho07_nvt.py --mode mc --steps 10000 \\
        --output-dir results/1024_mc_smoke

Set JAX_ENABLE_X64=True before launching Python for 64-bit support.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from reactive_md.config import SimConfig
from reactive_md.forcefield import build_forcefield
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_driver import run_md_nvt_a2

from ..systems.a2 import (
    KB_REAL,
    EPSILON,
    SIGMA,
    K_BOND_JAX,
    make_system,
)

N_A, N_B = 200, 824
N_ATOMS = N_A + N_B
TEMPERATURE_K = 188.0


def make_positions_1024(seed: int, rho_star: float):
    """Reproduce the supplied 10 x 10 x 11 site initialization."""
    length = SIGMA * (N_ATOMS / rho_star) ** (1.0 / 3.0)
    shape = np.array([10, 10, 11], dtype=np.int32)
    lattice = np.stack(
        np.meshgrid(*(np.arange(int(n)) for n in shape), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(lattice))[:N_ATOMS]
    positions = (lattice[selected] + 0.5) * (length / shape)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    return jnp.asarray(positions, dtype=dtype), jnp.full((3,), length, dtype=dtype)


def write_rows(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("mc", "rate"), required=True)
    p.add_argument("--rho-star", type=float, default=0.70)
    p.add_argument("--steps", type=int, default=15_000_000)
    p.add_argument("--check-every", type=int, default=10_000)
    p.add_argument("--dt-ps", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--mass", type=float, default=4.0,
                   help="Numerical JAX-MD mass; use identical value in MC/rate comparisons")
    p.add_argument("--association-cutoff-a", type=float, default=5.45,
                   help="Provisional RDF-informed cutoff, not specified numerically in paper")
    p.add_argument("--v0-kjmol", type=float, choices=(-0.4144, -1.9572, -3.5),
                   default=-1.9572)
    p.add_argument("--association-rate-ps", type=float, default=None)
    p.add_argument("--dissociation-rate-ps", type=float, default=None)
    p.add_argument("--relaxation-time-ps", type=float, default=1.0)
    p.add_argument("--mc-energy-mode", choices=("local_solvation", "full"),
                   default="local_solvation")
    p.add_argument("--rdf", action="store_true",
                   help="Enable your existing a2_rdf.ReactiveA2RDF observer")
    p.add_argument("--rdf-bin-width", type=float, default=0.1)
    p.add_argument("--rdf-r-max", type=float, default=14.0)
    p.add_argument("--rdf-equilibration-ps", type=float, default=5000.0)
    p.add_argument("--rdf-sample-every-checks", type=int, default=1)
    p.add_argument("--dump", action="store_true",
                   help="Enable existing a2_lammps_dump.py writer")
    p.add_argument("--dump-every-checks", type=int, default=1)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    for name in ("rho_star", "dt_ps", "mass", "association_cutoff_a", "relaxation_time_ps"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            p.error(f"{name.replace('_', '-')} must be finite and positive")
    if args.steps < 1 or args.check_every < 1 or args.dump_every_checks < 1:
        p.error("steps, check-every, and dump-every-checks must be positive")
    if args.rdf:
        if (args.rdf_bin_width <= 0 or args.rdf_r_max <= 0
                or args.rdf_equilibration_ps < 0 or args.rdf_sample_every_checks < 1):
            p.error("Invalid RDF configuration")
    if args.mode == "rate":
        if args.association_rate_ps is None or args.dissociation_rate_ps is None:
            p.error("Rate mode requires --association-rate-ps and --dissociation-rate-ps")
        for name, rate in (("association", args.association_rate_ps),
                           ("dissociation", args.dissociation_rate_ps)):
            if not np.isfinite(rate) or rate < 0 or rate * args.check_every * args.dt_ps > 1:
                p.error(f"{name} rate must satisfy 0 <= rate*check_every*dt <= 1")
    elif args.association_rate_ps is not None or args.dissociation_rate_ps is not None:
        p.error("Do not supply rate constants in MC mode")

    positions, box = make_positions_1024(args.seed, args.rho_star)
    L = float(box[0])
    if L <= 18.0:
        p.error("Cubic box must exceed 18 A for the 9 A LJ cutoff")
    if args.rdf and args.rdf_r_max > L / 2:
        p.error("RDF maximum radius cannot exceed half the box length")
    init_hash = hashlib.sha256(np.asarray(positions, dtype=np.float32).tobytes()).hexdigest()
    _, shift_fn = space.periodic(box)
    sys = make_system(N_ATOMS)
    ff = build_forcefield(
        positions, box,
        bond_idx=np.asarray(sys.bonds[0]), k_b=np.asarray(sys.bonds[1]),
        r0=np.asarray(sys.bonds[2]), angle_idx=np.asarray(sys.angles[0]),
        k_theta=np.asarray(sys.angles[1]), theta0=np.asarray(sys.angles[2]),
        torsions=tuple(np.asarray(x) for x in sys.torsions),
        impropers=tuple(np.asarray(x) for x in sys.impropers),
        charges=np.asarray(sys.charges), sigmas=np.asarray(sys.sigmas),
        epsilons=np.asarray(sys.epsilons), molecule_id=np.asarray(sys.molecule_id),
        r_cut=9.0, dr_threshold=0.5, use_shift_lj=True, energy_offset=0.0,
    )
    reaction = A2Reaction(
        a_indices=np.arange(N_A, dtype=np.int32),
        k_bond_jax=K_BOND_JAX, r0_bond=1.5,
        association_cutoff=args.association_cutoff_a,
        bond_energy_offset=args.v0_kjmol / 4.184,
        association_displacement=0.25, dissociation_displacement=0.25,
    )
    cfg = SimConfig(
        steps=args.steps, check_every=args.check_every,
        max_events=args.steps, dt=args.dt_ps,
        temperature_k=TEMPERATURE_K, kb_real=KB_REAL, tau_T=0.5,
    )
    rdf = None
    if args.rdf:
        try:
            from a2_rdf import ReactiveA2RDF
        except ImportError as exc:
            p.error(f"--rdf requires your original a2_rdf.py: {exc}")
        rdf = ReactiveA2RDF(
            box=box, n_a=N_A, n_atoms=N_ATOMS,
            bin_width=args.rdf_bin_width, r_max=args.rdf_r_max,
            equilibration_ps=args.rdf_equilibration_ps,
            sample_every_checks=args.rdf_sample_every_checks,
        )
    writer = None
    if args.dump:
        try:
            from a2_lammps_dump import write_lammps_dump_frame
        except ImportError as exc:
            p.error(f"--dump requires a2_lammps_dump.py: {exc}")
        writer = write_lammps_dump_frame

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mode: {args.mode}; particles: {N_ATOMS} ({N_A} A, {N_B} B)", flush=True)
    print(f"rho*: {args.rho_star:.6f}; box length: {L:.6f} A", flush=True)
    print(f"Active MD: {args.steps * args.dt_ps:.3f} ps; checks every "
          f"{args.check_every * args.dt_ps:.3f} ps", flush=True)
    with ExitStack() as stack:
        dump_fh = bonds_fh = frames_fh = None
        if args.dump:
            dump_fh = stack.enter_context((args.output_dir / "a2_positions.lammpstrj").open("w"))
            bonds_fh = stack.enter_context((args.output_dir / "a2_dump_bonds.csv").open("w"))
            frames_fh = stack.enter_context((args.output_dir / "a2_dump_frames.csv").open("w"))
            bonds_fh.write("frame,step,phase,atom_i_1based,atom_j_1based\n")
            frames_fh.write("frame,step,md_time_ps,phase,n_a2\n")
        result = run_md_nvt_a2(
            jax.random.PRNGKey(args.seed),
            cfg=cfg, init_positions=positions,
            masses=jnp.full((N_ATOMS,), args.mass, dtype=positions.dtype),
            box=box, shift_fn=shift_fn, ff=ff, sys=sys, reaction=reaction,
            reaction_mode="metropolis" if args.mode == "mc" else "rate",
            association_rate_ps=args.association_rate_ps,
            dissociation_rate_ps=args.dissociation_rate_ps,
            rng=np.random.default_rng(args.seed),
            relaxation_dt_ps=0.00025,
            relaxation_time_ps=args.relaxation_time_ps,
            relaxation_tau_ps=0.01,
            mc_energy_mode=args.mc_energy_mode,
            rdf_observer=rdf.observe if rdf is not None else None,
            dump_file=dump_fh, dump_writer=writer,
            atom_ids=np.arange(1, N_ATOMS + 1, dtype=np.int32) if args.dump else None,
            atom_types=np.where(np.arange(N_ATOMS) < N_A, 1, 2).astype(np.int32)
                       if args.dump else None,
            dump_every_checks=args.dump_every_checks,
            dump_bonds_file=bonds_fh, dump_frames_file=frames_fh,
        )

    sample_rows = [asdict(s) for s in result.samples]
    write_rows(args.output_dir / "samples.csv", sample_rows, list(sample_rows[0]))
    event_fields = ["step", "reaction_type", "i", "j", "distance_a", "probability",
                    "delta_energy_kcal_mol", "reaction_time_ps", "relaxation_end_time_ps"]
    event_rows = [dict(step=e.step, reaction_type=e.event.reaction_type,
                       i=e.event.i, j=e.event.j, distance_a=e.event.distance,
                       probability=e.event.probability,
                       delta_energy_kcal_mol=getattr(e.event, "delta_energy", None),
                       reaction_time_ps=e.reaction_time_ps,
                       relaxation_end_time_ps=e.relaxation_end_time_ps)
                  for e in result.event_history]
    write_rows(args.output_dir / "events.csv", event_rows, event_fields)
    attempt_fields = ["step", "reaction_type", "i", "j", "distance_a", "accepted",
                      "probability", "delta_energy_kcal_mol", "reason"]
    attempt_rows = [dict(step=r.step, reaction_type=r.attempt.reaction_type,
                         i=r.attempt.i, j=r.attempt.j, distance_a=r.attempt.distance,
                         accepted=r.attempt.accepted, probability=r.attempt.probability,
                         delta_energy_kcal_mol=r.attempt.delta_energy,
                         reason=r.attempt.reason) for r in result.mc_attempt_history]
    write_rows(args.output_dir / "mc_attempts.csv", attempt_rows, attempt_fields)
    if rdf is not None:
        rdf.write(str(args.output_dir))
    metadata = {
        "scope": "NVT PEL I algorithmic reproduction; NOT published NPT results",
        "mode": args.mode, "n_atoms": N_ATOMS, "n_a": N_A, "n_b": N_B,
        "rho_star": args.rho_star, "number_density_atoms_per_a3": N_ATOMS / L**3,
        "box_length_a": L, "temperature_k": TEMPERATURE_K,
        "lj_sigma_a": SIGMA, "lj_epsilon_kcal_mol": EPSILON,
        "lj_cutoff_a": 9.0, "lj_shifted": True,
        "v0_kj_mol": args.v0_kjmol,
        "association_cutoff_a": args.association_cutoff_a,
        "candidate_cutoff_is_provisional": True,
        "mass_numerical": args.mass, "dt_ps": args.dt_ps,
        "check_every": args.check_every, "seed": args.seed,
        "initial_positions_sha256": init_hash,
        "mc_energy_mode": args.mc_energy_mode if args.mode == "mc" else None,
        "association_rate_ps_inverse": args.association_rate_ps,
        "dissociation_rate_ps_inverse": args.dissociation_rate_ps,
        "relaxation_dt_ps": 0.00025,
        "relaxation_time_ps": args.relaxation_time_ps,
        "relaxation_tau_ps": 0.01,
        "accepted_events": result.accepted_events,
        "attempted_mc_moves": result.attempted_mc_moves,
        "rejected_mc_moves": result.rejected_mc_moves,
        "steps_done": result.steps_done,
        "rdf_enabled": args.rdf, "dump_enabled": args.dump,
        "rdf_equilibration_ps": args.rdf_equilibration_ps if args.rdf else None,
        "equilibrated_initial_configuration": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    cli()
