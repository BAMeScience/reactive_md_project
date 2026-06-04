# reactive_md/main.py
from __future__ import annotations

import argparse
import gzip
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax_md import space

from .extract_params_oplsaa import parse_lammps_data
from .config import SimConfig
from .templates_pf5 import make_pf5_template, make_lif_template
from .topology_opls import discover_pf6_and_li
from .forcefield import build_forcefield
from .reaction import SystemState, maybe_react_one_event
from .md_driver import run_md_nvt_with_reactions
from .lammps_io import write_lammps_dump_frame


def _open_text_output(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


def main(cfg: SimConfig):
    (
        positions,
        bonds,
        angles,
        torsions,
        impropers,
        nonbonded,
        molecule_id,
        box,
        masses,
        atom_types,
    ) = parse_lammps_data(cfg.data_file, cfg.settings_file)

    charges, sigmas, epsilons, pair_indices, is_14_mask = nonbonded

    positions = np.asarray(positions)
    masses = np.asarray(masses)
    atom_types = np.asarray(atom_types, dtype=np.int32)
    molecule_id = np.asarray(molecule_id, dtype=np.int32)
    box = np.asarray(box)

    atom_ids = np.arange(1, positions.shape[0] + 1, dtype=np.int32)

    print(
        f"Loaded: n_atoms={positions.shape[0]}, "
        f"bonds={bonds[0].shape[0]}, "
        f"angles={angles[0].shape[0]}"
    )

    pf6_atoms, li_atoms = discover_pf6_and_li(
        atom_types,
        molecule_id,
        p_type=cfg.p_type,
        f_type=cfg.f_type,
        li_type=cfg.li_type,
    )

    print(f"Discovered PF6 units: {pf6_atoms.shape[0]}")
    print(f"Discovered Li atoms:  {li_atoms.shape[0]}")
    print(
        "Reaction gates: "
        f"d_LiF < {cfg.r_lif_on:.3f} Å and "
        f"d_PF > {cfg.r_pf_break:.3f} Å"
    )

    if cfg.dump_file is not None:
        dump_every = cfg.dump_every if cfg.dump_every is not None else cfg.check_every
        print(f"Writing LAMMPS dump trajectory to: {cfg.dump_file}")
        print(f"Trajectory write interval: every {dump_every} steps")
        if cfg.dump_file.endswith(".gz"):
            print("Trajectory compression: gzip")

    if cfg.event_log_file is not None:
        print(f"Writing reaction event log to: {cfg.event_log_file}")
        if cfg.event_log_file.endswith(".gz"):
            print("Event log compression: gzip")

    if pf6_atoms.shape[0] > 0:
        print("First PF6 block indices:", pf6_atoms[0])
        print("First PF6 block types:", atom_types[pf6_atoms[0]])

    if li_atoms.shape[0] > 0:
        print("First Li index:", li_atoms[0], "type:", atom_types[li_atoms[0]])

    _disp_periodic, shift_fn = space.periodic(box)

    ff = build_forcefield(
        R=jnp.array(positions),
        box=box,
        bond_idx=np.array(bonds[0], dtype=np.int32),
        k_b=np.array(bonds[1], dtype=np.float32),
        r0=np.array(bonds[2], dtype=np.float32),
        angle_idx=np.array(angles[0], dtype=np.int32),
        k_theta=np.array(angles[1], dtype=np.float32),
        theta0=np.array(angles[2], dtype=np.float32),
        torsions=(
            np.array(torsions[0], dtype=np.int32),
            np.array(torsions[1], dtype=np.float32),
            np.array(torsions[2], dtype=np.int32),
            np.array(torsions[3], dtype=np.float32),
        ),
        impropers=(
            np.array(impropers[0], dtype=np.int32),
            np.array(impropers[1], dtype=np.float32),
            np.array(impropers[2], dtype=np.int32),
            np.array(impropers[3], dtype=np.float32),
        ),
        charges=np.array(charges, dtype=np.float32),
        sigmas=np.array(sigmas, dtype=np.float32),
        epsilons=np.array(epsilons, dtype=np.float32),
        molecule_id=molecule_id,
        r_cut=cfg.r_cut,
        dr_threshold=cfg.dr_threshold,
    )

    sys = SystemState(
        bonds=(
            jnp.array(bonds[0], dtype=int),
            jnp.array(bonds[1]),
            jnp.array(bonds[2]),
        ),
        angles=(
            jnp.array(angles[0], dtype=int),
            jnp.array(angles[1]),
            jnp.array(angles[2]),
        ),
        torsions=(
            jnp.array(torsions[0], dtype=int),
            jnp.array(torsions[1]),
            jnp.array(torsions[2]),
            jnp.array(torsions[3]),
        ),
        impropers=(
            jnp.array(impropers[0], dtype=int),
            jnp.array(impropers[1]),
            jnp.array(impropers[2]),
            jnp.array(impropers[3]),
        ),
        charges=jnp.array(charges),
        sigmas=jnp.array(sigmas),
        epsilons=jnp.array(epsilons),
        molecule_id=jnp.array(molecule_id, dtype=int),
        pf6_reacted=jnp.zeros((pf6_atoms.shape[0],), dtype=jnp.bool_),
    )

    pf5 = make_pf5_template()
    lif = make_lif_template()


    mc_energy_evaluator = None

    if cfg.use_mace_mc:
       from .mace_setup import build_mace_model
       from .mace_energy import build_mace_energy_system
       from .mace_mc_energy import MaceJaxEnergyEvaluator

       type_to_Z = {
        1: 6,   # C
        2: 8,   # O
        3: 8,   # O
        4: 6,   # C
        5: 1,   # H
        6: 15,  # P
        7: 9,   # F
        8: 3,   # Li
       }

       z_atomic = np.array([type_to_Z[int(t)] for t in atom_types], dtype=np.int32)

       jax_model, jax_model_config, _torch_config = build_mace_model(
          source=cfg.mace_source,
          variant=cfg.mace_variant,
       )

       (
         _mace_disp_fn,
         _mace_shift_fn,
         mace_neighbor_fn,
         mace_make_energy_fn,
       ) = build_mace_energy_system(
         jax_model=jax_model,
         jax_model_config=jax_model_config,
         z_atomic=z_atomic,
         box0=box,
         ensemble="nve",
         dr_threshold=cfg.mace_dr_threshold,
         capacity_multiplier=cfg.mace_capacity_multiplier,
       )

       mc_energy_evaluator = MaceJaxEnergyEvaluator(
         neighbor_fn=mace_neighbor_fn,
         make_energy_fn=mace_make_energy_fn,
         box0=box,
       )

       print("Metropolis energy: MACE-JAX")
    else:
       print("Metropolis energy: classical force field")

    # ------------------------------------
      # Beta depends on energy units
    # ------------------------------------

    if cfg.use_mace_mc:
       beta = 1.0 / (8.617333262e-5 * cfg.temperature_k)
    else:
       beta = 1.0 / (cfg.kb_real * cfg.temperature_k)

    def reaction_step_fn(key, R, ff_in, sys_in):
        if cfg.reaction_mode == "rate":
            return maybe_react_rate_events(
                key,
                R,
                box,
                shift_fn=shift_fn,
                ff=ff_in,
                sys=sys_in,
                pf6_atoms=pf6_atoms,
                li_atoms=li_atoms,
                atom_types=atom_types,
                pf5=pf5,
                lif=lif,
                p_type=cfg.p_type,
                f_type=cfg.f_type,
                li_type=cfg.li_type,
                r_lif_on=cfg.r_lif_on,
                r_pf_break=cfg.r_pf_break,
                r_pf_probe=cfg.r_pf_probe,
                reaction_rate_ps=cfg.reaction_rate_ps,
                reactive_interval_ps=cfg.check_every * cfg.dt,
                max_reactions_per_check=cfg.max_reactions_per_check,
            )

        return maybe_react_one_event(
            key,
            R,
            box,
            shift_fn=shift_fn,
            ff=ff_in,
            sys=sys_in,
            pf6_atoms=pf6_atoms,
            li_atoms=li_atoms,
            atom_types=atom_types,
            pf5=pf5,
            lif=lif,
            p_type=cfg.p_type,
            f_type=cfg.f_type,
            li_type=cfg.li_type,
            r_lif_on=cfg.r_lif_on,
            r_pf_break=cfg.r_pf_break,
            r_pf_probe=cfg.r_pf_probe,
            beta=beta,
            mc_energy_evaluator=mc_energy_evaluator,
        )


    key = jax.random.PRNGKey(cfg.prng_seed)

    with ExitStack() as stack:
        dump_file = None
        event_file = None

        if cfg.dump_file is not None:
            dump_file = stack.enter_context(_open_text_output(cfg.dump_file))

        if cfg.event_log_file is not None:
            event_file = stack.enter_context(_open_text_output(cfg.event_log_file))

        _result = run_md_nvt_with_reactions(
            key,
            cfg=cfg,
            init_positions=jnp.array(positions),
            masses=masses,
            shift_fn=shift_fn,
            ff=ff,
            sys=sys,
            reaction_step_fn=reaction_step_fn,
            box=box,
            atom_ids=atom_ids,
            atom_types=atom_types,
            dump_file=dump_file,
            dump_writer=write_lammps_dump_frame if dump_file is not None else None,
            event_file=event_file,
        )


def cli():
  
    default_cfg = SimConfig()

    parser = argparse.ArgumentParser(description="Reactive OPLS-AA MD using JAX-MD")

    parser.add_argument("--data", required=True, help="LAMMPS data file")
    parser.add_argument("--settings", required=True, help="LAMMPS settings file")

    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--check-every", type=int, default=None)

    parser.add_argument("--reaction-mode",choices=["metropolis", "rate"],
     default=default_cfg.reaction_mode,
    )
    parser.add_argument(
      "--reaction-rate-ps", type=float, default=default_cfg.reaction_rate_ps,
      help="Rate-based reaction probability parameter k in ps^-1.",
    )
    parser.add_argument("--max-reactions-per-check", type=int, default=default_cfg.max_reactions_per_check,)

    parser.add_argument("--r-lif-on", type=float, default=None)
    parser.add_argument("--r-pf-break", type=float, default=None)
    parser.add_argument("--r-pf-probe", type=float, default=None)

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--dump-file",
        default=None,
        help="LAMMPS dump trajectory output file. Use .gz for gzip compression.",
    )
    parser.add_argument(
        "--dump-every",
        type=int,
        default=None,
        help="Write trajectory every N MD steps. Defaults to check_every.",
    )
    parser.add_argument(
        "--event-log-file",
        default=None,
        help="CSV event log for accepted topology-changing reactions. Use .gz for gzip.",
    )

    parser.add_argument(
    "--use-mace-mc",
    action="store_true",
    help="Use MACE-JAX energies for Metropolis acceptance only.",
    ) 

    parser.add_argument("--mace-source", default=default_cfg.mace_source)
    parser.add_argument("--mace-variant", default=default_cfg.mace_variant)
    parser.add_argument("--mace-dr-threshold", type=float, default=default_cfg.mace_dr_threshold)
    parser.add_argument(
      "--mace-capacity-multiplier",
      type=float,
      default=default_cfg.mace_capacity_multiplier,
    )

    args = parser.parse_args()

    cfg = SimConfig(
        data_file=str(Path(args.data).expanduser().resolve()),
        settings_file=str(Path(args.settings).expanduser().resolve()),
        steps=args.steps if args.steps is not None else default_cfg.steps,
        check_every=args.check_every if args.check_every is not None else default_cfg.check_every,
        r_lif_on=args.r_lif_on if args.r_lif_on is not None else default_cfg.r_lif_on,
        r_pf_break=args.r_pf_break if args.r_pf_break is not None else default_cfg.r_pf_break,
        r_pf_probe=args.r_pf_probe if args.r_pf_probe is not None else default_cfg.r_pf_probe,
        temperature_k=args.temperature if args.temperature is not None else default_cfg.temperature_k,
        prng_seed=args.seed if args.seed is not None else default_cfg.prng_seed,
        dump_file=(
            str(Path(args.dump_file).expanduser().resolve())
            if args.dump_file is not None
            else None
        ),
        dump_every=args.dump_every,
        event_log_file=(
            str(Path(args.event_log_file).expanduser().resolve())
            if args.event_log_file is not None
            else None
        ),

        use_mace_mc=args.use_mace_mc,
        mace_source=args.mace_source,
        mace_variant=args.mace_variant,
        mace_dr_threshold=args.mace_dr_threshold,
        mace_capacity_multiplier=args.mace_capacity_multiplier

    )

    main(cfg)


if __name__ == "__main__":
    cli()
