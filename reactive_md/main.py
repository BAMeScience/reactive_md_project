# reactive_md/main.py
from __future__ import annotations
import numpy as np
import argparse
from pathlib import Path
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

def main(cfg: SimConfig):
    positions, bonds, angles, torsions, impropers, nonbonded, molecule_id, box, masses, atom_types = parse_lammps_data(
        cfg.data_file, cfg.settings_file
    )

    angle_theta0 = np.array(angles[2])
    if angle_theta0.size:
       print("[debug] theta0 stats:", angle_theta0.min(), angle_theta0.max())
    else:
       print("[debug] theta0 stats: (no angles)")

    charges, sigmas, epsilons, pair_indices, is_14_mask = nonbonded

    n_atoms = positions.shape[0]
    print(f"Loaded: n_atoms={n_atoms}, bonds={bonds[0].shape[0]}, angles={angles[0].shape[0]}")

    # Discover PF6 and Li
    pf6_atoms, li_atoms = discover_pf6_and_li(
        atom_types, molecule_id,
        p_type=cfg.p_type, f_type=cfg.f_type, li_type=cfg.li_type
    )
    if pf6_atoms.shape[0] > 0:
        print("First PF6 block indices:", pf6_atoms[0])
        print("First PF6 block types:", np.array(atom_types)[pf6_atoms[0]])
    if li_atoms.shape[0] > 0:
        print("First Li index:", li_atoms[0], "type:", np.array(atom_types)[li_atoms[0]])

    disp_periodic, shift_fn = space.periodic(box)

    # Build initial forcefield bundle
    ff = build_forcefield(
        R=jnp.array(positions),
        box=box,
        bond_idx=np.array(bonds[0], dtype=np.int32), k_b=np.array(bonds[1], dtype=np.float32), r0=np.array(bonds[2], dtype=np.float32),
        angle_idx=np.array(angles[0], dtype=np.int32), k_theta=np.array(angles[1], dtype=np.float32), theta0=np.array(angles[2], dtype=np.float32),
        torsions=(np.array(torsions[0], dtype=np.int32), np.array(torsions[1], dtype=np.float32), np.array(torsions[2], dtype=np.int32), np.array(torsions[3], dtype=np.float32)),
        impropers=(np.array(impropers[0], dtype=np.int32), np.array(impropers[1], dtype=np.float32), np.array(impropers[2], dtype=np.int32), np.array(impropers[3], dtype=np.float32)),
        charges=np.array(charges, dtype=np.float32),
        sigmas=np.array(sigmas, dtype=np.float32),
        epsilons=np.array(epsilons, dtype=np.float32),
        molecule_id=np.array(molecule_id, dtype=np.int32),
        r_cut=cfg.r_cut,
        dr_threshold=cfg.dr_threshold,
    )

    # SystemState wrapper (things that can change after reaction)
    sys = SystemState(
        bonds=(jnp.array(bonds[0], dtype=int), jnp.array(bonds[1]), jnp.array(bonds[2])),
        angles=(jnp.array(angles[0], dtype=int), jnp.array(angles[1]), jnp.array(angles[2])),
        torsions=(jnp.array(torsions[0], dtype=int), jnp.array(torsions[1]), jnp.array(torsions[2]), jnp.array(torsions[3])),
        impropers=(jnp.array(impropers[0], dtype=int), jnp.array(impropers[1]), jnp.array(impropers[2]), jnp.array(impropers[3])),
        charges=jnp.array(charges),
        sigmas=jnp.array(sigmas),
        epsilons=jnp.array(epsilons),
        molecule_id=jnp.array(molecule_id, dtype=int),
        pf6_reacted=jnp.zeros((pf6_atoms.shape[0],), dtype=jnp.bool_),
    )

    pf5 = make_pf5_template()
    lif = make_lif_template()

    beta = 1.0 / (cfg.kb_real * cfg.temperature_k)

    def reaction_step_fn(key, R, ff_in, sys_in):
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
            r_on=cfg.r_on,
            beta=beta,
        )

    key = jax.random.PRNGKey(cfg.prng_seed)

    _result = run_md_nvt_with_reactions(
        key,
        cfg=cfg,
        init_positions=jnp.array(positions),
        masses=masses,
        shift_fn=shift_fn,
        ff=ff,
        sys=sys,
        reaction_step_fn=reaction_step_fn,
    )


def cli():
    parser = argparse.ArgumentParser(description="Reactive OPLS-AA MD (JAX-MD)")
    parser.add_argument(
        "--data",
        required=True,
        help="LAMMPS data file",
    )
    parser.add_argument(
        "--settings",
        required=True,
        help="LAMMPS settings file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of MD steps (override config)",
    )
    args = parser.parse_args()

    cfg = SimConfig(
        data_file=str(Path(args.data).expanduser().resolve()),
        settings_file=str(Path(args.settings).expanduser().resolve()),
        steps=args.steps if args.steps is not None else SimConfig.steps,
    )

    main(cfg)

if __name__ == "__main__":
    cli()

