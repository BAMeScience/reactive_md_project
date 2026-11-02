"""NVT algorithmic reproduction of rs@md[MC] or rs@md[rate] with PEL I.

This is NOT the authors' NPT/PEL II numerical benchmark. Number density is
mandatory because the paper does not provide an equilibrium NPT density.

Example (set CANDIDATE_CUTOFF_A from an A-A RDF at your NVT density;
supplied rates are illustrative, not PEL II-derived):
    python -m scripts.run_a2_paper_nvt --mode mc --density-number-a3 0.0045 --association-cutoff-a $CANDIDATE_CUTOFF_A \\
        --steps 100 --check-every 10 --output-dir results/paper_mc_smoke
    python -m scripts.run_a2_paper_nvt --mode rate --density-number-a3 0.0045 --association-cutoff-a $CANDIDATE_CUTOFF_A \\
        --association-rate-ps 0.00015 --dissociation-rate-ps 0.00006 \\
        --steps 100 --check-every 10 --output-dir results/paper_rate_smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
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
from .run_a2_small import (
    KB_REAL, EPSILON, SIGMA, K_BOND_JAX, make_positions, make_system,
)

PAPER_V0_KJMOL = (-0.4144, -1.9572, -3.5000)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('mc', 'rate'), required=True)
    p.add_argument('--density-number-a3', type=float, required=True,
                   help='Explicit N/V in atoms per angstrom^3')
    p.add_argument('--v0-kjmol', type=float, choices=PAPER_V0_KJMOL,
                   default=-1.9572)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--check-every', type=int, default=10)
    p.add_argument('--dt-ps', type=float, default=0.001)
    p.add_argument('--mass', type=float, default=4.0,
                   help='JAX-MD integrator mass (numerical units; calibrate separately)')
    p.add_argument('--association-rate-ps', type=float)
    p.add_argument('--dissociation-rate-ps', type=float)
    p.add_argument('--association-cutoff-a', type=float, required=True,
                   help='User-chosen A-A candidate cutoff (angstrom); not published as 2.7 A')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--relaxation-time-ps', type=float, default=1.0,
                   help='Paper setting: 1 ps; shorter values are diagnostic only')
    p.add_argument('--mc-energy-mode', choices=('local_solvation', 'full'),
                   default='local_solvation')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.steps < 1 or args.check_every < 1 or args.dt_ps <= 0 or args.mass <= 0:
        p.error('steps, check-every, dt-ps, and mass must be positive')
    if not np.isfinite(args.density_number_a3) or args.density_number_a3 <= 0:
        p.error('density-number-a3 must be positive and finite')
    if not np.isfinite(args.association_cutoff_a) or args.association_cutoff_a <= 0:
        p.error('association-cutoff-a must be positive and finite')
    if not np.isfinite(args.relaxation_time_ps) or args.relaxation_time_ps <= 0:
        p.error('relaxation-time-ps must be finite and positive')
    if args.mode == 'rate':
        if args.association_rate_ps is None or args.dissociation_rate_ps is None:
            p.error('rate mode requires association and dissociation rates')
        for name, rate in (('association', args.association_rate_ps),
                           ('dissociation', args.dissociation_rate_ps)):
            if not np.isfinite(rate) or rate < 0 or rate * args.check_every * args.dt_ps > 1:
                p.error(f'{name} rate must satisfy 0 <= Gamma*tau <= 1')

    R, box = make_positions()
    n = int(R.shape[0])
    volume = n / args.density_number_a3
    L = volume ** (1. / 3.)
    if L <= 18.0:
        p.error('For the 9 A LJ cutoff, cubic box length must exceed 18 A; lower density')
    scale = L / float(box[0])
    R = R * scale
    box = jnp.asarray([L, L, L], dtype=R.dtype)
    # Place the one intentionally reactive pair at a known distance after scaling.
    R = R.at[1, 0].set(R[0, 0] + 2.65)
    initial_position_sha256 = hashlib.sha256(
        np.asarray(R, dtype=np.float32).tobytes()
    ).hexdigest()
    _, shift_fn = space.periodic(box)
    sys = make_system(n)
    ff = build_forcefield(
        R, box,
        bond_idx=np.asarray(sys.bonds[0]), k_b=np.asarray(sys.bonds[1]),
        r0=np.asarray(sys.bonds[2]), angle_idx=np.asarray(sys.angles[0]),
        k_theta=np.asarray(sys.angles[1]), theta0=np.asarray(sys.angles[2]),
        torsions=tuple(np.asarray(x) for x in sys.torsions),
        impropers=tuple(np.asarray(x) for x in sys.impropers),
        charges=np.asarray(sys.charges), sigmas=np.asarray(sys.sigmas),
        epsilons=np.asarray(sys.epsilons), molecule_id=np.asarray(sys.molecule_id),
        r_cut=9.0, dr_threshold=0.5, use_shift_lj=True,
        energy_offset=0.0,
    )
    reaction = A2Reaction(
        a_indices=np.arange(16, dtype=np.int32),
        k_bond_jax=K_BOND_JAX, r0_bond=1.5,
        association_cutoff=args.association_cutoff_a,
        bond_energy_offset=args.v0_kjmol / 4.184,
        association_displacement=0.25, dissociation_displacement=0.25,
    )
    cfg = SimConfig(steps=args.steps, check_every=args.check_every,
                    max_events=args.steps, dt=args.dt_ps,
                    temperature_k=188.0, kb_real=KB_REAL, tau_T=0.5)
    result = run_md_nvt_a2(
        jax.random.PRNGKey(args.seed), cfg=cfg, init_positions=R,
        masses=jnp.full((n,), args.mass), box=box, shift_fn=shift_fn,
        ff=ff, sys=sys, reaction=reaction,
        reaction_mode='metropolis' if args.mode == 'mc' else 'rate',
        association_rate_ps=args.association_rate_ps,
        dissociation_rate_ps=args.dissociation_rate_ps,
        rng=np.random.default_rng(args.seed),
        relaxation_dt_ps=0.00025,
        relaxation_time_ps=args.relaxation_time_ps,
        relaxation_tau_ps=0.01,
        mc_energy_mode=args.mc_energy_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'samples.csv').open('w', newline='') as f:
        rows = [asdict(s) for s in result.samples]
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / 'events.csv').open('w', newline='') as f:
        rows = [{'step': r.step, 'type': r.event.reaction_type,
                 'i': r.event.i, 'j': r.event.j,
                 'distance_a': r.event.distance,
                 'probability': r.event.probability}
                for r in result.event_history]
        writer = csv.DictWriter(f, fieldnames=['step','type','i','j','distance_a','probability'])
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / 'mc_attempts.csv').open('w', newline='') as f:
        fields = ['step', 'reaction_type', 'i', 'j', 'distance_a',
                  'accepted', 'probability', 'delta_energy_kcal_mol', 'reason']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in result.mc_attempt_history:
            a = record.attempt
            writer.writerow(dict(step=record.step, reaction_type=a.reaction_type,
                                 i=a.i, j=a.j, distance_a=a.distance,
                                 accepted=a.accepted, probability=a.probability,
                                 delta_energy_kcal_mol=a.delta_energy,
                                 reason=a.reason))
    metadata = {
        'scope': 'NVT PEL I algorithmic reproduction; NOT published NPT results',
        'mode': args.mode, 'n_atoms': n, 'n_a': 16, 'n_b': 16,
        'number_density_atoms_per_a3': args.density_number_a3,
        'box_length_a': L, 'volume_a3': volume,
        'temperature_k': 188., 'lj_sigma_a': SIGMA,
        'lj_epsilon_kcal_mol': EPSILON, 'lj_shifted': True,
        'v0_kj_mol': args.v0_kjmol, 'association_cutoff_a': args.association_cutoff_a,
        'candidate_cutoff_source': 'user-specified; not reported numerically in the paper',
        'candidate_cutoff_is_provisional': True,
        'rate_inputs_are_externally_supplied': args.mode == 'rate',
        'mass_numerical': args.mass, 'dt_ps': args.dt_ps,
        'initial_positions_sha256': initial_position_sha256,
        'mc_energy_mode': args.mc_energy_mode if args.mode == 'mc' else None,
        'association_rate_ps_inverse': args.association_rate_ps,
        'dissociation_rate_ps_inverse': args.dissociation_rate_ps,
        'rate_probability_association': (args.association_rate_ps * args.check_every * args.dt_ps if args.mode == 'rate' else None),
        'rate_probability_dissociation': (args.dissociation_rate_ps * args.check_every * args.dt_ps if args.mode == 'rate' else None),
        'check_every': args.check_every, 'seed': args.seed,
        'relaxation_dt_ps': 0.00025, 'relaxation_time_ps': args.relaxation_time_ps,
        'relaxation_tau_ps': 0.01,
        'relaxation_matches_paper': args.relaxation_time_ps == 1.0,
        'accepted_events': result.accepted_events,
        'attempted_mc_moves': result.attempted_mc_moves,
        'rejected_mc_moves': result.rejected_mc_moves,
    }
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
