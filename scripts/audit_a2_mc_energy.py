"""Phase 7: fixed-configuration rs@md A2 MC energy-accounting audit.

Run with JAX-MD installed, preferably with JAX_ENABLE_X64=True:
  python -m scripts.audit_a2_mc_energy --output-dir results/phase7_energy

This is a controlled diagnostic, not an equilibrium or detailed-balance proof.
No MC/rate algorithm is changed. Four independent Langevin relaxations are
performed: unbonded control, association trial, bonded control, dissociation
trial. Each control/trial pair starts at identical coordinates, velocities,
and PRNG key. The two *different* starting topologies are not paired to each
other. A control-adjusted energy is diagnostic only, NOT the MC acceptance rule.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from reactive_md.forcefield import build_forcefield
from reactive_md.reaction import build_trial_forcefield
from reactive_md.relaxation import langevin_relax_with_nlist
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_energy import local_solvation_energy
from reactive_md.reactions.a2_energy_reference import spectator_lj_reference
from reactive_md.reactions.a2_metropolis import _finalize_trial, _potential, _valid_nlist
from reactive_md.reactions.a2_rate import make_a2_system_state_from_trial, translate_reacting_pair
from .run_a2_small import KB_REAL, K_BOND_JAX, make_positions, make_system


def build_initial_ff(R, box, sys):
    return build_forcefield(
        R, box, bond_idx=np.asarray(sys.bonds[0]),
        k_b=np.asarray(sys.bonds[1]), r0=np.asarray(sys.bonds[2]),
        angle_idx=np.asarray(sys.angles[0]),
        k_theta=np.asarray(sys.angles[1]), theta0=np.asarray(sys.angles[2]),
        torsions=tuple(np.asarray(x) for x in sys.torsions),
        impropers=tuple(np.asarray(x) for x in sys.impropers),
        charges=np.asarray(sys.charges), sigmas=np.asarray(sys.sigmas),
        epsilons=np.asarray(sys.epsilons), molecule_id=np.asarray(sys.molecule_id),
        r_cut=9.0, dr_threshold=0.5, use_shift_lj=True, energy_offset=0.0,
    )


def energies(ff, R, sys, box, pair, sigma, epsilon):
    full = _potential(ff, R, _valid_nlist(ff, R))
    local = float(local_solvation_energy(ff, R, sys, pair, box))
    spectator_from_partition = full - local
    spectator_independent = spectator_lj_reference(
        R, box, pair, sigma=sigma, epsilon=epsilon,
    )
    offset = float(np.asarray(ff.energy_fn(R, _valid_nlist(ff, R)).get('offset', 0.0)))
    return {
        'full': full, 'local_solvation': local,
        'spectator_from_partition': spectator_from_partition,
        'spectator_independent_lj': spectator_independent,
        'spectator_residual': spectator_from_partition - spectator_independent,
        'topology_offset': offset,
    }


def energy_change(after, before, key):
    return float(after[key] - before[key])


def run_case(kind, R, box, system, ff, reaction, *, seed, mass,
             relaxation_time_ps, temperature_k, sigma, epsilon):
    disp_fn, shift_fn = space.periodic(box)
    pair = (0, 1)
    if kind == 'association':
        candidates = reaction.find_association_candidates(R, disp_fn, system=system)
    else:
        candidates = reaction.find_dissociation_candidates(R, disp_fn, system=system)
    selected = [c for c in candidates if {int(c.i), int(c.j)} == set(pair)]
    if len(selected) != 1:
        raise RuntimeError(f'{kind}: expected exactly one candidate for pair {pair}')
    candidate = selected[0]
    trial = (reaction.build_association_trial(system, candidate)
             if kind == 'association' else
             reaction.build_dissociation_trial(system, candidate))
    trial = _finalize_trial(trial, reaction, len(R))
    trial_system = make_a2_system_state_from_trial(trial, system)
    trial_R = translate_reacting_pair(
        R, *pair, disp_fn=disp_fn, shift_fn=shift_fn,
        displacement_per_atom=(reaction.association_displacement if kind == 'association'
                               else reaction.dissociation_displacement),
        move_apart=(kind == 'dissociation'),
    )
    trial_ff = build_trial_forcefield(trial_R, box, trial, ff)
    before = energies(ff, R, system, box, pair, sigma, epsilon)
    translated = energies(trial_ff, trial_R, trial_system, box, pair, sigma, epsilon)

    key = jax.random.PRNGKey(seed)
    v = jnp.zeros_like(R)
    masses = jnp.full((len(R),), mass)
    relax_kwargs = dict(shift_fn=shift_fn, masses=masses,
                        temperature_k=temperature_k, kb_real=KB_REAL,
                        dt_ps=0.00025, relaxation_time_ps=relaxation_time_ps,
                        thermostat_tau_ps=0.01)
    # Control: unchanged topology and positions, but identical relaxation
    # integrator, initial velocities and PRNG seed.
    _, control_R, _, _ = langevin_relax_with_nlist(
        key, R, v, ff=ff, **relax_kwargs,
    )
    control = energies(ff, control_R, system, box, pair, sigma, epsilon)
    _, relaxed_R, _, _ = langevin_relax_with_nlist(
        key, trial_R, v, ff=trial_ff, **relax_kwargs,
    )
    relaxed = energies(trial_ff, relaxed_R, trial_system, box, pair, sigma, epsilon)

    # Independent check that the offset enters exactly once at fixed coordinates.
    zero_offset_trial = dict(trial, energy_offset=0.0)
    ff_zero = build_trial_forcefield(relaxed_R, box, zero_offset_trial, ff)
    no_offset_full = _potential(ff_zero, relaxed_R, _valid_nlist(ff_zero, relaxed_R))
    offset_residual = (relaxed['full'] - no_offset_full) - float(trial['energy_offset'])

    result = {
        'reaction_type': kind,
        'reacting_pair': list(pair),
        'initial_pair_distance_a': float(candidate.distance),
        'translated_pair_distance_a': float(np.linalg.norm(
            np.asarray(disp_fn(trial_R[0], trial_R[1])))),
        'before': before,
        'translated_trial': translated,
        'control_relaxed': control,
        'trial_relaxed': relaxed,
        'delta_full_mc': energy_change(relaxed, before, 'full'),
        'delta_local_solvation_mc': energy_change(relaxed, before, 'local_solvation'),
        'delta_spectator': energy_change(relaxed, before, 'spectator_independent_lj'),
        'delta_local_control': energy_change(control, before, 'local_solvation'),
        'delta_local_trial_after_translation': energy_change(
            relaxed, translated, 'local_solvation'),
        'delta_local_control_adjusted': energy_change(
            relaxed, control, 'local_solvation'),
        'expected_trial_offset': float(trial['energy_offset']),
        'offset_once_residual': float(offset_residual),
    }
    for label, entry in (('before', before), ('translated_trial', translated),
                         ('control_relaxed', control), ('trial_relaxed', relaxed)):
        for k, val in entry.items():
            if not math.isfinite(val):
                raise FloatingPointError(f'{kind}/{label}/{k} is not finite')
    return result


def summarize(results, settings, tolerance):
    lines = [
        '# Phase 7 — fixed-configuration A2 MC energy audit', '',
        'This is a **diagnostic**, not an equilibrium simulation or proof of detailed balance.',
        'No MC acceptance or reaction algorithm was modified.', '',
        f"Relaxation: {settings['relaxation_time_ps']} ps at dt=0.00025 ps; "
        'Langevin tau=0.01 ps. Initial velocities are zero.',
        'Each control/trial pair uses the same starting positions, velocities and JAX PRNG seed.',
        'Association and dissociation use **different** fixed initial topologies and distances.',
        '',
        '| Starting state | MC ΔU local+solvation | MC ΔU full | '
        'Control-only ΔU local | Trial−control local | Offset residual |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for x in results:
        lines.append(
            f"| {x['reaction_type']} | {x['delta_local_solvation_mc']:.6f} | "
            f"{x['delta_full_mc']:.6f} | {x['delta_local_control']:.6f} | "
            f"{x['delta_local_control_adjusted']:.6f} | {x['offset_once_residual']:.3g} |"
        )
    lines.extend([
        '', 'All energies in kcal/mol. The MC ΔU is trial-relaxed minus **unrelaxed** starting energy.',
        'The trial−control column is **not** an alternative MC acceptance criterion.',
        'Independent spectator LJ values and partition residuals are in `energy_audit.json`.',
        f"Reference-check tolerance: {tolerance} kcal/mol (numerical diagnostic, not paper accuracy).",
        '', '### Checks', '',
    ])
    for x in results:
        worst = max(abs(x[k]['spectator_residual']) for k in
                    ('before', 'translated_trial', 'control_relaxed', 'trial_relaxed'))
        passed = worst <= tolerance and abs(x['offset_once_residual']) <= tolerance
        lines.append(f"- {x['reaction_type']}: {'PASS' if passed else 'INVESTIGATE'}; "
                     f'max spectator residual={worst:.6g}; '
                     f"offset residual={x['offset_once_residual']:.6g} kcal/mol")
    lines.extend([
        '', '### Limits', '',
        '- The NumPy spectator reference assumes monatomic, uncharged, unbonded spectators.',
        '- Both branches use a constructed, unequilibrated 16 A + 16 B lattice.',
        '- A matched control measures relaxation effects but does not establish MC detailed balance.',
        '- The initial state here is not the step-10 or step-20 state of the Phase 6 trajectory.',
        '- A negative trial energy does not establish an equilibrium free-energy difference.',
    ])
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--density-number-a3', type=float, default=0.0045)
    p.add_argument('--association-cutoff-a', type=float, default=3.0)
    p.add_argument('--v0-kjmol', type=float, choices=(-0.4144, -1.9572, -3.5),
                   default=-1.9572)
    p.add_argument('--relaxation-time-ps', type=float, default=1.0)
    p.add_argument('--mass', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--output-dir', type=Path, default=Path('results/phase7_energy'))
    args = p.parse_args()
    for name in ('density_number_a3', 'association_cutoff_a', 'relaxation_time_ps', 'mass'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f'{name} must be positive and finite')
    n_relax = round(args.relaxation_time_ps / 0.00025)
    if n_relax < 1 or not np.isclose(n_relax * 0.00025, args.relaxation_time_ps,
                                    rtol=0, atol=1e-12):
        p.error('relaxation-time-ps must be a positive multiple of 0.00025 ps')
    R, original_box = make_positions()
    n = len(R)
    L = (n / args.density_number_a3) ** (1.0 / 3.0)
    if L <= 18.0:
        p.error('box must exceed twice the 9 A cutoff')
    R = R * (L / float(original_box[0]))
    box = jnp.asarray([L, L, L], dtype=R.dtype)
    R = R.at[1, 0].set(R[0, 0] + 2.65)
    unbonded_system = make_system(n)
    unbonded_ff = build_initial_ff(R, box, unbonded_system)
    reaction = A2Reaction(
        a_indices=np.arange(16, dtype=np.int32),
        k_bond_jax=K_BOND_JAX, r0_bond=1.5,
        association_cutoff=args.association_cutoff_a,
        bond_energy_offset=args.v0_kjmol / 4.184,
        association_displacement=0.25, dissociation_displacement=0.25,
    )
    sigma = float(np.asarray(unbonded_system.sigmas)[0])
    epsilon = float(np.asarray(unbonded_system.epsilons)[0])
    association = run_case(
        'association', R, box, unbonded_system, unbonded_ff, reaction,
        seed=args.seed, mass=args.mass, relaxation_time_ps=args.relaxation_time_ps,
        temperature_k=188.0, sigma=sigma, epsilon=epsilon,
    )
    # A separate fixed bonded starting state at the bond equilibrium distance.
    # It is NOT the association branch's relaxed endpoint.
    R_bonded = R.at[1, 0].set(R[0, 0] + 1.5)
    base_ff = build_initial_ff(R_bonded, box, unbonded_system)
    disp_fn, _ = space.periodic(box)
    candidates = reaction.find_association_candidates(
        R_bonded, disp_fn, system=unbonded_system)
    pair_candidates = [c for c in candidates if {int(c.i), int(c.j)} == {0, 1}]
    if len(pair_candidates) != 1:
        raise RuntimeError('Cannot construct bonded dissociation reference')
    bonded_trial = _finalize_trial(
        reaction.build_association_trial(unbonded_system, pair_candidates[0]),
        reaction, n,
    )
    bonded_system = make_a2_system_state_from_trial(bonded_trial, unbonded_system)
    bonded_ff = build_trial_forcefield(R_bonded, box, bonded_trial, base_ff)
    dissociation = run_case(
        'dissociation', R_bonded, box, bonded_system, bonded_ff, reaction,
        seed=args.seed, mass=args.mass, relaxation_time_ps=args.relaxation_time_ps,
        temperature_k=188.0, sigma=sigma, epsilon=epsilon,
    )
    settings = vars(args).copy()
    settings['output_dir'] = str(args.output_dir)
    settings.update(n_atoms=n, temperature_k=188.0, relaxation_dt_ps=0.00025,
                    relaxation_tau_ps=0.01, initial_velocities='zero',
                    jax_enable_x64=bool(jax.config.jax_enable_x64))
    results = [association, dissociation]
    tolerance = 0.05 if not jax.config.jax_enable_x64 else 0.005
    report = summarize(results, settings, tolerance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'energy_audit.json').write_text(
        json.dumps({'settings': settings, 'cases': results,
                    'reference_tolerance_kcal_mol': tolerance}, indent=2))
    (args.output_dir / 'energy_audit.md').write_text(report)
    print(report)
    if any(abs(c['offset_once_residual']) > tolerance or
           any(abs(c[s]['spectator_residual']) > tolerance for s in
               ('before', 'translated_trial', 'control_relaxed', 'trial_relaxed'))
           for c in results):
        raise SystemExit('Audit reference checks exceeded tolerance; see energy_audit.json')


if __name__ == '__main__':
    main()
