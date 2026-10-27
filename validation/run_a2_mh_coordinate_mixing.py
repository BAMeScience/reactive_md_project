"""Two-atom NVT-MH coordinate-mixing diagnostic using the production driver.

Produces per-check CSV traces and a JSON diagnostic summary. NOT an
unconditional equilibrium PASS/FAIL test: Nose-Hoover ergodicity and finite
sampling remain scientific questions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from reactive_md.config import SimConfig
from reactive_md.md_driver import run_md_nvt
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_handlers import A2MetropolisHandler
from validation.a2_mh_moving_reference import TwoAtomModel, integrate_reference
from validation.a2_mh_mixing_reference import (
    radial_reference, integrated_autocorrelation, batch_means_se, observed_radial,
)
from validation.run_a2_mh_moving_pilot import (
    system_two_atoms, make_forcefield, verify_real_energies,
)


class MixingObserver:
    def __init__(self, model, burn_in_checks):
        self.model = model
        self.burn_in_checks = burn_in_checks
        self.rows = []
        self.accepted = self.rejected = self.null = 0

    def on_observation(self, obs):
        if obs.outcome is None:
            return
        step = int(obs.step)
        n_bonds = int(np.asarray(obs.system.bonds[0]).shape[0])
        if n_bonds not in (0, 1):
            raise AssertionError('Invalid two-atom topology')
        positions = np.asarray(obs.state.position)
        momentum = np.asarray(obs.state.momentum)
        if not (np.isfinite(positions).all() and np.isfinite(momentum).all()):
            raise AssertionError('Nonfinite MD coordinates or momentum')
        if bool(np.asarray(obs.ff.nlist.did_buffer_overflow)):
            raise AssertionError('Neighbor list overflow')
        disp = np.asarray(obs.ff.disp_fn(obs.state.position[0], obs.state.position[1]))
        r = float(np.linalg.norm(disp))
        if r < 0 or r > self.model.box * np.sqrt(3) / 2 + 1e-3:
            raise AssertionError('Invalid periodic minimum-image distance')
        attempts = tuple((obs.outcome.metadata or {}).get('attempts', ()))
        event = 'accepted' if obs.outcome.events else ('rejected' if attempts else 'null')
        if event == 'accepted':
            self.accepted += 1
        elif event == 'rejected':
            self.rejected += 1
        else:
            self.null += 1
        check = len(self.rows) + 1
        self.rows.append({
            'check': check, 'step': step, 'time_ps': float(obs.md_time_ps),
            'bonded': n_bonds, 'r_A': r,
            'dx_A': float(disp[0]), 'dy_A': float(disp[1]), 'dz_A': float(disp[2]),
            'outside_cutoff': int(r > self.model.association_cutoff),
            'event': event, 'post_burn_in': int(check > self.burn_in_checks),
        })


def run_chain(model, *, chain_index, initial_dimers, start_geometry, steps,
              check_every, burn_in_checks, seed, mass, dt, tau_T):
    if start_geometry == 'near':
        displacement = [2.0, 0.0, 0.0]
    elif start_geometry == 'far':
        displacement = [2.2, 2.2, 2.2]  # 3.81 A; beyond association cutoff
    else:
        raise ValueError('start_geometry must be near or far')
    positions = jnp.asarray([[1.5, 1.5, 1.5],
                             (np.asarray([1.5, 1.5, 1.5]) + displacement).tolist()],
                            dtype=jnp.float32)
    box = jnp.full((3,), model.box, dtype=jnp.float32)
    _, shift = space.periodic(box)
    system = system_two_atoms(bool(initial_dimers), model)
    ff = make_forcefield(system, model, positions, bool(initial_dimers))
    reaction = A2Reaction(
        a_indices=np.arange(2, dtype=np.int32),
        k_bond_jax=model.bond_k, r0_bond=model.bond_r0,
        association_cutoff=model.association_cutoff,
        bond_energy_offset=model.bond_offset,
    )
    observer = MixingObserver(model, burn_in_checks)
    cfg = SimConfig(
        steps=steps, check_every=check_every, max_events=steps + 1,
        dt=dt, temperature_k=model.temperature_k, kb_real=model.kb_real,
        tau_T=tau_T,
    )
    result = run_md_nvt(
        jax.random.PRNGKey(seed), cfg=cfg, init_positions=positions,
        masses=jnp.full((2,), mass, dtype=jnp.float32), box=box,
        shift_fn=shift, ff=ff, sys=system,
        reaction_handler=A2MetropolisHandler(
            reaction=reaction, rng=np.random.default_rng(seed + 10000)),
        observers=(observer,),
    )
    if result.steps_done != steps:
        raise AssertionError('Incomplete MD trajectory')
    if observer.accepted != result.accepted_events:
        raise AssertionError('Accepted reaction count mismatch')
    if observer.accepted + observer.rejected + observer.null != len(observer.rows):
        raise AssertionError('Reaction attempt accounting mismatch')
    kept = [r for r in observer.rows if r['post_burn_in']]
    if not kept:
        raise ValueError('No post-burn-in observations')
    bonded = np.asarray([r['bonded'] for r in kept], dtype=float)
    distances = np.asarray([r['r_A'] for r in kept], dtype=float)
    halfway = len(kept) // 2
    chain_report = {
        'chain': chain_index + 1, 'seed': seed, 'initial_dimers': initial_dimers,
        'start_geometry': start_geometry,
        'initial_distance_A': float(np.linalg.norm(displacement)),
        'checks': len(observer.rows), 'retained_checks': len(kept),
        'accepted': observer.accepted, 'rejected': observer.rejected,
        'null': observer.null,
        'bonded_fraction': float(bonded.mean()),
        'first_half_bonded_fraction': (float(bonded[:halfway].mean()) if halfway else None),
        'second_half_bonded_fraction': float(bonded[halfway:].mean()),
        'mean_r_A': float(distances.mean()),
        'min_r_A': float(distances.min()), 'max_r_A': float(distances.max()),
        'fraction_outside_association_cutoff': float((distances > model.association_cutoff).mean()),
        'signed_displacement_std_A': [float(np.std([r[k] for r in kept]))
                                      for k in ('dx_A', 'dy_A', 'dz_A')],
        'bonded_iat': integrated_autocorrelation(bonded),
        'radial_iat': integrated_autocorrelation(distances),
        'final_dimers': int(np.asarray(result.sys.bonds[0]).shape[0]),
    }
    return chain_report, observer.rows, np.column_stack([bonded, distances])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--check-every', type=int, default=20)
    parser.add_argument('--burn-in-checks', type=int, default=10)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--seed', type=int, default=193)
    parser.add_argument('--mass', type=float, default=4.0)
    parser.add_argument('--dt', type=float, default=0.00025)
    parser.add_argument('--tau-T', type=float, default=0.1)
    parser.add_argument('--quadrature-order', type=int, default=96)
    parser.add_argument('--radial-bins', type=int, default=12)
    parser.add_argument('--reference-check-order', type=int, default=128,
                        help='Second quadrature order to estimate radial bin error')
    parser.add_argument('--block-size', type=int, default=10,
                        help='Nonoverlapping retained-check block size; diagnostic only')
    parser.add_argument('--output', type=Path,
                        default=Path('validation/a2_mh_coordinate_mixing.json'))
    args = parser.parse_args(argv)
    if args.steps < 1 or args.check_every < 1 or args.steps % args.check_every:
        parser.error('steps must be a positive multiple of check-every')
    if args.chains < 4 or args.mass <= 0 or args.dt <= 0 or args.tau_T <= 0:
        parser.error('Require >= 4 chains and positive mass, dt, tau-T')
    if args.burn_in_checks < 0 or args.burn_in_checks >= args.steps // args.check_every:
        parser.error('burn-in-checks must be below the total checks per chain')
    if args.reference_check_order < 8:
        parser.error('reference-check-order must be >= 8')
    if args.block_size < 1 or args.radial_bins < 2:
        parser.error('block-size must be positive and radial-bins >= 2')
    model = TwoAtomModel()
    reference = integrate_reference(model, order=args.quadrature_order)
    radial = radial_reference(model, order=args.quadrature_order,
                              bins=args.radial_bins)
    if abs(reference['p_bonded'] - radial['p_bonded']) > 1e-9:
        raise AssertionError('Independent partition and shell reference disagree')
    radial_check = radial_reference(model, order=args.reference_check_order,
                                    bins=args.radial_bins)
    radial_quadrature_tv = float(0.5 * np.sum(np.abs(
        np.asarray(radial['prob_r_by_shell']) -
        np.asarray(radial_check['prob_r_by_shell']))))
    spot_checks = verify_real_energies(model)
    print('Real force-field energy spot checks: PASS', flush=True)
    print(f'Full-box reference P(bonded): {reference["p_bonded"]:.6f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chain_reports = []
    sample_arrays = []
    combos = ((0, 'near'), (1, 'near'), (0, 'far'), (1, 'far'))
    for chain in range(args.chains):
        initial_dimers, geometry = combos[chain % 4]
        report, rows, samples = run_chain(
            model, chain_index=chain, initial_dimers=initial_dimers,
            start_geometry=geometry, steps=args.steps,
            check_every=args.check_every, burn_in_checks=args.burn_in_checks,
            seed=args.seed + 101 * chain, mass=args.mass,
            dt=args.dt, tau_T=args.tau_T,
        )
        chain_reports.append(report)
        sample_arrays.append(samples)
        trace_path = args.output.with_name(f'{args.output.stem}_chain{chain+1}.csv')
        with trace_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Chain {chain+1}/{args.chains} ({geometry}, initial dimers={initial_dimers}): '
              f'bonded={report["bonded_fraction"]:.4f}, '
              f'mean r={report["mean_r_A"]:.3f} A, '
              f'r range=[{report["min_r_A"]:.3f},{report["max_r_A"]:.3f}] A, '
              f'accepted={report["accepted"]}, rejected={report["rejected"]}, '
              f'null={report["null"]}', flush=True)
    pooled = np.concatenate(sample_arrays, axis=0)
    block = batch_means_se([a[:, 0] for a in sample_arrays], block_size=args.block_size)
    hist = observed_radial(pooled, radial)
    pooled_bonded = float(pooled[:, 0].mean())
    result = {
        'stage': 'two_atom_nvt_mh_coordinate_mixing_diagnostic',
        'baseline': 'develop 1d35ef8 plus additive validation files',
        'settings': {'steps': args.steps, 'check_every': args.check_every,
                     'burn_in_checks': args.burn_in_checks, 'chains': args.chains,
                     'mass': args.mass, 'dt': args.dt, 'tau_T': args.tau_T,
                     'block_size': args.block_size},
        'reference': reference, 'radial_reference': radial,
        'radial_quadrature_comparison': {
            'check_order': args.reference_check_order,
            'radial_total_variation_between_orders': radial_quadrature_tv,
            'p_bonded_check_order': radial_check['p_bonded'],
        },
        'forcefield_energy_spot_checks': spot_checks,
        'chains': chain_reports, 'radial_diagnostics': hist,
        'pooled_bonded_fraction': pooled_bonded,
        'pooled_mean_r_A': float(pooled[:, 1].mean()),
        'reference_mean_r_A': float(
            (1.0 - reference['p_bonded']) * reference['mean_r_unbound'] +
            reference['p_bonded'] * reference['mean_r_bonded']),
        'difference_from_reference': pooled_bonded - reference['p_bonded'],
        'bonded_block_se_heuristic': block,
        'equilibrium_validated': False,
        'interpretation': (
            'Diagnostic only. Nose-Hoover may be nonergodic for two atoms. '
            'The block SE is unreliable if blocks are shorter than correlation '
            'times or chains are not equilibrated. No equilibrium PASS is claimed.'
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Pooled bonded={pooled_bonded:.6f} (reference {reference["p_bonded"]:.6f}); '
          f'pooled mean r={result["pooled_mean_r_A"]:.3f} A; '
          f'radial TV={hist["radial_total_variation"]:.4f}; '
          f'joint TV={hist["joint_total_variation"]:.4f}', flush=True)
    print(f'Radial quadrature TV ({args.quadrature_order} vs '
          f'{args.reference_check_order}): {radial_quadrature_tv:.5f}', flush=True)
    print(f'Block SE (heuristic): {block["se"]}; '
          f'number of blocks={block["n_batches"]}', flush=True)
    print('Coordinate mixing diagnostic: COMPLETE (equilibrium NOT yet validated)', flush=True)
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
