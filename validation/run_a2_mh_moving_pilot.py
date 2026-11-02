"""Two-atom moving-coordinate NVT-MH pilot using the production MD driver.

This is a deliberately small diagnostic, NOT a pass/fail equilibrium test.
The reference is an independent full-periodic-cube configurational integral.
Run on HPC: python -m validation.run_a2_mh_moving_pilot --steps 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from reactive_md.config import SimConfig
from reactive_md.forcefield import build_forcefield
from reactive_md.md_driver import run_md_nvt
from reactive_md.reaction import SystemState
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_handlers import A2MetropolisHandler
from validation.a2_mh_moving_reference import (
    TwoAtomModel, energy_bonded, energy_unbound, integrate_reference,
)


def system_two_atoms(bonded: bool, model: TwoAtomModel):
    bonds = (
        jnp.asarray([[0, 1]] if bonded else [], dtype=jnp.int32).reshape(-1, 2),
        jnp.asarray([model.bond_k] if bonded else [], dtype=jnp.float32),
        jnp.asarray([model.bond_r0] if bonded else [], dtype=jnp.float32),
    )
    angles = (jnp.empty((0, 3), dtype=jnp.int32),
              jnp.empty((0,), dtype=jnp.float32), jnp.empty((0,), dtype=jnp.float32))
    four = (jnp.empty((0, 4), dtype=jnp.int32),
            jnp.empty((0,), dtype=jnp.float32),
            jnp.empty((0,), dtype=jnp.int32), jnp.empty((0,), dtype=jnp.float32))
    return SystemState(
        bonds=bonds, angles=angles, torsions=four, impropers=four,
        charges=jnp.zeros(2, dtype=jnp.float32),
        sigmas=jnp.full(2, model.sigma, dtype=jnp.float32),
        epsilons=jnp.full(2, model.epsilon, dtype=jnp.float32),
        molecule_id=jnp.asarray([0, 0] if bonded else [0, 1], dtype=jnp.int32),
        pf6_reacted=jnp.zeros((0,), dtype=bool),
    )


def make_forcefield(system, model, positions, bonded):
    box = jnp.full((3,), model.box, dtype=jnp.float32)
    return build_forcefield(
        positions, box,
        bond_idx=np.asarray(system.bonds[0]),
        k_b=np.asarray(system.bonds[1]), r0=np.asarray(system.bonds[2]),
        angle_idx=np.asarray(system.angles[0]),
        k_theta=np.asarray(system.angles[1]), theta0=np.asarray(system.angles[2]),
        torsions=tuple(np.asarray(x) for x in system.torsions),
        impropers=tuple(np.asarray(x) for x in system.impropers),
        charges=np.asarray(system.charges), sigmas=np.asarray(system.sigmas),
        epsilons=np.asarray(system.epsilons), molecule_id=np.asarray(system.molecule_id),
        r_cut=model.r_cut, dr_threshold=0.3,
        use_shift_lj=False, energy_offset=model.bond_offset if bonded else 0.0,
    )


def verify_real_energies(model):
    """Reject a mismatched reference instead of silently comparing ensembles."""
    results = []
    for bonded in (False, True):
        system = system_two_atoms(bonded, model)
        for r in (1.5, 2.0, 2.5, 3.3):
            positions = jnp.asarray([[1., 1., 1.], [1. + r, 1., 1.]], dtype=jnp.float32)
            ff = make_forcefield(system, model, positions, bonded)
            actual = float(ff.energy_fn(positions, ff.nlist)['total'])
            expected = float(energy_bonded(r, model) if bonded else energy_unbound(r, model))
            if not np.isfinite(actual) or abs(actual - expected) > 2e-3:
                raise AssertionError(
                    f'Independent reference does not match real JAX-MD energy: '
                    f'bonded={bonded}, r={r}, actual={actual}, expected={expected}. '
                    'Do not interpret the equilibrium comparison until resolved.'
                )
            results.append({'bonded': bonded, 'distance': r,
                            'actual': actual, 'independent': expected})
    return results


class PilotObserver:
    def __init__(self, model, burn_in_checks):
        self.model = model
        self.burn_in_checks = burn_in_checks
        self.checks = 0
        self.samples = []
        self.accepted = 0
        self.rejected = 0
        self.null = 0

    def on_observation(self, obs):
        if obs.outcome is None:
            return  # initial frame
        self.checks += 1
        n_bonds = int(np.asarray(obs.system.bonds[0]).shape[0])
        if n_bonds not in (0, 1):
            raise AssertionError('Invalid two-atom topology')
        if not np.isfinite(np.asarray(obs.state.position)).all():
            raise AssertionError('Nonfinite coordinates')
        if not np.isfinite(np.asarray(obs.state.momentum)).all():
            raise AssertionError('Nonfinite momenta')
        if bool(np.asarray(obs.ff.nlist.did_buffer_overflow)):
            raise AssertionError('Neighbor-list overflow')
        disp = obs.ff.disp_fn(obs.state.position[0], obs.state.position[1])
        r = float(np.linalg.norm(np.asarray(disp)))
        if not 0 <= r <= self.model.box * np.sqrt(3) / 2 + 1e-3:
            raise AssertionError('Invalid minimum-image distance')
        attempts = tuple((obs.outcome.metadata or {}).get('attempts', ()))
        if obs.outcome.events:
            self.accepted += 1
        elif attempts:
            self.rejected += 1
        else:
            self.null += 1
        if self.checks > self.burn_in_checks:
            self.samples.append((n_bonds, r))


def run_chain(model, *, initial_dimers, steps, check_every, burn_in_checks,
              seed, mass, dt, tau_T):
    if initial_dimers not in (0, 1):
        raise ValueError('Two-atom system must start with zero or one dimer')
    positions = jnp.asarray([[1.5, 1.5, 1.5], [3.5, 1.5, 1.5]], dtype=jnp.float32)
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
    audit = PilotObserver(model, burn_in_checks)
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
        observers=(audit,),
    )
    if result.steps_done != steps or audit.accepted != result.accepted_events:
        raise AssertionError('Incomplete or inconsistent coupled trajectory')
    if audit.accepted + audit.rejected + audit.null != audit.checks:
        raise AssertionError('Reaction attempt accounting mismatch')
    if not audit.samples:
        raise ValueError('No post-burn-in observations; reduce burn-in')
    samples = np.asarray(audit.samples, dtype=float)
    return {
        'seed': seed, 'initial_dimers': initial_dimers,
        'checks': audit.checks, 'post_burn_in_samples': len(samples),
        'accepted': audit.accepted, 'rejected': audit.rejected, 'null': audit.null,
        'bonded_fraction': float(samples[:, 0].mean()),
        'mean_distance_unbound': (float(samples[samples[:, 0] == 0, 1].mean())
                                  if np.any(samples[:, 0] == 0) else None),
        'mean_distance_bonded': (float(samples[samples[:, 0] == 1, 1].mean())
                                 if np.any(samples[:, 0] == 1) else None),
        'final_dimers': int(np.asarray(result.sys.bonds[0]).shape[0]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--check-every', type=int, default=10)
    parser.add_argument('--burn-in-checks', type=int, default=5)
    parser.add_argument('--chains', type=int, default=2)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--mass', type=float, default=4.0,
                        help='Artificially low mass accelerates coordinate motion; not an argon kinetics model')
    parser.add_argument('--dt', type=float, default=0.00025)
    parser.add_argument('--tau-T', type=float, default=0.1)
    parser.add_argument('--quadrature-order', type=int, default=96)
    parser.add_argument('--output', type=Path,
                        default=Path('validation/a2_mh_moving_pilot.json'))
    args = parser.parse_args(argv)
    if args.steps < 1 or args.check_every < 1 or args.chains < 2 or args.mass <= 0:
        parser.error('Require positive steps/check-every/mass and at least two chains')
    if args.steps % args.check_every:
        parser.error('steps must be divisible by check-every for this pilot')
    if args.burn_in_checks < 0 or args.burn_in_checks >= args.steps // args.check_every:
        parser.error('burn-in-checks must be less than the number of reaction checks')
    model = TwoAtomModel()
    reference = integrate_reference(model, order=args.quadrature_order)
    spot_checks = verify_real_energies(model)
    print('Independent two-atom equilibrium reference: '
          f'p(bonded)={reference["p_bonded"]:.6f}', flush=True)
    print('Real force-field vs independent energy spot checks: PASS', flush=True)
    reports = []
    for chain in range(args.chains):
        initial = chain % 2
        record = run_chain(model, initial_dimers=initial, steps=args.steps,
                           check_every=args.check_every,
                           burn_in_checks=args.burn_in_checks,
                           seed=args.seed + 101 * chain,
                           mass=args.mass, dt=args.dt, tau_T=args.tau_T)
        reports.append(record)
        print(f'Chain {chain + 1}/{args.chains} (initial dimers={initial}): '
              f'bonded fraction={record["bonded_fraction"]:.5f}, '
              f'accepted={record["accepted"]}, rejected={record["rejected"]}, '
              f'null={record["null"]}', flush=True)
    pooled = float(np.average([r['bonded_fraction'] for r in reports],
                              weights=[r['post_burn_in_samples'] for r in reports]))
    result = {
        'stage': 'moving_coordinate_nvt_mh_pilot',
        'reference': reference, 'forcefield_energy_spot_checks': spot_checks,
        'settings': {'steps': args.steps, 'check_every': args.check_every,
                     'burn_in_checks': args.burn_in_checks, 'chains': args.chains,
                     'mass': args.mass, 'dt': args.dt, 'tau_T': args.tau_T},
        'chains': reports, 'pooled_bonded_fraction': pooled,
        'difference_from_reference': pooled - reference['p_bonded'],
        'equilibrium_validated': False,
        'interpretation': 'Pilot only: samples are correlated, thermostat may be nonergodic for two particles; no convergence claim.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Pooled bonded fraction: {pooled:.6f} '
          f'(reference {reference["p_bonded"]:.6f})', flush=True)
    print('Moving-coordinate pilot: COMPLETE (equilibrium NOT yet validated)', flush=True)
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
