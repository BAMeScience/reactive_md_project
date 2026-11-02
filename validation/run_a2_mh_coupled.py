"""Four-atom *real* JAX-MD + A2 MH coupled integration audit.

Run from the repository root in the HPC JAX-MD environment:
    python -m validation.run_a2_mh_coupled --steps 60 --replay

This deliberately tests finite-time integration invariants, NOT canonical
stationary probabilities. Production code is not modified or monkeypatched.
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

from reactive_md.config import SimConfig
from reactive_md.md_driver import run_md_nvt
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_handlers import A2MetropolisHandler
from validation.run_a2_mh_sampling import (
    BOX, POSITIONS, empty_system, make_system, matching, real_forcefield,
)

# Four A atoms on a square. The same topology states and force-field settings
# were used in the passing fixed-coordinate real-energy validation.
STARTS = ((), ((0, 1), (2, 3)))
OFFSET = -0.5
MASS = jnp.full((4,), 39.948, dtype=jnp.float32)


def _assert_finite(name, value):
    if not bool(np.all(np.isfinite(np.asarray(jax.device_get(value))))):
        raise AssertionError(f'Nonfinite {name}')


def _assert_topology(system):
    state = matching(system)
    atoms = [a for pair in state for a in pair]
    if len(atoms) != len(set(atoms)):
        raise AssertionError(f'A atom belongs to multiple dimers: {state}')
    if any(a not in range(4) for a in atoms):
        raise AssertionError(f'Out-of-range atom in bonds: {state}')
    if len(state) > 2:
        raise AssertionError('More than two A2 dimers')
    return state


def _energy(ff, positions):
    nl = ff.neighbor_fn.update(positions, ff.nlist)
    if bool(np.asarray(nl.did_buffer_overflow)):
        raise AssertionError('Neighbor list overflowed at audit coordinates')
    return float(ff.energy_fn(positions, nl)['total'])


class CouplingAudit:
    """Observe both sides of each real MD-MH check without changing physics."""

    def __init__(self, offset):
        self.offset = float(offset)
        self.before = None
        self.checks = 0
        self.accepted = 0
        self.rejected = 0
        self.null = 0
        self.records = []
        self.energy_errors = []

    def on_before_reaction(self, obs):
        self.before = obs

    def on_observation(self, obs):
        state = _assert_topology(obs.system)
        _assert_finite('positions', obs.state.position)
        _assert_finite('momenta', obs.state.momentum)
        _assert_finite('velocities', obs.state.velocity)
        _assert_finite('forces', obs.state.force)
        if bool(np.asarray(obs.ff.nlist.did_buffer_overflow)):
            raise AssertionError('Committed neighbor list overflowed')
        # An all-monomer initial FF may omit the explicit zero offset key.
        energies = obs.ff.energy_fn(obs.state.position, obs.ff.nlist)
        _assert_finite('total potential energy', energies['total'])
        actual_offset = float(energies.get('offset', 0.0))
        if not np.isclose(actual_offset, len(state) * self.offset, atol=1e-5):
            raise AssertionError(
                f'Energy offset mismatch: {actual_offset} vs {len(state) * self.offset}'
            )
        if obs.outcome is None:
            if obs.step != 0:
                raise AssertionError('Unexpected missing outcome')
            return
        self.checks += 1
        if self.before is None or self.before.step != obs.step:
            raise AssertionError('Missing matching pre-reaction observation')
        before = self.before
        outcome = obs.outcome
        attempts = tuple((outcome.metadata or {}).get('attempts', ()))
        if len(attempts) > 1 or len(outcome.events) > 1:
            raise AssertionError('MH must attempt at most one event per check')
        if outcome.reinitialize_thermostat or outcome.relaxation_time_ps != 0:
            raise AssertionError('MH unexpectedly reset thermostat or relaxed')
        if obs.relaxation_time_ps != 0:
            raise AssertionError('MH accumulated relaxation time')
        if outcome.events:
            if len(attempts) != 1 or not attempts[0].accepted:
                raise AssertionError('Accepted event missing accepted attempt')
            self.accepted += 1
            # The MH reaction is fixed-coordinate: no displacement or impulse.
            np.testing.assert_array_equal(
                np.asarray(obs.state.position), np.asarray(before.state.position)
            )
            np.testing.assert_array_equal(
                np.asarray(obs.state.momentum), np.asarray(before.state.momentum)
            )
            np.testing.assert_array_equal(
                np.asarray(obs.state.velocity), np.asarray(before.state.velocity)
            )
            old_energy = _energy(before.ff, before.state.position)
            new_energy = _energy(obs.ff, before.state.position)
            delta = new_energy - old_energy
            reported_delta = float(outcome.events[0].delta_energy)
            error = abs(delta - reported_delta)
            self.energy_errors.append(error)
            if error > 2e-3:
                raise AssertionError(
                    f'MH energy delta mismatch: {delta} vs {reported_delta}'
                )
            # The integrator must refresh cached force after topology changes.
            force = -jax.grad(
                lambda r: obs.ff.energy_fn(r, obs.ff.nlist)['total']
            )(obs.state.position)
            np.testing.assert_allclose(
                np.asarray(obs.state.force), np.asarray(force), rtol=2e-4, atol=2e-4
            )
            before_state = _assert_topology(before.system)
            if abs(len(state) - len(before_state)) != 1:
                raise AssertionError('Accepted event did not change dimer count by one')
        elif attempts:
            self.rejected += 1
            if attempts[0].accepted:
                raise AssertionError('Accepted attempt without event')
            if obs.ff is not before.ff or obs.system is not before.system:
                raise AssertionError('Rejected attempt changed topology or force field')
        else:
            self.null += 1
            if obs.ff is not before.ff or obs.system is not before.system:
                raise AssertionError('Null proposal changed topology or force field')
        self.records.append({
            'step': int(obs.step), 'state': [list(p) for p in state],
            'accepted': bool(outcome.events),
            'reaction': outcome.events[0].reaction_type if outcome.events else None,
            'potential_energy': float(energies['total']),
        })
        self.before = None


def run_case(*, initial_dimers, steps=60, check_every=1, seed=23, offset=OFFSET):
    if steps < 1 or check_every < 1:
        raise ValueError('steps and check_every must be positive')
    if initial_dimers not in (0, 2):
        raise ValueError('initial_dimers must be 0 or 2')
    start = STARTS[0 if initial_dimers == 0 else 1]
    reaction = A2Reaction(
        a_indices=np.arange(4, dtype=np.int32), k_bond_jax=1.5,
        r0_bond=2.0, association_cutoff=3.0, bond_energy_offset=offset,
    )
    system = empty_system() if not start else make_system(start, reaction)
    ff = real_forcefield(system, initial_dimers * offset)
    # Periodic boundary handling must be identical in MD and MC.
    _, shift_fn = space.periodic(BOX)
    cfg = SimConfig(
        steps=int(steps), check_every=int(check_every), max_events=int(steps) + 1,
        dt=0.00025, temperature_k=300.0, kb_real=0.0019872041,
        tau_T=0.1,
    )
    audit = CouplingAudit(offset)
    result = run_md_nvt(
        jax.random.PRNGKey(seed), cfg=cfg, init_positions=POSITIONS,
        masses=MASS, box=BOX, shift_fn=shift_fn, ff=ff, sys=system,
        reaction_handler=A2MetropolisHandler(
            reaction=reaction, rng=np.random.default_rng(seed + 10000),
        ), observers=(audit,),
    )
    if result.steps_done != steps:
        raise AssertionError(f'Unexpected early stop: {result.steps_done}/{steps}')
    if audit.checks != math.ceil(steps / check_every):
        raise AssertionError('Incorrect number of reaction checks')
    if audit.accepted != result.accepted_events:
        raise AssertionError('Accepted-event count mismatch')
    if audit.accepted + audit.rejected + audit.null != audit.checks:
        raise AssertionError('MC accounting mismatch')
    if result.cumulative_relaxation_time_ps != 0:
        raise AssertionError('Unexpected relaxation time')
    return {
        'initial_dimers': initial_dimers, 'steps': steps,
        'checks': audit.checks, 'accepted': audit.accepted,
        'rejected': audit.rejected, 'null': audit.null,
        'max_delta_energy_error': max(audit.energy_errors, default=0.0),
        'final_state': [list(p) for p in _assert_topology(result.sys)],
        'final_positions': np.asarray(result.final_md_state.position).tolist(),
        'final_momenta': np.asarray(result.final_md_state.momentum).tolist(),
        'records': audit.records,
    }


def check_replay(first, second):
    """Same seed on the same device must give the same trajectory."""
    if first['records'] != second['records']:
        raise AssertionError('Same-seed replay produced different MH events or energies')
    np.testing.assert_array_equal(first['final_positions'], second['final_positions'])
    np.testing.assert_array_equal(first['final_momenta'], second['final_momenta'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--check-every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--replay', action='store_true',
                        help='Repeat both starts with the same seed to check determinism')
    parser.add_argument('--output', type=Path,
                        default=Path('validation/a2_mh_coupled_integration.json'))
    args = parser.parse_args(argv)
    if args.steps < 1 or args.check_every < 1:
        parser.error('steps and check-every must be positive')
    reports = []
    for initial_dimers in (0, 2):
        first = run_case(initial_dimers=initial_dimers, steps=args.steps,
                         check_every=args.check_every, seed=args.seed)
        if args.replay:
            second = run_case(initial_dimers=initial_dimers, steps=args.steps,
                              check_every=args.check_every, seed=args.seed)
            check_replay(first, second)
        reports.append(first)
        print(f'Initial dimers={initial_dimers}: checks={first["checks"]}, '
              f'accepted={first["accepted"]}, rejected={first["rejected"]}, '
              f'null={first["null"]}, max |dE error|='
              f'{first["max_delta_energy_error"]:.6g} kcal/mol', flush=True)
    report = {'stage': 'coupled_nvt_mh_integration', 'passed': True,
              'replay': args.replay, 'seed': args.seed, 'cases': reports,
              'scope': 'integration invariants only; not canonical equilibrium validation'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Coupled MD-MH integration: PASS')
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
