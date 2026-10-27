"""Audit production two-atom MH transitions against an independent kernel.

Run from the repository root on the HPC JAX-MD environment:
  python -m validation.run_a2_mh_kernel_audit

No MD trajectories are run, and no production modules are modified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_metropolis import a2_metropolis_reactive_cycle
from validation.a2_mh_kernel_reference import kernel_at_distance
from validation.a2_mh_moving_reference import TwoAtomModel, integrate_reference
from validation.run_a2_mh_moving_pilot import make_forcefield, system_two_atoms


class ForcedRNG:
    """Deterministically select direction and accept any nonzero probability."""

    def __init__(self, direction: str):
        if direction not in ('association', 'dissociation'):
            raise ValueError(direction)
        self.direction = direction
        self.draws = 0

    def random(self):
        self.draws += 1
        if self.draws == 1:
            return 0.0 if self.direction == 'association' else 0.75
        return 0.0

    def integers(self, high):
        assert high == 1, f'Expected one candidate, got {high}'
        return 0


def _positions(r, model, periodic_edge=False):
    if periodic_edge:
        # Minimum-image separation r, despite a raw coordinate gap L-r.
        return jnp.asarray([[model.box - 1.0, 1.0, 1.0],
                            [r - 1.0, 1.0, 1.0]], dtype=jnp.float32)
    return jnp.asarray([[1.0, 1.0, 1.0], [1.0 + r, 1.0, 1.0]], dtype=jnp.float32)


def audit_case(model, r, *, periodic_edge=False, energy_tol=2e-3,
               acceptance_tol=3e-3):
    """Check both topology energies and both forced reaction directions."""
    positions = _positions(r, model, periodic_edge)
    box = jnp.full((3,), model.box, dtype=jnp.float32)
    velocities = jnp.zeros_like(positions)
    reaction = A2Reaction(
        a_indices=np.arange(2, dtype=np.int32),
        k_bond_jax=model.bond_k, r0_bond=model.bond_r0,
        association_cutoff=model.association_cutoff,
        bond_energy_offset=model.bond_offset,
    )
    systems = {}
    ffs = {}
    energies = {}
    for bonded in (False, True):
        system = system_two_atoms(bonded, model)
        ff = make_forcefield(system, model, positions, bonded)
        disp = np.asarray(ff.disp_fn(positions[0], positions[1]))
        measured_r = float(np.linalg.norm(disp))
        if abs(measured_r - r) > 2e-5:
            raise AssertionError(f'Periodic minimum-image mismatch: {measured_r} vs {r}')
        u = float(ff.energy_fn(positions, ff.nlist)['total'])
        if not np.isfinite(u):
            raise AssertionError('Nonfinite production energy')
        systems[bonded], ffs[bonded], energies[bonded] = system, ff, u
    ref = kernel_at_distance(r, model)
    for bonded, expected in ((False, ref['energy_unbound']),
                             (True, ref['energy_bonded'])):
        if abs(energies[bonded] - expected) > energy_tol:
            raise AssertionError(
                f'Energy mismatch r={r}, bonded={bonded}: '
                f'actual={energies[bonded]}, independent={expected}'
            )
    results = {}
    for direction, bonded in (('association', False), ('dissociation', True)):
        rng = ForcedRNG(direction)
        ff, system = ffs[bonded], systems[bonded]
        result = a2_metropolis_reactive_cycle(
            key=None, R=positions, velocities=velocities, box=box,
            system=system, ff=ff, reaction=reaction, disp_fn=ff.disp_fn,
            temperature_k=model.temperature_k, kb_real=model.kb_real,
            rng=rng,
        )
        if result.positions is not positions or result.velocities is not velocities:
            raise AssertionError('MH unexpectedly changed positions/velocities')
        expected_p = ref[f'{direction}_acceptance']
        eligible = ref['association_eligible']
        if eligible:
            if len(result.attempts) != 1 or len(result.events) != 1:
                raise AssertionError(f'{direction} at r={r} should be accepted with forced RNG')
            attempt = result.attempts[0]
            if not attempt.accepted or not math.isclose(attempt.probability, expected_p,
                                                        abs_tol=acceptance_tol):
                raise AssertionError(
                    f'{direction} probability mismatch at r={r}: '
                    f'{attempt.probability} vs {expected_p}'
                )
            if abs(attempt.log_proposal_ratio) > 1e-12:
                raise AssertionError('Two-atom proposal ratio must be 1')
            expected_delta = ref['delta_energy_association'] * (1 if not bonded else -1)
            if abs(attempt.delta_energy - expected_delta) > energy_tol * 2:
                raise AssertionError('Production MH delta energy mismatch')
            resulting_bonded = bool(np.asarray(result.system.bonds[0]).shape[0])
            if resulting_bonded == bonded:
                raise AssertionError('Accepted proposal did not flip topology')
            if rng.draws != 2:
                raise AssertionError('Expected direction and acceptance RNG draws')
            results[direction] = {'outcome': 'accepted', 'probability': attempt.probability,
                                  'delta_energy': attempt.delta_energy}
        else:
            if result.events or result.system is not system or result.ff is not ff:
                raise AssertionError('Forbidden transition changed the system')
            if direction == 'association':
                if result.attempts or rng.draws != 1:
                    raise AssertionError('Outside-cutoff association must be null')
                reason = 'no candidate'
            else:
                if len(result.attempts) != 1 or rng.draws != 1:
                    raise AssertionError('Outside-cutoff dissociation must be reverse-blocked')
                attempt = result.attempts[0]
                if attempt.reason != 'reverse move unavailable' or attempt.probability != 0:
                    raise AssertionError('Wrong outside-cutoff dissociation outcome')
                reason = attempt.reason
            results[direction] = {'outcome': reason, 'probability': 0.0}
    if not math.isclose(ref['forward_flux'], ref['reverse_flux'], rel_tol=1e-12,
                        abs_tol=1e-12):
        raise AssertionError('Independent MH kernel violates local detailed balance')
    return {
        'distance': r, 'periodic_edge': periodic_edge,
        'reference': ref,
        'production_energies': {'unbound': energies[False], 'bonded': energies[True]},
        'production_transitions': results,
        'status': 'PASS',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('validation/a2_mh_kernel_audit.json'))
    parser.add_argument('--quadrature-order', type=int, default=96)
    args = parser.parse_args(argv)
    model = TwoAtomModel()
    quadrature = integrate_reference(model, order=args.quadrature_order)
    cases = [(2.0, False), (2.95, False), (3.05, False), (2.0, True)]
    records = []
    for r, edge in cases:
        record = audit_case(model, r, periodic_edge=edge)
        records.append(record)
        ref = record['reference']
        print(f'r={r:.2f} A, periodic_edge={edge}: '
              f'P(bonded|r)={ref["conditional_bonded_probability"]:.5f}, '
              f'eligible={ref["association_eligible"]}, '
              'production kernel PASS', flush=True)
    result = {
        'stage': 'two_atom_reference_and_reaction_kernel_audit',
        'model': vars(model),
        'configurational_reference': quadrature,
        'cases': records,
        'production_physics_modified': False,
        'status': 'PASS',
        'equilibrium_validated': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Full-box reference P(bonded)={quadrature["p_bonded"]:.6f}', flush=True)
    print('Reference and reaction-kernel audit: PASS', flush=True)
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
