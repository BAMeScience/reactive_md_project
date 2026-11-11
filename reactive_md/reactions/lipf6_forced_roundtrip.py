"""Diagnostic only: force a forward and reverse LiPF6 reaction during real NVT.

This is not a kinetic simulation. It bypasses event probabilities, but uses the
production topology, probe, force-field rebuild, Langevin and MD synchronization.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..reaction import build_trial_forcefield, make_system_state_from_trial
from ..relaxation import langevin_relax_with_nlist
from .lipf6 import ReactionCandidate, reaction_coordinate


def _canonical_terms(terms):
    """Canonicalize topology terms, preserving None and duplicates."""
    from collections import Counter

    def normalize(value):
        if value is None:
            return None
        if isinstance(value, (tuple, list, np.ndarray)):
            return tuple(normalize(v) for v in value)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float, bool)):
            return float(value)
        return value

    arrays = [np.asarray(a, dtype=object) for a in terms]
    return Counter(
        tuple(normalize(x) for x in row)
        for row in zip(*arrays)
    )

def verify_restoration(reference, final):
    """Check restoration of topology and per-atom parameters."""

    # Compare bonded topology, including duplicate terms.
    for name in ('bonds', 'angles', 'torsions', 'impropers'):
        a = _canonical_terms(getattr(reference, name))
        b = _canonical_terms(getattr(final, name))

        if a != b:
            raise AssertionError(
                f'{name} did not return to the original topology'
            )

    # Compare per-atom parameters, allowing optional None values.
    for name in ('charges', 'sigmas', 'epsilons', 'molecule_id'):
        original = getattr(reference, name)
        restored = getattr(final, name)

        if original is None or restored is None:
            if original is not None or restored is not None:
                raise AssertionError(f'{name} not restored')
            continue

        a = np.asarray(original, dtype=object)
        b = np.asarray(restored, dtype=object)

        if a.shape != b.shape:
            raise AssertionError(
                f'{name} shape mismatch: {a.shape} != {b.shape}'
            )

        for index, (x, y) in enumerate(zip(a.flat, b.flat)):
            if x is None or y is None:
                if x is not None or y is not None:
                    raise AssertionError(
                        f'{name}[{index}] not restored: {x} != {y}'
                    )
            elif not np.isclose(
                float(x), float(y), rtol=0, atol=1e-6
            ):
                raise AssertionError(
                    f'{name}[{index}] not restored: {x} != {y}'
                )

    if np.any(np.asarray(final.pf6_reacted)):
        raise AssertionError('A PF6 unit remains reacted')

    if (
        np.any(np.asarray(final.pf6_leave_f) != -1)
        or np.any(np.asarray(final.pf6_li_partner) != -1)
    ):
        raise AssertionError(
            'Reaction partner markers were not cleared'
        )

@dataclass
class ForcedRoundtrip:
    reaction: object
    box: object
    shift_fn: object
    masses: object
    cfg: object
    initial_system: object
    events: list = field(default_factory=list)
    initial_charge: float = field(init=False)
    initial_bond_count: int = field(init=False)
    initial_angle_count: int = field(init=False)

    def __post_init__(self):
        self.initial_charge = float(np.sum(np.asarray(self.initial_system.charges, dtype=np.float64)))
        self.initial_bond_count = len(self.initial_system.bonds[0])
        self.initial_angle_count = len(self.initial_system.angles[0])

    def __call__(self, key, R, ff, sys, velocities):
        check = len(self.events)
        if check >= 2:
            return key, False, ff, sys, {'mode': 'forced_roundtrip', 'n_candidates': 0}, R

        if check == 0:
            candidates = self.reaction.find_candidates(R, ff.disp_fn, pf6_reacted=sys.pf6_reacted)
            if not candidates:
                raise RuntimeError('No forward candidates found')
            candidate = max(candidates, key=lambda c: reaction_coordinate(d_pf=c.d_pf, d_lif=c.d_lif))
        else:
            old = self.events[0]['candidate']
            k, li, f = old['k_pf6'], old['li_idx'], old['leave_F']
            p = int(self.reaction.pf6_atoms[k, 0])
            d_pf = float(jnp.linalg.norm(ff.disp_fn(R[p], R[f])))
            d_lif = float(jnp.linalg.norm(ff.disp_fn(R[li], R[f])))
            candidate = ReactionCandidate(k, li, f, d_lif, d_pf, 'reverse')
            if not bool(np.asarray(sys.pf6_reacted)[k]):
                raise AssertionError('Selected PF6 is not reacted before reverse event')

        trial, _ = self.reaction.build_trial(sys, candidate)
        if trial is None:
            raise RuntimeError(f'{candidate.direction} trial construction failed')
        R_trial = jnp.asarray(R)
        if candidate.direction == 'forward':
            p = int(self.reaction.phosphorus_index(candidate))
            R_trial = self.reaction.prepare_probe(
                R_trial, candidate, sigma_p=float(trial['sigmas'][p]),
                sigma_f=float(trial['sigmas'][candidate.leave_F]),
                disp_fn=ff.disp_fn, shift_fn=self.shift_fn)
        else:
            R_trial = self.reaction.prepare_reverse_probe(
                R_trial, candidate, disp_fn=ff.disp_fn, shift_fn=self.shift_fn)

        reacted = np.asarray(sys.pf6_reacted, dtype=bool).copy()
        leave = np.asarray(sys.pf6_leave_f, dtype=np.int32).copy()
        partners = np.asarray(sys.pf6_li_partner, dtype=np.int32).copy()
        k = candidate.k_pf6
        forward = candidate.direction == 'forward'
        reacted[k] = forward
        leave[k] = candidate.leave_F if forward else -1
        partners[k] = candidate.li_idx if forward else -1
        next_sys = make_system_state_from_trial(trial, reacted, leave, partners)
        next_ff = build_trial_forcefield(R_trial, self.box, trial, ff)

        charge_delta = float(np.sum(np.asarray(next_sys.charges, dtype=np.float64))) - self.initial_charge
        if abs(charge_delta) > 1e-4:
            raise AssertionError(f'Charge conservation failed: {charge_delta:+.6g} e')
        expected_bonds = self.initial_bond_count - (1 if forward else 0)
        expected_angles = self.initial_angle_count - (5 if forward else 0)
        if len(next_sys.bonds[0]) != expected_bonds or len(next_sys.angles[0]) != expected_angles:
            raise AssertionError('Unexpected number of bonds/angles after reaction')
        if not forward:
            verify_restoration(self.initial_system, next_sys)

        key, relaxed_R, relaxed_v, nlist = langevin_relax_with_nlist(
            key, R_trial, velocities, ff=next_ff, shift_fn=self.shift_fn,
            masses=self.masses, temperature_k=self.cfg.temperature_k,
            kb_real=self.cfg.kb_real, dt_ps=self.cfg.relaxation_dt_ps,
            relaxation_time_ps=self.cfg.relaxation_time_ps,
            thermostat_tau_ps=self.cfg.relaxation_tau_ps)
        next_ff.nlist = nlist
        energy = float(next_ff.energy_fn(relaxed_R, nlist)['total'])
        if not np.isfinite(energy) or not bool(jnp.all(jnp.isfinite(relaxed_v))):
            raise FloatingPointError('Nonfinite relaxed energy/velocities')
        if not forward:
            p = int(self.reaction.pf6_atoms[k, 0])
            pf_distance = float(jnp.linalg.norm(next_ff.disp_fn(relaxed_R[p], relaxed_R[candidate.leave_F])))
            if not np.isfinite(pf_distance) or pf_distance > 3.0:
                raise AssertionError(f'Unstable restored P-F geometry: {pf_distance:.3f} A')

        event = {
            'direction': candidate.direction, 'k_pf6': int(k),
            'li_idx': int(candidate.li_idx), 'leave_F': int(candidate.leave_F),
            'd_pf': float(candidate.d_pf), 'd_lif': float(candidate.d_lif),
            'sigma': float(reaction_coordinate(d_pf=candidate.d_pf, d_lif=candidate.d_lif)),
        }
        self.events.append({'candidate': event, 'energy_after_relax_kcal_mol': energy,
                            'charge_delta_e': charge_delta,
                            'reacted_count': int(reacted.sum())})
        print(f'[forced {candidate.direction}] PF6={k} P-F={candidate.d_pf:.3f} A '
              f'PE_after_relax={energy:.6f} kcal/mol reacted={int(reacted.sum())}', flush=True)
        info = {'mode': 'forced_roundtrip', 'accepted_event': event,
                'accepted_events': [event], 'n_candidates': 1, 'n_accepted_this_check': 1,
                'relaxation_time_ps': self.cfg.relaxation_time_ps,
                '_relaxed_velocities': relaxed_v}
        return key, True, next_ff, next_sys, info, relaxed_R

    def verify_final(self, result):
        if len(self.events) != 2 or [x['candidate']['direction'] for x in self.events] != ['forward', 'reverse']:
            raise AssertionError('Did not complete exactly one forward/reverse pair')
        if result.accepted_events != 2:
            raise AssertionError(
              f'Expected 2 accepted events, got {result.accepted_events}'
            )
        verify_restoration(self.initial_system, result.sys)
        if not bool(jnp.all(jnp.isfinite(result.final_md_state.position))):
            raise AssertionError('Nonfinite final MD coordinates')
        if not bool(jnp.all(jnp.isfinite(result.final_md_state.velocity))):
            raise AssertionError('Nonfinite final MD velocities')
        final_energy = float(result.ff.energy_fn(result.final_md_state.position, result.ff.nlist)['total'])
        if not np.isfinite(final_energy):
            raise AssertionError('Nonfinite final potential energy')
        return {'status': 'PASS', 'configured_steps': self.cfg.steps, 'accepted_events': result.accepted_events,
                'relaxation_time_ps': result.cumulative_relaxation_time_ps,
                'final_energy_kcal_mol': final_energy, 'events': self.events}

    def verify_final(self, result):
        if len(self.events) != 2:
            raise AssertionError(
               f'Expected 2 forced events, got {len(self.events)}'
            )

        directions = [e['candidate']['direction'] for e in self.events]
        if directions != ['forward', 'reverse']:
            raise AssertionError(
               f'Unexpected event sequence: {directions}'
            )

        if result.accepted_events != 2:
            raise AssertionError(
               f'Expected 2 accepted events, got {result.accepted_events}'
            )

        # Verify exact topology and parameter restoration.
        verify_restoration(self.initial_system, result.sys)

        # Verify final NVT state.
        positions = result.final_md_state.position
        velocities = result.final_md_state.velocity

        if not bool(jnp.all(jnp.isfinite(positions))):
            raise AssertionError('Nonfinite final MD coordinates')

        if not bool(jnp.all(jnp.isfinite(velocities))):
            raise AssertionError('Nonfinite final MD velocities')

        final_energy = float(
            result.ff.energy_fn(positions, result.ff.nlist)['total']
        )

        if not np.isfinite(final_energy):
            raise AssertionError('Nonfinite final potential energy')

        # The diagnostic performs exactly two forced relaxations.
        total_relaxation_ps = (
            len(self.events) * self.cfg.relaxation_time_ps
        )

        return {
            'status': 'PASS',
            'configured_steps': self.cfg.steps,
            'accepted_events': result.accepted_events,
            'relaxation_time_ps': total_relaxation_ps,
            'final_energy_kcal_mol': final_energy,
            'events': self.events,
        }

