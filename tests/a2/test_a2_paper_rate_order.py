"""Deterministic Figure 3b tests: all candidates, randomized order, one relaxation.

These deliberately test the paper's rate workflow, not a Metropolis kernel.
"""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions import a2_rate as rate
from reactive_md.reaction import SystemState


def make_system(n_atoms=4):
    bonds = (jnp.empty((0, 2), dtype=jnp.int32), jnp.empty(0), jnp.empty(0))
    angles = (jnp.empty((0, 3), dtype=jnp.int32), jnp.empty(0), jnp.empty(0))
    four = (jnp.empty((0, 4), dtype=jnp.int32), jnp.empty(0),
            jnp.empty(0, dtype=jnp.int32), jnp.empty(0))
    return SystemState(bonds, angles, four, four, jnp.zeros(n_atoms),
                       jnp.ones(n_atoms), jnp.ones(n_atoms),
                       jnp.arange(n_atoms), jnp.zeros(0, dtype=bool))


class OrderedRNG:
    def __init__(self, order, draws):
        self.order = np.asarray(order, dtype=int)
        self.draws = iter(draws)
        self.draw_count = 0
        self.permutations = []

    def permutation(self, n):
        assert n == len(self.order)
        self.permutations.append(n)
        return self.order

    def random(self):
        self.draw_count += 1
        return next(self.draws)


def _run(monkeypatch, *, positions, system, rng, association_rate=1.,
         dissociation_rate=0., interval=1.):
    steps = []
    old_ff = SimpleNamespace(nlist=SimpleNamespace(did_buffer_overflow=False))
    new_ff = SimpleNamespace(nlist=SimpleNamespace(did_buffer_overflow=False))

    def translate(R, i, j, **kwargs):
        steps.append(('translate', i, j, kwargs['move_apart']))
        return R

    def build(R, box, trial, ff):
        steps.append(('build', len(trial['bonds'][0])))
        assert ff is old_ff
        return new_ff

    def relax(key, R, velocities, **kwargs):
        steps.append(('relax',))
        assert kwargs['ff'] is new_ff
        return 'next-key', R + 0.01, velocities + 0.02, new_ff.nlist

    monkeypatch.setattr(rate, 'translate_reacting_pair', translate)
    monkeypatch.setattr(rate, 'build_trial_forcefield', build)
    monkeypatch.setattr(rate, 'langevin_relax_with_nlist', relax)
    reaction = A2Reaction(a_indices=np.arange(len(positions)),
                          k_bond_jax=1., r0_bond=1.5,
                          association_cutoff=1.1, bond_energy_offset=-0.5)
    result = rate.a2_rate_reactive_cycle(
        key='original-key', R=positions, velocities=jnp.zeros_like(positions),
        box=jnp.ones(3)*20, system=system, ff=old_ff, reaction=reaction,
        disp_fn=lambda a, b: a-b, shift_fn=lambda x, dx: x+dx,
        masses=jnp.ones(len(positions)), association_rate_ps=association_rate,
        dissociation_rate_ps=dissociation_rate, reactive_interval_ps=interval,
        temperature_k=188., kb_real=0.0019872041, rng=rng)
    return result, steps, old_ff, new_ff


def test_rate_randomized_all_candidates_nonoverlapping_events_one_relax(monkeypatch):
    # Only 01, 12 and 23 are in the association candidate list.
    R = jnp.asarray([[0.,0.,0.], [1.,0.,0.], [2.,0.,0.], [3.,0.,0.]])
    # Try 12 first, then 01, then 23. Once 12 reacts, both others overlap.
    rng = OrderedRNG([1, 0, 2], [0., 0., 0.])
    result, steps, _, new_ff = _run(monkeypatch, positions=R,
                                     system=make_system(), rng=rng)
    assert rng.permutations == [3]
    assert len(result.events) == 1
    assert (result.events[0].i, result.events[0].j) == (1, 2)
    assert rng.draw_count == 1  # Overlapping candidates are not attempted.
    assert steps == [('translate', 1, 2, False), ('build', 1), ('relax',)]
    assert result.ff is new_ff
    assert result.key == 'next-key'
    np.testing.assert_allclose(result.positions, np.asarray(R) + 0.01)


def test_rate_accepts_multiple_disjoint_pairs_but_relaxes_once(monkeypatch):
    R = jnp.asarray([[0.,0.,0.], [1.,0.,0.], [8.,0.,0.], [9.,0.,0.]])
    rng = OrderedRNG([1, 0], [0., 0.])
    result, steps, _, _ = _run(monkeypatch, positions=R,
                               system=make_system(), rng=rng)
    assert len(result.events) == 2
    assert {(e.i, e.j) for e in result.events} == {(0,1), (2,3)}
    assert len(np.asarray(result.system.bonds[0])) == 2
    assert steps[-2:] == [('build', 2), ('relax',)]
    assert sum(step[0] == 'relax' for step in steps) == 1


def test_rate_no_accepted_events_no_translation_or_relaxation(monkeypatch):
    R = jnp.asarray([[0.,0.,0.], [1.,0.,0.], [8.,0.,0.], [9.,0.,0.]])
    rng = OrderedRNG([0, 1], [0.8, 0.8])
    original_system = make_system()
    result, steps, old_ff, _ = _run(monkeypatch, positions=R,
                                    system=original_system, rng=rng,
                                    association_rate=0.5)
    assert not result.events
    assert result.ff is old_ff
    assert result.system is original_system
    assert result.positions is R
    assert result.key == 'original-key'
    assert steps == []
    assert rng.draw_count == 2


@pytest.mark.parametrize('rate_value', [float('nan'), float('inf'), -0.1])
def test_rate_invalid_rate_fails_closed(monkeypatch, rate_value):
    R = jnp.asarray([[0.,0.,0.], [1.,0.,0.]])
    with pytest.raises(ValueError):
        _run(monkeypatch, positions=R, system=make_system(2),
             rng=OrderedRNG([0], [0.]), association_rate=rate_value)


def test_rate_overflowing_relaxation_is_not_committed(monkeypatch):
    R = jnp.asarray([[0.,0.,0.], [1.,0.,0.]])
    rng = OrderedRNG([0], [0.])
    bad_nlist = SimpleNamespace(did_buffer_overflow=True)

    def build(R, box, trial, ff):
        return SimpleNamespace(nlist=bad_nlist)

    def relax(key, R, velocities, **kwargs):
        return 'next-key', R, velocities, bad_nlist

    monkeypatch.setattr(rate, 'build_trial_forcefield', build)
    monkeypatch.setattr(rate, 'langevin_relax_with_nlist', relax)
    with pytest.raises(RuntimeError, match='overflowing neighbor list'):
        rate.a2_rate_reactive_cycle(
            key='original-key', R=R, velocities=jnp.zeros_like(R),
            box=jnp.ones(3)*20, system=make_system(2),
            ff=SimpleNamespace(nlist=None),
            reaction=A2Reaction(a_indices=np.arange(2), k_bond_jax=1.,
                                r0_bond=1.5, association_cutoff=1.1),
            disp_fn=lambda a,b: a-b, shift_fn=lambda x,dx: x+dx,
            masses=jnp.ones(2), association_rate_ps=1.,
            dissociation_rate_ps=0., reactive_interval_ps=1.,
            temperature_k=188., kb_real=0.0019872041, rng=rng)
