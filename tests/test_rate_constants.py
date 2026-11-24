"""Regression tests for the existing linear LiPF6 rate law."""
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from reactive_md.reaction import resolve_rate_ps, tst_rate_ps, maybe_react_rate_events


def test_tst_default_prefactor():
    assert tst_rate_ps(temperature_k=300., activation_energy_eV=0.) == pytest.approx(6.250985736998272)


def test_tst_custom_prefactor_and_barrier():
    # Choose a barrier of kBT: the Arrhenius factor is exp(-1).
    assert tst_rate_ps(temperature_k=400., activation_energy_eV=8.617333262145e-5 * 400.,
                       prefactor_ps=2.) == pytest.approx(2. / math.e)


@pytest.mark.parametrize('explicit, barrier, expected', [
    (2.5, None, 2.5), (0., None, 0.), (None, None, 0.), (None, 0., 3.),
])
def test_resolve_rate(explicit, barrier, expected):
    assert resolve_rate_ps(reaction_rate_ps=explicit, activation_energy_eV=barrier,
                           temperature_k=300., prefactor_ps=3.) == pytest.approx(expected)


def test_resolve_rate_rejects_two_inputs():
    with pytest.raises(ValueError, match='not both'):
        resolve_rate_ps(reaction_rate_ps=0., activation_energy_eV=0., temperature_k=300.)


def rate_attempt(rate, interval, reverse=0.):
    R = jnp.zeros((1, 3))
    ff = SimpleNamespace(disp_fn=lambda a, b: a-b)
    system = SimpleNamespace(pf6_reacted=jnp.array([False]),
                             pf6_leave_f=jnp.array([-1]), pf6_li_partner=jnp.array([-1]))
    reaction = SimpleNamespace(find_candidates=lambda *a, **kw: [],
                               find_reverse_candidates=lambda *a, **kw: [])
    return maybe_react_rate_events(
        jax.random.PRNGKey(0), R, jnp.eye(3), shift_fn=None, ff=ff, sys=system,
        reaction=reaction, reaction_rate_ps=rate, activation_energy_eV=None,
        temperature_k=300., prefactor_ps=None, reactive_interval_ps=interval,
        reverse_rate_ps=reverse,
    )


@pytest.mark.parametrize('rate, interval', [(0., .5), (2., .5)])
def test_linear_rate_probability_boundaries(rate, interval):
    _, accepted, _, _, info, _ = rate_attempt(rate, interval)
    assert not accepted
    assert info['k_rate_ps'] == rate
    assert info['dt_reactive_ps'] == interval


@pytest.mark.parametrize('rate', [-1., 2.01, float('nan'), float('inf')])
def test_invalid_linear_rate_probability(rate):
    with pytest.raises(ValueError, match='probability must lie between 0 and 1'):
        rate_attempt(rate, .5)


@pytest.mark.parametrize('reverse, message', [
    (-1., 'finite and nonnegative'), (float('nan'), 'finite and nonnegative'),
    (2.01, 'Reverse linear rate probability exceeds 1'),
])
def test_invalid_reverse_rate(reverse, message):
    with pytest.raises(ValueError, match=message):
        rate_attempt(0., .5, reverse)


@pytest.mark.parametrize('direction, expected', [('forward', .1), ('reverse', .3)])
def test_linear_rate_gate_uses_direction_specific_rate(monkeypatch, direction, expected):
    from reactive_md.reactions.lipf6 import ReactionCandidate

    reverse = direction == 'reverse'
    candidate = ReactionCandidate(k_pf6=0, li_idx=2, leave_F=1,
                                  d_lif=2., d_pf=2., direction=direction)
    R = jnp.zeros((3, 3))
    ff = SimpleNamespace(disp_fn=lambda a, b: a-b)
    system = SimpleNamespace(pf6_reacted=jnp.array([reverse]),
                             pf6_leave_f=jnp.array([1 if reverse else -1]),
                             pf6_li_partner=jnp.array([2 if reverse else -1]))
    reaction = SimpleNamespace(
        find_candidates=lambda *a, **kw: [] if reverse else [candidate],
        find_reverse_candidates=lambda *a, **kw: [candidate] if reverse else [],
    )
    draws = iter([.1, .9])  # Pass sigma gate; reject kinetic gate.
    monkeypatch.setattr(jax.random, 'uniform', lambda *a, **kw: next(draws))
    _, accepted, ff_out, sys_out, info, R_out = maybe_react_rate_events(
        jax.random.PRNGKey(0), R, jnp.eye(3), shift_fn=None, ff=ff, sys=system,
        reaction=reaction, reaction_rate_ps=.4, activation_energy_eV=None,
        temperature_k=300., prefactor_ps=None, reactive_interval_ps=.5,
        reverse_rate_ps=1.2,
    )
    assert not accepted
    assert ff_out is ff and sys_out is system and R_out is R
    assert info['p_sigma'] == pytest.approx(.5)
    assert info['p_total'] == pytest.approx(expected)
    assert next(draws, None) is None
