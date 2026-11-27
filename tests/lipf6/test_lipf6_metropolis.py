"""Reaction-cycle contracts with real LiPF6 topology and controlled energies."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md import reaction as rm
from reactive_md.reactions.lipf6 import LiPF6Reaction, ReactionCandidate
from reactive_md.reactions.templates_pf5 import make_lif_template, make_pf5_template


@pytest.fixture
def cycle(monkeypatch):
    bonds = np.array([[0, f] for f in range(1, 7)], dtype=np.int32)
    angles = np.array([[i, 0, j] for i in range(1, 7)
                       for j in range(i + 1, 7)], dtype=np.int32)
    empty = (np.empty((0, 4), np.int32), np.empty(0),
             np.empty(0, np.int32), np.empty(0))
    sys = rm.SystemState(
        bonds=(bonds, np.full(6, 370.4589), np.full(6, 1.606)),
        angles=(angles, np.full(15, 139.2209), np.full(15, 1.5707963)),
        torsions=empty, impropers=empty,
        charges=np.array([1.34] + [-.39]*6 + [1.]),
        sigmas=np.array([3.74] + [3.118]*6 + [2.12645]),
        epsilons=np.array([.2] + [.061]*6 + [.0182792]),
        molecule_id=np.array([1]*7 + [2], np.int32),
        pf6_reacted=jnp.array([False]),
    )
    reaction = LiPF6Reaction(
        pf6_atoms=np.array([[0, 1, 2, 3, 4, 5, 6]], np.int32),
        li_atoms=np.array([7], np.int32),
        atom_types=np.array([6] + [7]*6 + [8], np.int32),
        pf5=make_pf5_template(), lif=make_lif_template(),
        p_type=6, f_type=7, li_type=8,
    )
    reaction.set_reference_state(sys)
    R = jnp.array([[0.,0.,0.], [1.6,0.,0.], [-1.6,0.,0.],
                   [0.,1.6,0.], [0.,-1.6,0.], [0.,0.,-1.6],
                   [0.,0.,1.6], [0.,0.,6.]])
    candidate = ReactionCandidate(0, 7, 6, 4.4, 1.6)
    monkeypatch.setattr(reaction, 'find_candidates', lambda *a, **k: [candidate])
    # Keep PRNG splitting real; control the two uniform draws.
    monkeypatch.setattr(rm.jax.random, 'uniform', lambda *a, **k: jnp.asarray(.5))
    c = SimpleNamespace(sys=sys, reaction=reaction, R=R, calls=[],
                        before=5., after=4., invalid=None, mode='classical')
    c.old_list, c.trial_list, c.relaxed_list = object(), object(), object()
    c.md_energy, c.md_neighbors = object(), object()
    c.v = jnp.zeros_like(R)
    c.relaxed_v = jnp.ones_like(R)
    def energy_before(positions, neighbors):
        assert neighbors is c.old_list
        c.calls.append(('classical_before', positions))
        return {'total': jnp.asarray(c.before)}
    def energy_after(positions, neighbors):
        assert neighbors is c.relaxed_list
        c.calls.append(('classical_after', positions))
        return {'total': jnp.asarray(c.after)}
    disp = lambda a, b: a - b
    c.ff = SimpleNamespace(disp_fn=disp, nlist=c.old_list,
        neighbor_fn=SimpleNamespace(update=lambda R, nl: nl), energy_fn=energy_before)
    c.trial_ff = SimpleNamespace(disp_fn=disp, nlist=c.trial_list, energy_fn=energy_after)
    def build(R, box, trial, ff):
        c.calls.append(('build', R))
        assert ff is c.ff
        return c.trial_ff
    def relax(key, positions, velocities, **kwargs):
        c.calls.append(('relax', positions))
        assert kwargs['ff'] is c.trial_ff
        assert kwargs['md_energy_fn'] is (c.md_energy if c.mode == 'mace' else None)
        assert kwargs['md_neighbor_fn'] is (c.md_neighbors if c.mode == 'mace' else None)
        np.testing.assert_array_equal(velocities, c.v)
        c.relaxed_R = positions + .125
        v = c.relaxed_v
        if c.invalid == 'positions':
            c.relaxed_R = c.relaxed_R.at[0, 0].set(jnp.nan)
        if c.invalid == 'velocities':
            v = v.at[0, 0].set(jnp.inf)
        return key, c.relaxed_R, v, c.relaxed_list
    def mace_energy(positions):
        first = not any(name == 'mace_before' for name, _ in c.calls)
        c.calls.append(('mace_before' if first else 'mace_after', positions))
        return c.before if first else c.after
    monkeypatch.setattr(rm, 'build_trial_forcefield', build)
    monkeypatch.setattr(rm, 'langevin_relax_with_nlist', relax)
    c.evaluator = SimpleNamespace(energy=mace_energy)
    def run(**overrides):
        kwargs = dict(shift_fn=lambda R, d: R+d, ff=c.ff, sys=c.sys,
            reaction=c.reaction, beta=1., sigma_mid=-1000., velocities=c.v,
            relaxation_time_ps=.01,
            mc_energy_evaluator=None if c.mode == 'classical' else c.evaluator,
            md_energy_fn=c.md_energy if c.mode == 'mace' else None,
            md_neighbor_fn=c.md_neighbors if c.mode == 'mace' else None)
        kwargs.update(overrides)
        return rm.maybe_react_one_event(jax.random.PRNGKey(0), c.R,
                                       jnp.eye(3)*20, **kwargs)
    c.run = run
    return c


def assert_unchanged(c, result):
    _, accepted, ff, sys, info, R = result
    assert not accepted
    assert ff is c.ff and sys is c.sys
    assert ff.nlist is c.old_list
    np.testing.assert_array_equal(R, c.R)
    assert 'accepted_event' not in info
    assert '_relaxed_velocities' not in info


@pytest.mark.parametrize('mode', ['classical', 'hybrid', 'mace'])
@pytest.mark.parametrize('after', [4., 7., float('nan'), float('inf')])
def test_mc_energy_decision(cycle, mode, after):
    c = cycle
    c.mode, c.after = mode, after
    result = c.run()
    _, accepted, ff, sys, info, R = result
    prefix = 'classical' if mode == 'classical' else 'mace'
    assert [name for name, _ in c.calls] == [prefix+'_before', 'build', 'relax', prefix+'_after']
    np.testing.assert_array_equal(c.calls[0][1], c.R)
    np.testing.assert_array_equal(c.calls[-1][1], c.relaxed_R)
    assert info['p_sigma'] == 1.
    expected_p = min(1., np.exp(5.-after)) if np.isfinite(after) else 0.
    assert info['p_metropolis'] == pytest.approx(expected_p)
    assert info['p_total'] == pytest.approx(expected_p)
    if after == 4.:
        assert accepted and ff is c.trial_ff and sys is not c.sys
        assert ff.nlist is (c.trial_list if mode == 'mace' else c.relaxed_list)
        np.testing.assert_array_equal(R, c.relaxed_R)
        np.testing.assert_array_equal(info['_relaxed_velocities'], c.relaxed_v)
        assert info['relaxation_time_ps'] == .01
        assert info['accepted_event']['direction'] == 'forward'
        np.testing.assert_array_equal(sys.pf6_reacted, [True])
        np.testing.assert_array_equal(sys.pf6_leave_f, [6])
        np.testing.assert_array_equal(sys.pf6_li_partner, [7])
        assert len(sys.bonds[0]) == 5 and len(sys.angles[0]) == 10
        np.testing.assert_allclose(np.sum(sys.charges), np.sum(c.sys.charges), atol=1e-6)
    else:
        assert_unchanged(c, result)
    np.testing.assert_array_equal(c.sys.pf6_reacted, [False])
    assert len(c.sys.bonds[0]) == 6
    assert c.ff.nlist is c.old_list


@pytest.mark.parametrize('invalid', ['positions', 'velocities'])
def test_mc_rejects_nonfinite_relaxation(cycle, invalid):
    c = cycle
    c.invalid = invalid
    result = c.run()
    assert_unchanged(c, result)
    assert result[4]['reason'] == 'nonfinite_relaxed_geometry'
    assert [name for name, _ in c.calls] == ['classical_before', 'build', 'relax']


def test_mc_no_candidates(cycle, monkeypatch):
    c = cycle
    monkeypatch.setattr(c.reaction, 'find_candidates', lambda *a, **k: [])
    result = c.run()
    assert_unchanged(c, result)
    assert result[4]['reason'] == 'no_candidates'
    assert result[4]['candidate_records'] == []
    assert not c.calls


def test_mc_sigma_gate_skips_expensive_work(cycle):
    c = cycle
    result = c.run(sigma_mid=1000.)
    assert_unchanged(c, result)
    assert result[4]['reason'] == 'sigma_gate_rejected'
    assert not c.calls


def test_mc_reverse_restores_topology_and_bookkeeping(cycle, monkeypatch):
    c = cycle
    original = c.sys
    trial, _ = c.reaction.build_trial(original, ReactionCandidate(0, 7, 6, 4.4, 1.6))
    c.sys = rm.make_system_state_from_trial(trial, np.array([True]), [6], [7])
    monkeypatch.setattr(c.reaction, 'find_candidates', lambda *a, **k: [])
    monkeypatch.setattr(c.reaction, 'find_reverse_candidates', lambda *a, **k: [
        ReactionCandidate(0, 7, 6, 4.4, 1.6, 'reverse')])
    _, accepted, _, sys, info, _ = c.run(allow_reverse=True)
    assert accepted and info['accepted_event']['direction'] == 'reverse'
    np.testing.assert_array_equal(sys.pf6_reacted, [False])
    np.testing.assert_array_equal(sys.pf6_leave_f, [-1])
    np.testing.assert_array_equal(sys.pf6_li_partner, [-1])
    for name in ('charges', 'sigmas', 'epsilons', 'molecule_id'):
        np.testing.assert_allclose(getattr(sys, name), getattr(original, name), atol=1e-6)
    for name in ('bonds', 'angles', 'torsions', 'impropers'):
        for actual, expected in zip(getattr(sys, name), getattr(original, name)):
            np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize('sigma_mid,expected_stages', [
    (1000., ['candidate_search']),
    (-1000., ['candidate_search', 'reaction_energy_before', 'trial_topology',
              'trial_forcefield', 'trial_probe', 'trial_relaxation',
              'reaction_energy_after']),
])
def test_mc_performance_preserves_outcome(cycle, monkeypatch, capsys,
                                          sigma_mid, expected_stages):
    import json
    from reactive_md.performance import PerformanceTimer

    monkeypatch.setenv('REACTIVE_MD_PERFORMANCE', '0')
    baseline = cycle.run(sigma_mid=sigma_mid)
    calls = [name for name, _ in cycle.calls]
    cycle.calls.clear()
    assert '[REACTIVE_MD PERF]' not in capsys.readouterr().out
    monkeypatch.setenv('REACTIVE_MD_PERFORMANCE', '1')
    with PerformanceTimer().measure('reaction_attempt', 100):
        measured = cycle.run(sigma_mid=sigma_mid)
    prefix = '[REACTIVE_MD PERF] '
    rows = [json.loads(line[len(prefix):]) for line in capsys.readouterr().out.splitlines()
            if line.startswith(prefix)]
    assert [r['stage'] for r in rows] == expected_stages + ['reaction_attempt']
    assert all(r['step'] == 100 and r['status'] == 'ok' for r in rows)
    assert all(r['parent_id'] == rows[-1]['span_id'] for r in rows[:-1])
    assert [name for name, _ in cycle.calls] == calls
    assert baseline[1] == measured[1]
    np.testing.assert_array_equal(baseline[0], measured[0])
    np.testing.assert_array_equal(baseline[-1], measured[-1])
    np.testing.assert_array_equal(baseline[3].pf6_reacted, measured[3].pf6_reacted)
    assert baseline[4]['p_total'] == measured[4]['p_total']
