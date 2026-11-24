"""Checkpoint I/O and aligned-boundary continuation with real NVT dynamics."""
from dataclasses import replace
from types import SimpleNamespace
from typing import NamedTuple
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md import checkpoint, md_driver
from reactive_md.config import SimConfig
from reactive_md.reaction import SystemState


class Neighbor(NamedTuple):
    idx: object
    did_buffer_overflow: object


class NeighborFn:
    def allocate(self, positions, *, extra_capacity=0):
        return Neighbor(jnp.zeros((len(positions), 1 + extra_capacity), dtype=jnp.int32),
                        jnp.asarray(False))

    def update(self, positions, neighbor):
        return neighbor


def system():
    def empty(n):
        return (np.zeros((0, n), np.int32), np.zeros(0), np.zeros(0))
    four = (*empty(4), np.zeros(0))
    return SystemState(bonds=empty(2), angles=empty(3), torsions=four,
                       impropers=four, charges=jnp.zeros(2), sigmas=jnp.ones(2),
                       epsilons=jnp.ones(2), molecule_id=jnp.arange(2),
                       pf6_reacted=jnp.array([False]),
                       pf6_leave_f=jnp.array([-1]), pf6_li_partner=jnp.array([-1]))


def forcefield(positions, sys):
    stiffness = 2.0 if bool(sys.pf6_reacted[0]) else 1.0
    def energy(positions, neighbor):
        return {'total': 0.5 * stiffness * jnp.sum(positions ** 2)}
    neighbors = NeighborFn()
    return SimpleNamespace(energy_fn=energy, neighbor_fn=neighbors,
                           nlist=neighbors.allocate(positions),
                           nb_options=SimpleNamespace(r_cut=8., dr_threshold=.5,
                                                      use_shift_lj=True))


class Handler:
    def __init__(self, react=False):
        self.react = react

    def attempt(self, context):
        key, _ = jax.random.split(context.key)
        sys, ff = context.system, context.ff
        events = ()
        if self.react and context.step == 2:
            sys = replace(sys, pf6_reacted=jnp.array([True]),
                          pf6_leave_f=jnp.array([1]), pf6_li_partner=jnp.array([0]))
            ff = forcefield(context.positions, sys)
            events = ('reaction',)
        return md_driver.ReactionOutcome(
            key=key, positions=context.positions, velocities=context.velocities,
            system=sys, ff=ff, events=events,
            relaxation_time_ps=.01 if events else 0., preserve_momenta=True)


class Recorder:
    def __init__(self):
        self.steps = []

    def on_observation(self, observation):
        self.steps.append(observation.step)

    def checkpoint_state(self):
        return {'steps': list(self.steps)}

    def restore_checkpoint_state(self, state):
        self.steps = list(state['steps'])


def run(steps, *, react=False, **kwargs):
    positions = jnp.array([[1., 0., 0.], [-1., 0., 0.]])
    sys = system()
    return md_driver.run_md_nvt(
        jax.random.PRNGKey(19), cfg=SimConfig(steps=steps, check_every=2, dt=.001),
        init_positions=positions, masses=jnp.ones(2),
        shift_fn=lambda R, dR, **kw: R + dR,
        ff=forcefield(positions, sys), sys=sys, reaction_handler=Handler(react),
        **kwargs)


class ExternalNeighbor(NamedTuple):
    idx: object
    did_buffer_overflow: object


class ExternalNeighborFn(NeighborFn):
    def allocate(self, positions, *, extra_capacity=0):
        return ExternalNeighbor(*super().allocate(positions, extra_capacity=extra_capacity))


def external_backend(identity='surrogate-v1'):
    def energy(positions, neighbor):
        assert isinstance(neighbor, ExternalNeighbor)
        return 1.5 * jnp.sum(positions ** 2)
    return dict(md_energy_fn=energy, md_neighbor_fn=ExternalNeighborFn(),
                md_backend_id=identity)


@pytest.mark.parametrize('external', [False, True])
@pytest.mark.parametrize('react', [False, True])
def test_aligned_restart_matches_uninterrupted(tmp_path, monkeypatch, react, external):
    path = tmp_path / 'state.pkl'
    full_obs, first_obs, resumed_obs = Recorder(), Recorder(), Recorder()
    backend = external_backend if external else dict
    full = run(6, react=react, observers=(full_obs,), **backend())
    first = run(2, react=react, observers=(first_obs,), checkpoint_path=path,
                checkpoint_every_steps=2, **backend())
    saved = checkpoint.load_checkpoint(path)
    assert saved['steps_done'] == 2
    if external:
        assert saved['metadata']['md_backend_id'] == 'surrogate-v1'
    assert saved['accepted_events'] == int(react)
    assert saved['observer_state']['0']['steps'] == [0, 2]
    np.testing.assert_array_equal(saved['key'], first.key)
    rebuilt = []

    def rebuild(state, sys, options, offset, box):
        assert options == {'r_cut': 8., 'dr_threshold': .5, 'use_shift_lj': True}
        assert offset is None
        assert box is None
        assert bool(sys.pf6_reacted[0]) == react
        rebuilt.append(sys)
        return forcefield(state.position, sys)

    monkeypatch.setattr(md_driver, '_rebuild_forcefield', rebuild)
    def forbidden_initialize(*args, **kwargs):
        pytest.fail('Restart must not reinitialize the thermostat')
    monkeypatch.setattr(md_driver.NVTStepper, 'initialize', forbidden_initialize)
    resumed = run(6, react=react, observers=(resumed_obs,), restart_path=path, **backend())
    assert isinstance(resumed.ff.nlist, Neighbor)
    assert len(rebuilt) == 1
    assert resumed.steps_done == full.steps_done == 6
    assert resumed.accepted_events == full.accepted_events == int(react)
    assert resumed.cumulative_relaxation_time_ps == full.cumulative_relaxation_time_ps
    assert resumed_obs.steps == full_obs.steps == [0, 2, 4, 6]
    np.testing.assert_array_equal(resumed.key, full.key)
    leaves_a, tree_a = jax.tree_util.tree_flatten(resumed.final_md_state)
    leaves_b, tree_b = jax.tree_util.tree_flatten(full.final_md_state)
    assert tree_a == tree_b
    for a, b in zip(leaves_a, leaves_b):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    for name in ('pf6_reacted', 'pf6_leave_f', 'pf6_li_partner'):
        np.testing.assert_array_equal(getattr(resumed.sys, name), getattr(full.sys, name))


@pytest.mark.parametrize('field,value', [
    ('dt', .002), ('temperature_k', 500.), ('check_every', 3),
    ('float_dtype', 'different'), ('masses', np.array([1., 2.])),
    ('box', np.ones(3)), ('md_unit_system', 'metal'),
])
def test_restart_metadata_rejects_changes(field, value):
    cfg = SimConfig()
    current = md_driver._restart_metadata(cfg, None, jnp.ones(2), jnp.float32)
    saved = dict(current, **{field: value})
    with pytest.raises(ValueError, match=f'current {field}'):
        md_driver._validate_restart_metadata(saved, current)


@pytest.mark.parametrize('argument', ['checkpoint_path', 'restart_path'])
def test_external_checkpoint_requires_identity(tmp_path, argument):
    with pytest.raises(ValueError, match='requires md_backend_id'):
        run(2, md_energy_fn=lambda R, neighbor: jnp.sum(R**2),
            md_neighbor_fn=NeighborFn(), **{argument: tmp_path / 'state.pkl'})


@pytest.mark.parametrize('replacement', ['changed', 'classical', 'external'])
def test_restart_rejects_backend_change(tmp_path, monkeypatch, replacement):
    path = tmp_path / 'state.pkl'
    initial = {} if replacement == 'external' else external_backend()
    run(2, checkpoint_path=path, **initial)
    original = path.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('Backend mismatch must fail before force-field reconstruction')
    monkeypatch.setattr(md_driver, '_rebuild_forcefield', forbidden)
    resumed = {} if replacement == 'classical' else external_backend('surrogate-v2')
    with pytest.raises(ValueError, match='Checkpoint.*(metadata|md_backend_id)'):
        run(4, restart_path=path, checkpoint_path=path, **resumed)
    assert path.read_bytes() == original


@pytest.mark.parametrize('interval', [0, -2, 3])
def test_invalid_checkpoint_interval(tmp_path, interval):
    with pytest.raises(ValueError, match='positive multiple of check_every'):
        run(2, checkpoint_path=tmp_path / 'state.pkl', checkpoint_every_steps=interval)


def test_atomic_write_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'state.pkl'
    args = dict(md_state={'position': jnp.ones((2, 3))}, system_state=system(),
                key=jax.random.PRNGKey(0), steps_done=2, accepted_events=0,
                relaxation_elapsed_ps=0., observer_state=None,
                forcefield_options={'r_cut': 8., 'dr_threshold': .5, 'use_shift_lj': True})
    checkpoint.save_checkpoint(path, **args)
    original = path.read_bytes()
    def fail(payload, stream, **kwargs):
        stream.write(b'incomplete')
        raise OSError('simulated write failure')
    monkeypatch.setattr(pickle, 'dump', fail)
    with pytest.raises(OSError, match='simulated write failure'):
        checkpoint.save_checkpoint(path, **dict(args, steps_done=4))
    assert path.read_bytes() == original
    assert checkpoint.load_checkpoint(path)['steps_done'] == 2
    assert list(tmp_path.iterdir()) == [path]
