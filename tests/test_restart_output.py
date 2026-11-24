"""Restart boundaries, actual force-field rebuild, and LiPF6 output replay."""
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md import md_driver
from reactive_md.config import SimConfig
from reactive_md.engine.contracts import MDObservation, ReactionOutcome
from reactive_md.reactions.lipf6_output import LiPF6OutputObserver
from test_checkpoint_restart import run, system, forcefield


def test_partial_checkpoint_cannot_be_extended(tmp_path, monkeypatch):
    path = tmp_path / 'partial.pkl'
    run(3, checkpoint_path=path)
    original = path.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid continuation must fail before reconstruction')
    monkeypatch.setattr(md_driver, '_rebuild_forcefield', forbidden)
    with pytest.raises(ValueError, match='nonaligned reaction boundary'):
        run(6, restart_path=path, checkpoint_path=path)
    assert path.read_bytes() == original


def test_partial_checkpoint_same_target_is_allowed(tmp_path, monkeypatch):
    path = tmp_path / 'partial.pkl'
    first = run(3, checkpoint_path=path)
    monkeypatch.setattr(md_driver, '_rebuild_forcefield',
                        lambda state, sys, options, offset, box:
                        forcefield(state.position, sys))
    resumed = run(3, restart_path=path)
    assert resumed.steps_done == first.steps_done == 3
    np.testing.assert_array_equal(resumed.key, first.key)
    np.testing.assert_array_equal(resumed.final_md_state.position,
                                  first.final_md_state.position)


def test_real_forcefield_restart(tmp_path):
    from jax_md import space
    from reactive_md.checkpoint import load_checkpoint

    positions = jnp.array([[2., 2., 2.], [3.2, 2., 2.]], dtype=jnp.float32)
    box = jnp.full(3, 20., dtype=positions.dtype)
    sys = replace(system(),
                  bonds=(np.array([[0, 1]], np.int32),
                         np.array([10.], np.float32), np.array([1.], np.float32)),
                  molecule_id=jnp.array([0, 0]),
                  torsions=(np.zeros((0, 4), np.int32), np.zeros(0, np.float32),
                            np.zeros(0, np.int32), np.zeros(0, np.float32)),
                  impropers=(np.zeros((0, 4), np.int32), np.zeros(0, np.float32),
                             np.zeros(0, np.int32), np.zeros(0, np.float32)))
    options = dict(r_cut=5., dr_threshold=.5, use_shift_lj=True)
    offset = 3.25
    _, shift = space.periodic(box)

    class Handler:
        def attempt(self, context):
            key, _ = jax.random.split(context.key)
            return ReactionOutcome(key, context.positions, context.velocities,
                                   context.system, context.ff)

    def simulate(steps, **kwargs):
        ff = md_driver._rebuild_forcefield(SimpleNamespace(position=positions),
                                           sys, options, offset, box)
        return md_driver.run_md_nvt(
            jax.random.PRNGKey(7), cfg=SimConfig(steps=steps, check_every=2, dt=.001),
            init_positions=positions, masses=jnp.ones(2), shift_fn=shift,
            ff=ff, sys=sys, box=box, reaction_handler=Handler(),
            energy_offset=offset, **kwargs)

    path = tmp_path / 'real.pkl'
    full = simulate(4)
    simulate(2, checkpoint_path=path)
    assert load_checkpoint(path)['energy_offset'] == offset
    resumed = simulate(4, restart_path=path)
    np.testing.assert_array_equal(resumed.key, full.key)
    for a, b in zip(jax.tree_util.tree_leaves(resumed.final_md_state),
                    jax.tree_util.tree_leaves(full.final_md_state)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    for result in (full, resumed):
        energy = result.ff.energy_fn(result.final_md_state.position, result.ff.nlist)
        assert float(energy['offset']) == offset
    np.testing.assert_allclose(
        resumed.ff.energy_fn(resumed.final_md_state.position, resumed.ff.nlist)['total'],
        full.ff.energy_fn(full.final_md_state.position, full.ff.nlist)['total'],
        rtol=1e-6, atol=1e-7)


LABELS = ('dump', 'events', 'candidates')


def observer(stack, directory, *, resume=False, dump_every=2, omit=None, alternate=None):
    directory.mkdir(exist_ok=True)
    streams = {}
    for label in LABELS:
        path = directory / (label + ('.other' if label == alternate else '.txt'))
        streams[label] = None if label == omit else stack.enter_context(
            path.open('r+' if resume else 'w+', encoding='utf-8'))
    def writer(**kwargs):
        kwargs['file'].write(f"{kwargs['step']} {kwargs['unwrapped_positions'].tolist()}\n")
    return LiPF6OutputObserver(cfg=SimConfig(check_every=2, dump_every=dump_every),
        box=jnp.full(3, 10.), masses=jnp.ones(1), dump_writer=writer,
        dump_file=streams['dump'], event_file=streams['events'],
        candidate_file=streams['candidates'], resume=resume)


def emit(obs, step, x):
    pos = jnp.array([[x, 0., 0.]])
    state = SimpleNamespace(position=pos, velocity=jnp.zeros_like(pos))
    ff = SimpleNamespace(disp_fn=lambda a, b: (a-b+5.) % 10.-5.,
                         energy_fn=lambda R, n: {'total': jnp.sum(R*0)}, nlist=None)
    sys = SimpleNamespace(pf6_reacted=jnp.array([step >= 2]))
    event = dict(k_pf6=0, li_idx=0, leave_F=0, d_lif=1., d_pf=2., rank=1)
    info = dict(mode='metropolis', accepted_event=event, candidate_records=[event])
    outcome = None if step == 0 else ReactionOutcome(
        None, pos, state.velocity, sys, ff, events=(event,), metadata=info)
    record = MDObservation(step, step*.001, 0., state, ff, sys, outcome)
    if step:
        obs.on_before_reaction(record)
    obs.on_observation(record)


def contents(directory):
    return {label: (directory / (label+'.txt')).read_bytes() for label in LABELS}


def test_output_rollback_and_replay(tmp_path):
    full, split = tmp_path/'full', tmp_path/'split'
    with ExitStack() as stack:
        obs = observer(stack, full)
        for step, x in [(0, 9.), (2, 1.), (4, 2.)]:
            emit(obs, step, x)
        expected_position = np.asarray(obs.unwrapped_position)
    with ExitStack() as stack:
        obs = observer(stack, split)
        emit(obs, 0, 9.)
        emit(obs, 2, 1.)
        saved = obs.checkpoint_state()
        prefix = contents(split)
        emit(obs, 4, 3.)  # output beyond last durable checkpoint
    with ExitStack() as stack:
        obs = observer(stack, split, resume=True)
        obs.restore_checkpoint_state(saved)
        assert contents(split) == prefix
        assert obs.accepted_events == 1
        emit(obs, 4, 2.)
        assert obs.accepted_events == 2
        np.testing.assert_array_equal(obs.unwrapped_position, expected_position)
    assert contents(split) == contents(full)


@pytest.mark.parametrize('problem,match', [
    ('short', 'shorter'), ('path', 'path differs'),
    ('missing', 'presence differs'), ('frequency', 'dump_every differs'),
])
def test_invalid_output_restore_does_not_truncate_other_files(tmp_path, problem, match):
    with ExitStack() as stack:
        obs = observer(stack, tmp_path)
        emit(obs, 0, 9.)
        emit(obs, 2, 1.)
        saved = obs.checkpoint_state()
        emit(obs, 4, 2.)
    if problem == 'short':
        (tmp_path/'candidates.txt').write_text('')
    if problem == 'path':
        (tmp_path/'candidates.other').write_bytes((tmp_path/'candidates.txt').read_bytes())
    before = contents(tmp_path)
    with ExitStack() as stack:
        obs = observer(stack, tmp_path, resume=True,
                       omit='candidates' if problem == 'missing' else None,
                       alternate='candidates' if problem == 'path' else None,
                       dump_every=4 if problem == 'frequency' else 2)
        with pytest.raises(ValueError, match=match):
            obs.restore_checkpoint_state(saved)
    assert contents(tmp_path) == before
