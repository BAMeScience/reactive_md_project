"""LiPF6 plugin migration tests using fake MD snapshots (no JAX-MD physics)."""
from io import StringIO
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from reactive_md.config import SimConfig
from reactive_md.md_driver import MDObservation, MDRunResult, ReactionOutcome
from reactive_md.reactions import lipf6_driver
from reactive_md.reactions.lipf6_output import LiPF6OutputObserver


def _system(n):
    return SimpleNamespace(pf6_reacted=jnp.asarray([True] * n + [False] * (3-n)))


def _state(x):
    pos = jnp.asarray([[x, 0., 0.], [x + 1., 0., 0.]])
    return SimpleNamespace(position=pos, velocity=jnp.zeros_like(pos))


def _ff():
    return SimpleNamespace(
        nlist='nl',
        disp_fn=lambda a, b: b - a,
        energy_fn=lambda r, nl: {'total': jnp.asarray(4.)},
    )


def _event(i):
    return dict(k_pf6=i, li_idx=7, leave_F=8, d_lif=2., d_pf=3.,
                sigma=1., p_rate=.02)


def _obs(*, cfg=None):
    dump, events, candidates = StringIO(), StringIO(), StringIO()
    frames = []

    def writer(**kwargs):
        frames.append((kwargs['step'], np.asarray(kwargs['unwrapped_positions']).copy()))

    observer = LiPF6OutputObserver(
        cfg=cfg or SimConfig(dt=.01, check_every=2, dump_every=2),
        box=jnp.asarray([10., 10., 10.]), masses=jnp.ones(2),
        atom_ids=np.arange(1, 3), atom_types=np.ones(2),
        dump_file=dump, dump_writer=writer, event_file=events,
        candidate_file=candidates,
    )
    return observer, frames, events, candidates


def test_lipf6_rate_multi_event_output_and_unwrapped():
    observer, frames, events, candidates = _obs()
    ff = _ff()
    observer.on_observation(MDObservation(
        0, 0., 0., _state(0.), ff, _system(0), None,
    ))
    observer.on_before_reaction(MDObservation(
        2, .02, 0., _state(.2), ff, _system(0), None,
    ))
    info = {
        'mode': 'rate', 'n_candidates': 2, 'n_accepted_this_check': 2,
        'p_rate': .02, 'k_rate_ps': .01, 'dt_reactive_ps': 2.,
        'accepted_events': [_event(1), _event(2)],
        'candidate_records': [dict(rank=1, **_event(1)),
                              dict(rank=2, **_event(2))],
    }
    outcome = ReactionOutcome(
        key=None, positions=_state(.3).position,
        velocities=_state(.3).velocity, system=_system(2), ff=ff,
        events=tuple(info['accepted_events']), metadata=info,
    )
    observer.on_observation(MDObservation(
        2, .02, 0., _state(.3), ff, _system(2), outcome,
        pre_reaction_positions=_state(.2).position,
    ))
    assert [f[0] for f in frames] == [0, 2, 2]
    np.testing.assert_allclose(frames[-1][1][0, 0], .3, atol=1e-6)
    event_lines = events.getvalue().splitlines()
    assert len(event_lines) == 4  # header, two accepted, one rate_check
    assert '1,rate,2,2' in event_lines[1]
    assert '2,rate,2,2' in event_lines[2]
    assert event_lines[-1].startswith('2,rate_check,,rate,')
    candidate_lines = candidates.getvalue().splitlines()
    assert len(candidate_lines) == 3
    assert candidate_lines[1].endswith(',1')
    assert candidate_lines[2].endswith(',1')
    assert observer.accepted_events == 2


def test_lipf6_rejection_and_no_extra_dump():
    observer, frames, events, candidates = _obs(cfg=SimConfig(
        dt=.01, check_every=2, dump_every=5,
    ))
    ff = _ff()
    observer.on_observation(MDObservation(0, 0., 0., _state(0.), ff, _system(0), None))
    observer.on_before_reaction(MDObservation(2, .02, 0., _state(.2), ff, _system(0), None))
    info = {'mode': 'metropolis', 'candidate_records': [dict(rank=1, **_event(1))]}
    outcome = ReactionOutcome(None, _state(.2).position, _state(.2).velocity,
                              _system(0), ff, metadata=info)
    observer.on_observation(MDObservation(2, .02, 0., _state(.2), ff, _system(0), outcome))
    assert [step for step, _ in frames] == [0]
    assert len(events.getvalue().splitlines()) == 1
    assert candidates.getvalue().splitlines()[-1].endswith(',0')


def test_lipf6_compatibility_runner_delegates_to_generic(monkeypatch):
    calls = []
    ff, system = _ff(), _system(0)

    def fake_run(key, *, cfg, init_positions, masses, shift_fn, ff, sys,
                 box, reaction_handler, observers):
        calls.append((reaction_handler, observers[0]))
        return MDRunResult(_state(0.), ff, sys, 0, 0, key, 0.)

    monkeypatch.setattr(lipf6_driver, 'run_md_nvt', fake_run)
    result = lipf6_driver.run_md_nvt_with_reactions(
        key=None, cfg=SimConfig(steps=0), init_positions=_state(0.).position,
        masses=jnp.ones(2), shift_fn=None, ff=ff, sys=system,
        reaction_step_fn=lambda *args: None,
    )
    assert calls[0][0].__class__.__name__ == 'LiPF6Handler'
    assert isinstance(calls[0][1], LiPF6OutputObserver)
    assert result.ff is ff and result.sys is system
    assert result.accepted_events == 0
