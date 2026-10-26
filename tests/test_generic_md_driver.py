"""Fast plugin-contract tests; the MD integrator is replaced by a test double."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from reactive_md.config import SimConfig
from reactive_md import md_driver


class FakeStepper:
    def __init__(self, *, cfg, ff, shift_fn, masses):
        self.ff = ff
        self.mass = jnp.asarray(masses)

    def initialize(self, key, positions):
        self.ff.nlist = 'initial-list'
        return SimpleNamespace(
            position=positions,
            velocity=jnp.zeros_like(positions),
            momentum=jnp.zeros_like(positions),
        )

    def advance(self, state, nlist, *, chunk):
        state.position = state.position + chunk
        return state, nlist

    def synchronize(self, state, outcome, *, key=None):
        self.ff = outcome.ff
        state.position = outcome.positions
        state.velocity = outcome.velocities
        return state, 'updated-list'


class NoReaction:
    def __init__(self):
        self.intervals = []

    def attempt(self, context):
        self.intervals.append(context.interval_ps)
        return md_driver.ReactionOutcome(
            key=context.key, positions=context.positions,
            velocities=context.velocities, system=context.system,
            ff=context.ff,
        )


class SingleReaction:
    def __init__(self):
        self.calls = 0

    def attempt(self, context):
        self.calls += 1
        return md_driver.ReactionOutcome(
            key=context.key, positions=context.positions,
            velocities=context.velocities, system=context.system,
            ff=context.ff, events=({'reaction_type': 'example'},),
            relaxation_time_ps=0.5,
        )


class Recorder:
    def __init__(self):
        self.records = []

    def on_observation(self, observation):
        self.records.append(observation)


def test_generic_driver_no_reaction(monkeypatch):
    monkeypatch.setattr(md_driver, 'NVTStepper', FakeStepper)
    cfg = SimConfig(steps=5, check_every=2, dt=0.01)
    handler, recorder = NoReaction(), Recorder()
    ff = SimpleNamespace(nlist=None)
    result = md_driver.run_md_nvt(
        jax.random.PRNGKey(0), cfg=cfg,
        init_positions=jnp.zeros((2, 3)), masses=jnp.ones(2),
        shift_fn=None, ff=ff, sys=object(),
        reaction_handler=handler, observers=(recorder,),
    )
    assert result.steps_done == 5
    assert result.accepted_events == 0
    assert handler.intervals == [0.02, 0.02, 0.01]
    assert [record.step for record in recorder.records] == [0, 2, 4, 5]


def test_generic_driver_accepted_event(monkeypatch):
    monkeypatch.setattr(md_driver, 'NVTStepper', FakeStepper)
    handler, recorder = SingleReaction(), Recorder()
    result = md_driver.run_md_nvt(
        jax.random.PRNGKey(0),
        cfg=SimConfig(steps=10, check_every=2, max_events=1),
        init_positions=jnp.zeros((2, 3)), masses=jnp.ones(2),
        shift_fn=None, ff=SimpleNamespace(nlist=None), sys=object(),
        reaction_handler=handler, observers=(recorder,),
    )
    assert result.steps_done == 2
    assert result.accepted_events == 1
    assert result.cumulative_relaxation_time_ps == 0.5
    assert len(recorder.records[-1].outcome.events) == 1


def test_optional_pre_reaction_hook_sees_old_topology(monkeypatch):
    monkeypatch.setattr(md_driver, 'NVTStepper', FakeStepper)
    sequence = []
    old_sys, new_sys = object(), object()

    class SwappingHandler:
        def attempt(self, context):
            assert context.system is old_sys
            sequence.append(('attempt', context.step))
            return md_driver.ReactionOutcome(
                key=context.key, positions=context.positions,
                velocities=context.velocities, system=new_sys,
                ff=context.ff, events=({'reaction_type': 'swap'},),
            )

    class PhasedObserver:
        def on_before_reaction(self, observation):
            assert observation.system is old_sys
            assert observation.outcome is None
            sequence.append(('before', observation.step))

        def on_observation(self, observation):
            if observation.step:
                assert observation.system is new_sys
            sequence.append(('after', observation.step))

    result = md_driver.run_md_nvt(
        jax.random.PRNGKey(0),
        cfg=SimConfig(steps=2, check_every=2, max_events=1),
        init_positions=jnp.zeros((2, 3)), masses=jnp.ones(2),
        shift_fn=None, ff=SimpleNamespace(nlist=None), sys=old_sys,
        reaction_handler=SwappingHandler(), observers=(PhasedObserver(),),
    )
    assert sequence == [('after', 0), ('before', 2), ('attempt', 2), ('after', 2)]
    assert result.sys is new_sys
