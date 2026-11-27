"""Opt-in timing must not change driver results or wait when disabled."""
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.performance import PerformanceTimer


PREFIX = '[REACTIVE_MD PERF] '


def records(capsys):
    return [json.loads(line[len(PREFIX):])
            for line in capsys.readouterr().out.splitlines()
            if line.startswith(PREFIX)]


@pytest.mark.parametrize('enabled', [False, True])
def test_timer_wait_and_output(monkeypatch, capsys, enabled):
    monkeypatch.setenv('REACTIVE_MD_PERFORMANCE', '1' if enabled else '0')
    calls = []
    leaf = SimpleNamespace(block_until_ready=lambda: calls.append('wait'))
    with PerformanceTimer().measure('md_chunk', 10, chunk_steps=10) as wait:
        wait({'array': leaf, 'other': None})
    rows = records(capsys)
    assert calls == (['wait'] if enabled else [])
    if enabled:
        assert len(rows) == 1
        assert rows[0]['step'] == 10
        assert rows[0]['status'] == 'ok'
        assert rows[0]['elapsed_s'] >= 0
        assert rows[0]['steps_per_s'] > 0
    else:
        assert rows == []


def test_timer_preserves_exception(monkeypatch, capsys):
    monkeypatch.setenv('REACTIVE_MD_PERFORMANCE', '1')
    with pytest.raises(RuntimeError, match='original failure'):
        with PerformanceTimer().measure('md_chunk', 5, chunk_steps=5):
            raise RuntimeError('original failure')
    row, = records(capsys)
    assert row['status'] == 'error'
    assert 'steps_per_s' not in row


def test_driver_timing_preserves_results(monkeypatch, capsys):
    from reactive_md import md_driver, checkpoint
    from reactive_md.config import SimConfig
    from typing import NamedTuple

    class State(NamedTuple):
        position: object
        velocity: object

    class Stepper:
        def __init__(self, *, cfg, ff, shift_fn, masses):
            self.mass = masses
            self.nlist = jnp.array([0])

        def initialize(self, key, positions):
            return State(positions, jnp.zeros_like(positions))

        def advance(self, state, nlist, *, chunk):
            return State(state.position + chunk, state.velocity), nlist

        def synchronize(self, state, outcome, *, key=None):
            return State(outcome.positions, outcome.velocities), self.nlist

    class Handler:
        def attempt(self, ctx):
            return md_driver.ReactionOutcome(
                key=ctx.key, positions=ctx.positions + 1,
                velocities=ctx.velocities, system=ctx.system, ff=ctx.ff,
                events=({'accepted': True},), relaxation_time_ps=0.25,
            )

    monkeypatch.setattr(md_driver, 'NVTStepper', Stepper)
    saved_steps = []
    monkeypatch.setattr(checkpoint, 'save_checkpoint',
                        lambda path, **kwargs: saved_steps.append(kwargs['steps_done']))

    def run(enabled):
        monkeypatch.setenv('REACTIVE_MD_PERFORMANCE', '1' if enabled else '0')
        return md_driver.run_md_nvt(
            jax.random.PRNGKey(0), cfg=SimConfig(steps=3, check_every=2),
            init_positions=jnp.zeros((2, 3)), masses=jnp.ones(2),
            shift_fn=None, ff=SimpleNamespace(nlist=None, nb_options=SimpleNamespace(
                r_cut=2.0, dr_threshold=0.5)), sys=object(),
            reaction_handler=Handler(), checkpoint_path='unused.pkl',
            checkpoint_every_steps=2,
        )

    baseline = run(False)
    assert records(capsys) == []
    measured = run(True)
    rows = records(capsys)
    np.testing.assert_array_equal(baseline.final_md_state.position,
                                  measured.final_md_state.position)
    np.testing.assert_array_equal(baseline.key, measured.key)
    assert baseline.accepted_events == measured.accepted_events == 2
    assert baseline.steps_done == measured.steps_done == 3
    assert baseline.cumulative_relaxation_time_ps == measured.cumulative_relaxation_time_ps
    assert saved_steps == [2, 3, 2, 3]
    assert [(r['step'], r['chunk_steps']) for r in rows if r['stage'] == 'md_chunk'] == [(2, 2), (3, 1)]
    assert [r['step'] for r in rows if r['stage'] == 'checkpoint'] == [2, 3]
    assert [r['stage'] for r in rows if r['step'] == 2] == [
        'md_chunk', 'before_reaction_observers', 'reaction_attempt',
        'reaction_sync', 'after_reaction_observers', 'checkpoint',
    ]
    assert all(r['status'] == 'ok' and r['elapsed_s'] >= 0 for r in rows)


def test_nested_spans_inherit_step_and_restore_context(monkeypatch, capsys):
    monkeypatch.setenv('REACTIVE_MD_PERFORMANCE', '1')
    with PerformanceTimer().measure('reaction_attempt', 100):
        with PerformanceTimer().measure('trial_relaxation'):
            with PerformanceTimer().measure('langevin_chunk', chunk_steps=50):
                pass
    with PerformanceTimer().measure('checkpoint', 100):
        pass
    chunk, relax, reaction, checkpoint = records(capsys)
    assert chunk['step'] == relax['step'] == reaction['step'] == 100
    assert chunk['parent_id'] == relax['span_id']
    assert relax['parent_id'] == reaction['span_id']
    assert reaction['parent_id'] is None and checkpoint['parent_id'] is None
    assert chunk['steps_per_s'] > 0
    assert len({r['span_id'] for r in (chunk, relax, reaction, checkpoint)}) == 4


def test_nested_error_restores_context(monkeypatch, capsys):
    monkeypatch.setenv('REACTIVE_MD_PERFORMANCE', '1')
    with pytest.raises(ValueError):
        with PerformanceTimer().measure('outer', 5):
            with PerformanceTimer().measure('inner'):
                raise ValueError('failure')
    with PerformanceTimer().measure('standalone'):
        pass
    inner, outer, standalone = records(capsys)
    assert inner['status'] == outer['status'] == 'error'
    assert inner['parent_id'] == outer['span_id']
    assert standalone['parent_id'] is None and standalone['step'] is None
