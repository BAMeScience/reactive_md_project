"""Stable chemistry-independent entry point for reactive NVT dynamics.

Core data contracts live in :mod:`reactive_md.engine.contracts`; the
Nose-Hoover stepper lives in :mod:`reactive_md.engine.nvt`.  The generic
orchestration below deliberately knows nothing about A2 or LiPF6.

Historical chemistry-specific entry points remain available lazily.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from .config import SimConfig
from .engine import (
    ReactionContext, ReactionOutcome, ReactionHandler,
    MDObservation, MDObserver, MDRunResult, NVTStepper,
)

if TYPE_CHECKING:
    from .forcefield import FFBundle


def run_md_nvt(
    key,
    *,
    cfg: SimConfig,
    init_positions,
    masses,
    shift_fn,
    ff: FFBundle,
    sys: object,
    reaction_handler: ReactionHandler,
    box=None,
    observers: tuple[MDObserver, ...] = (),
) -> MDRunResult:
    """Generic NVT driver. Adding chemistry does not require editing it.

    Handlers may update positions, velocities, force fields and topology, but
    must return the original objects when no reaction is accepted. Observer
    callbacks receive post-check snapshots; step zero is included.
    """
    if cfg.steps < 0 or cfg.max_events < 0:
        raise ValueError('steps and max_events must be nonnegative')
    stepper = NVTStepper(cfg=cfg, ff=ff, shift_fn=shift_fn, masses=masses)
    key, sub = jax.random.split(key)
    state = stepper.initialize(sub, init_positions)
    accepted_events = 0
    steps_done = 0
    relaxation_elapsed = 0.0

    def notify(outcome=None, pre_positions=None):
        observation = MDObservation(
            step=steps_done,
            md_time_ps=steps_done * float(cfg.dt),
            relaxation_time_ps=relaxation_elapsed,
            state=state,
            ff=ff,
            system=sys,
            outcome=outcome,
            pre_reaction_positions=pre_positions,
        )
        for observer in observers:
            observer.on_observation(observation)

    notify()
    while steps_done < cfg.steps and accepted_events < cfg.max_events:
        chunk = min(int(cfg.check_every), int(cfg.steps - steps_done))
        state, ff.nlist = stepper.advance(state, ff.nlist, chunk=chunk)
        steps_done += chunk
        before = state.position
        # Optional pre-reaction observation: chemistry-specific diagnostics and
        # periodic trajectory frames must see the old topology and force field.
        # The regular on_observation callback remains post-reaction so existing
        # observers (including A2) retain their original contract.
        before_observation = MDObservation(
            step=steps_done,
            md_time_ps=steps_done * float(cfg.dt),
            relaxation_time_ps=relaxation_elapsed,
            state=state,
            ff=ff,
            system=sys,
            outcome=None,
            pre_reaction_positions=before,
        )
        for observer in observers:
            before_callback = getattr(observer, 'on_before_reaction', None)
            if before_callback is not None:
                before_callback(before_observation)
        outcome = reaction_handler.attempt(ReactionContext(
            key=key,
            step=steps_done,
            interval_ps=chunk * float(cfg.dt),
            positions=state.position,
            velocities=state.velocity,
            system=sys,
            ff=ff,
            box=box,
            shift_fn=shift_fn,
            masses=stepper.mass,
            temperature_k=cfg.temperature_k,
            kb_real=cfg.kb_real,
        ))
        key = outcome.key
        if outcome.events:
            accepted_events += len(outcome.events)
            ff, sys = outcome.ff, outcome.system
            if outcome.reinitialize_thermostat:
                key, sub = jax.random.split(key)
            else:
                sub = None
            state, ff.nlist = stepper.synchronize(state, outcome, key=sub)
            relaxation_elapsed += float(outcome.relaxation_time_ps)
        notify(outcome, before)
    return MDRunResult(
        state, ff, sys, accepted_events, steps_done, key, relaxation_elapsed
    )


# Compatibility with imports from versions where chemistry lived in this file.
# Kept lazy to avoid importing chemistry into the generic MD engine.
def __getattr__(name):
    if name == 'run_md_nvt_with_reactions' or name == 'RunResult':
        from .reactions import lipf6_driver
        return getattr(lipf6_driver, name)
    if name in ('run_md_nvt_a2', 'A2MDRecordedEvent', 'A2MDSample', 'A2MDRunResult'):
        from .reactions import a2_driver
        return getattr(a2_driver, name)
    raise AttributeError(name)
