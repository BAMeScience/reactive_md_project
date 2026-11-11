"""Adapter for the existing LiPF6 reaction callback contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..md_driver import ReactionContext, ReactionOutcome


@dataclass
class LiPF6Handler:
    """Wrap an existing ``reaction_step_fn(key, R, ff, system)`` callback.

    The callback is deliberately kept unchanged to preserve the established
    LiPF6 candidate selection, rate probabilities and logging metadata.
    """

    reaction_step_fn: Callable
    passes_velocities: bool = False

    def attempt(self, context: ReactionContext) -> ReactionOutcome:
        args = (context.key, context.positions, context.ff, context.system)
        if self.passes_velocities:
            args += (context.velocities,)
        key, accepted, ff, system, info, positions = self.reaction_step_fn(*args)
        info = dict(info)
        relaxed_velocities = info.pop("_relaxed_velocities", context.velocities)
        relaxation_time_ps = float(info.pop("relaxation_time_ps", 0.0)) if accepted else 0.0
        events = info.get('accepted_events')
        if events is None:
            event = info.get('accepted_event')
            events = (event,) if event is not None else ()
        if not accepted:
            events = ()
        return ReactionOutcome(
            key=key,
            positions=positions if accepted else context.positions,
            velocities=relaxed_velocities if accepted else context.velocities,
            ff=ff if accepted else context.ff,
            system=system if accepted else context.system,
            events=tuple(events),
            metadata=info,
            relaxation_time_ps=relaxation_time_ps,
            reinitialize_thermostat=bool(accepted and relaxation_time_ps),
        )
