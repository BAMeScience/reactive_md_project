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

    def attempt(self, context: ReactionContext) -> ReactionOutcome:
        key, accepted, ff, system, info, positions = self.reaction_step_fn(
            context.key, context.positions, context.ff, context.system
        )
        events = info.get('accepted_events')
        if events is None:
            event = info.get('accepted_event')
            events = (event,) if event is not None else ()
        if not accepted:
            events = ()
        return ReactionOutcome(
            key=key,
            positions=positions if accepted else context.positions,
            velocities=context.velocities,
            ff=ff if accepted else context.ff,
            system=system if accepted else context.system,
            events=tuple(events),
            metadata=info,
        )
