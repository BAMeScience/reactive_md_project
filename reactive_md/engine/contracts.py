"""Chemistry-independent interfaces for reactive molecular dynamics.

Reaction handlers own chemistry; observers own output.  The NVT engine only
consumes these contracts.  Importing this module does not require JAX-MD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..forcefield import FFBundle


@dataclass(frozen=True)
class ReactionContext:
    """Immutable inputs for one reaction check; no chemistry-specific fields."""

    key: object
    step: int
    interval_ps: float
    positions: object
    velocities: object
    system: object
    ff: FFBundle
    box: object
    shift_fn: object
    masses: object
    temperature_k: float
    kb_real: float


@dataclass(frozen=True)
class ReactionOutcome:
    """One reaction check, including any reaction-specific coordinate changes.

    ``reinitialize_thermostat`` is an explicit compatibility choice for
    legacy kinetic models that apply a separate post-reaction trajectory.
    Ordinary fixed-coordinate MC should leave it False.

    ``preserve_momenta`` tells the NVT stepper that a reaction changes only
    topology/forces, not particle momenta. This avoids a velocity-to-momentum
    conversion when the handler did not apply any velocity impulse.
    """

    key: object
    positions: object
    velocities: object
    system: object
    ff: FFBundle
    events: tuple = ()
    relaxation_time_ps: float = 0.0
    reinitialize_thermostat: bool = False
    metadata: object = None
    preserve_momenta: bool = False


class ReactionHandler(Protocol):
    """A chemistry-specific reaction strategy injected into the MD engine."""

    def attempt(self, context: ReactionContext) -> ReactionOutcome: ...


@dataclass(frozen=True)
class MDObservation:
    step: int
    md_time_ps: float
    relaxation_time_ps: float
    state: object
    ff: FFBundle
    system: object
    outcome: ReactionOutcome | None
    pre_reaction_positions: object = None


class MDObserver(Protocol):
    def on_observation(self, observation: MDObservation) -> None: ...


@dataclass
class MDRunResult:
    final_md_state: object
    ff: FFBundle
    sys: object
    accepted_events: int
    steps_done: int
    key: object
    cumulative_relaxation_time_ps: float
