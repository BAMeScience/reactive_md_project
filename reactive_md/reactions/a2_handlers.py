"""A2 reaction plugins for the chemistry-neutral ``run_md_nvt`` engine.

The legacy ``run_md_nvt_a2`` adapter remains available for its historical
sampling, RDF and LAMMPS output formats.  New simulations may use these
handlers with the generic engine and independent observers.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..md_driver import ReactionContext, ReactionOutcome
from .a2 import A2Reaction
from .a2_rate import a2_rate_reactive_cycle
from .a2_metropolis import a2_metropolis_reactive_cycle


@dataclass
class A2RateHandler:
    reaction: A2Reaction
    rng: np.random.Generator
    association_rate_ps: float
    dissociation_rate_ps: float
    relaxation_dt_ps: float = 0.00025
    relaxation_time_ps: float = 1.0
    relaxation_tau_ps: float = 0.01

    def attempt(self, context: ReactionContext) -> ReactionOutcome:
        result = a2_rate_reactive_cycle(
            key=context.key,
            R=context.positions,
            velocities=context.velocities,
            box=context.box,
            system=context.system,
            ff=context.ff,
            reaction=self.reaction,
            disp_fn=context.ff.disp_fn,
            shift_fn=context.shift_fn,
            masses=context.masses,
            association_rate_ps=self.association_rate_ps,
            dissociation_rate_ps=self.dissociation_rate_ps,
            reactive_interval_ps=context.interval_ps,
            temperature_k=context.temperature_k,
            kb_real=context.kb_real,
            rng=self.rng,
            relaxation_dt_ps=self.relaxation_dt_ps,
            relaxation_time_ps=self.relaxation_time_ps,
            relaxation_tau_ps=self.relaxation_tau_ps,
        )
        return ReactionOutcome(
            key=result.key,
            positions=result.positions,
            velocities=result.velocities,
            system=result.system,
            ff=result.ff,
            events=result.events,
            relaxation_time_ps=self.relaxation_time_ps if result.events else 0.0,
            reinitialize_thermostat=bool(result.events),
            metadata={
                'n_association_candidates': result.n_association_candidates,
                'n_dissociation_candidates': result.n_dissociation_candidates,
            },
        )


@dataclass
class A2MetropolisHandler:
    reaction: A2Reaction
    rng: np.random.Generator
    association_proposal_probability: float = 0.5

    def attempt(self, context: ReactionContext) -> ReactionOutcome:
        result = a2_metropolis_reactive_cycle(
            key=context.key,
            R=context.positions,
            velocities=context.velocities,
            box=context.box,
            system=context.system,
            ff=context.ff,
            reaction=self.reaction,
            disp_fn=context.ff.disp_fn,
            temperature_k=context.temperature_k,
            kb_real=context.kb_real,
            rng=self.rng,
            association_proposal_probability=self.association_proposal_probability,
        )
        return ReactionOutcome(
            key=result.key,
            positions=result.positions,
            velocities=result.velocities,
            system=result.system,
            ff=result.ff,
            events=result.events,
            metadata={
                'attempts': result.attempts,
                'n_association_candidates': result.n_association_candidates,
                'n_dissociation_candidates': result.n_dissociation_candidates,
            },
        )
