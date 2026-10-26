"""Generic output observers; no reaction-specific CSV schema in the driver."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass

import jax.numpy as jnp

from .md_driver import MDObservation


class EnergyCSVObserver:
    """Write one energy sample at step zero and after each reaction check."""

    def __init__(self, file):
        self.writer = csv.writer(file)
        self.writer.writerow(('step', 'md_time_ps', 'relaxation_time_ps',
                              'potential_energy', 'kinetic_energy', 'event_count'))

    def on_observation(self, observation: MDObservation) -> None:
        st = observation.state
        potential = float(observation.ff.energy_fn(st.position, observation.ff.nlist)['total'])
        kinetic = float(0.5 * jnp.sum(st.momentum * st.velocity))
        self.writer.writerow((
            observation.step, observation.md_time_ps,
            observation.relaxation_time_ps, potential, kinetic,
            len(observation.outcome.events) if observation.outcome else 0,
        ))


class EventJSONLObserver:
    """Write accepted event data without assuming LiPF6 or A2 fields."""

    def __init__(self, file):
        self.file = file

    def on_observation(self, observation: MDObservation) -> None:
        outcome = observation.outcome
        if outcome is None:
            return
        for event in outcome.events:
            fields = asdict(event) if is_dataclass(event) else dict(event)
            self.file.write(json.dumps({
                'step': observation.step,
                'md_time_ps': observation.md_time_ps,
                'event': fields,
            }) + '\n')
