"""Backward-compatible A2 runner built on the chemistry-neutral MD engine.

All NVT stepping, reaction dispatch, thermostat synchronization, and neighbor
list recovery happen in ``run_md_nvt``.  This module only configures the A2
reaction handler and records A2-specific outputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..config import SimConfig
from ..forcefield import FFBundle
from ..md_driver import MDObservation, run_md_nvt
from ..reaction import SystemState
from .a2 import A2Reaction
from .a2_handlers import A2MetropolisHandler, A2RateHandler
from .a2_metropolis import A2MCEvent
from .a2_rate import A2RateEvent


@dataclass(frozen=True)
class A2MDRecordedEvent:
    step: int
    event: A2RateEvent | A2MCEvent
    reaction_active_md_time_ps: float
    reaction_time_ps: float
    relaxation_end_time_ps: float


@dataclass(frozen=True)
class A2MDSample:
    step: int
    md_time_ps: float
    cumulative_relaxation_time_ps: float
    elapsed_time_ps: float
    n_a2: int
    n_association: int
    n_dissociation: int
    potential_energy_kcal_mol: float
    kinetic_energy_kcal_mol: float
    n_association_candidates: int
    n_dissociation_candidates: int


@dataclass
class A2MDRunResult:
    final_md_state: object
    ff: FFBundle
    sys: SystemState
    accepted_events: int
    steps_done: int
    event_history: tuple[A2MDRecordedEvent, ...]
    population_history: tuple[tuple[int, int], ...]
    samples: tuple[A2MDSample, ...]
    attempted_mc_moves: int = 0
    rejected_mc_moves: int = 0


def _count_a2(system: SystemState, reaction: A2Reaction) -> int:
    a_atoms = set(int(i) for i in np.asarray(reaction.a_indices))
    bonds = np.asarray(system.bonds[0]).reshape(-1, 2)
    return sum(int(i) in a_atoms and int(j) in a_atoms for i, j in bonds)


class A2OutputObserver:
    """Own A2 history, RDF and paired LAMMPS dump frames.

    Observations are delivered *after* reaction checks.  The engine also
    supplies pre-reaction positions so the observer can write both sides of
    accepted events without participating in reaction selection or stepping.
    """

    def __init__(
        self, *, reaction, cfg, box, masses, rdf_observer=None,
        dump_file=None, dump_writer=None, atom_ids=None, atom_types=None,
        dump_every_checks=1, dump_bonds_file=None, dump_frames_file=None,
    ):
        self.reaction = reaction
        self.cfg = cfg
        self.box = box
        self.mass = jnp.asarray(masses)
        self.rdf_observer = rdf_observer
        self.dump_file = dump_file
        self.dump_writer = dump_writer
        self.atom_ids = atom_ids
        self.atom_types = atom_types
        self.dump_every_checks = dump_every_checks
        self.dump_bonds_file = dump_bonds_file
        self.dump_frames_file = dump_frames_file
        self.event_history = []
        self.population_history = []
        self.samples = []
        self.n_association = 0
        self.n_dissociation = 0
        self.attempted_mc_moves = 0
        self.rejected_mc_moves = 0
        self.checks_seen = 0
        self.dump_frame_index = 0
        self.previous_position = None
        self.unwrapped_position = None
        self.previous_system = None
        self.box_np = np.asarray(box, dtype=np.float64).reshape(3)

    def _advance_unwrapped(self, positions):
        pos = np.asarray(jax.device_get(positions), dtype=np.float64)
        delta = pos - self.previous_position
        delta -= self.box_np * np.rint(delta / self.box_np)
        self.unwrapped_position += delta
        self.previous_position = pos.copy()

    def _write_frame(self, step, phase, positions, unwrapped, system):
        if self.dump_file is None:
            return
        self.dump_writer(
            file=self.dump_file,
            step=step,
            box=self.box,
            atom_ids=self.atom_ids,
            atom_types=self.atom_types,
            masses=np.asarray(jax.device_get(self.mass)),
            wrapped_positions=np.asarray(jax.device_get(positions)),
            unwrapped_positions=np.asarray(unwrapped),
        )
        self.dump_file.flush()
        if self.dump_frames_file is not None:
            self.dump_frames_file.write(
                f"{self.dump_frame_index},{int(step)},"
                f"{step * float(self.cfg.dt):.9f},{phase},"
                f"{_count_a2(system, self.reaction)}\n"
            )
            self.dump_frames_file.flush()
        if self.dump_bonds_file is not None:
            bonds = np.asarray(system.bonds[0], dtype=np.int64).reshape(-1, 2)
            for ai, aj in bonds:
                self.dump_bonds_file.write(
                    f"{self.dump_frame_index},{int(step)},{phase},"
                    f"{int(ai)+1},{int(aj)+1}\n"
                )
            self.dump_bonds_file.flush()
        self.dump_frame_index += 1

    def on_observation(self, observation: MDObservation) -> None:
        step = observation.step
        outcome = observation.outcome
        system = observation.system
        state = observation.state
        metadata = outcome.metadata if outcome is not None else None
        metadata = metadata or {}
        events = outcome.events if outcome is not None else ()

        if self.previous_position is None:
            self.previous_position = np.asarray(
                jax.device_get(state.position), dtype=np.float64
            ).copy()
            self.unwrapped_position = self.previous_position.copy()
            self._write_frame(0, 'initial', state.position,
                              self.unwrapped_position, system)
        else:
            self.checks_seen += 1
            pre_positions = observation.pre_reaction_positions
            if pre_positions is None:
                raise ValueError('A2 observer requires pre-reaction positions')
            self._advance_unwrapped(pre_positions)
            pre_unwrapped = self.unwrapped_position.copy()
            if self.dump_file is not None and (
                self.checks_seen % self.dump_every_checks == 0 or events
            ):
                self._write_frame(step, 'pre_check', pre_positions,
                                  pre_unwrapped, self.previous_system)

            if events:
                old_relaxation = observation.relaxation_time_ps - float(
                    outcome.relaxation_time_ps
                )
                self.n_association += sum(
                    event.reaction_type == 'association' for event in events
                )
                self.n_dissociation += sum(
                    event.reaction_type == 'dissociation' for event in events
                )
                self.event_history.extend(
                    A2MDRecordedEvent(
                        step=step,
                        event=event,
                        reaction_active_md_time_ps=observation.md_time_ps,
                        reaction_time_ps=observation.md_time_ps + old_relaxation,
                        relaxation_end_time_ps=(
                            observation.md_time_ps + observation.relaxation_time_ps
                        ),
                    )
                    for event in events
                )
                self._advance_unwrapped(state.position)
                phase = ('post_relaxation' if outcome.reinitialize_thermostat
                         else 'post_reaction')
                self._write_frame(step, phase, state.position,
                                  self.unwrapped_position, system)

        attempts = metadata.get('attempts', ())
        self.attempted_mc_moves += len(attempts)
        self.rejected_mc_moves += sum(not attempt.accepted for attempt in attempts)
        potential = float(observation.ff.energy_fn(
            state.position, observation.ff.nlist
        )['total'])
        kinetic = float(0.5 * jnp.sum(self.mass[:, None] * state.velocity ** 2))
        if not np.isfinite(potential) or not np.isfinite(kinetic):
            raise FloatingPointError(
                f'Nonfinite A2 MD energy at step {step}: PE={potential}, KE={kinetic}'
            )
        self.population_history.append((step, _count_a2(system, self.reaction)))
        self.samples.append(A2MDSample(
            step=step,
            md_time_ps=observation.md_time_ps,
            cumulative_relaxation_time_ps=observation.relaxation_time_ps,
            elapsed_time_ps=(observation.md_time_ps + observation.relaxation_time_ps),
            n_a2=_count_a2(system, self.reaction),
            n_association=self.n_association,
            n_dissociation=self.n_dissociation,
            potential_energy_kcal_mol=potential,
            kinetic_energy_kcal_mol=kinetic,
            n_association_candidates=metadata.get('n_association_candidates', 0),
            n_dissociation_candidates=metadata.get('n_dissociation_candidates', 0),
        ))
        if self.rdf_observer is not None:
            self.rdf_observer(
                step=step, md_time_ps=observation.md_time_ps,
                positions=state.position, system=system,
            )
        self.previous_system = system


def run_md_nvt_a2(
    key, *, cfg: SimConfig, init_positions, masses, box, shift_fn,
    ff: FFBundle, sys: SystemState, reaction: A2Reaction,
    association_rate_ps: float | None = None,
    dissociation_rate_ps: float | None = None,
    rng: np.random.Generator | None = None,
    reaction_mode: str = 'rate',
    mc_association_proposal_probability: float = 0.5,
    relaxation_dt_ps: float = 0.00025,
    relaxation_time_ps: float = 1.0,
    relaxation_tau_ps: float = 0.01,
    rdf_observer=None,
    dump_file=None,
    dump_writer=None,
    atom_ids=None,
    atom_types=None,
    dump_every_checks=1,
    dump_bonds_file=None,
    dump_frames_file=None,
) -> A2MDRunResult:
    """Compatibility API; the generic engine performs all integration."""
    if reaction_mode not in ('rate', 'metropolis'):
        raise ValueError('reaction_mode must be rate or metropolis')
    if rng is None:
        raise ValueError('rng is required')
    if reaction_mode == 'rate' and (
        association_rate_ps is None or dissociation_rate_ps is None
    ):
        raise ValueError('rate mode requires both association and dissociation rates')
    if int(cfg.check_every) <= 0:
        raise ValueError('cfg.check_every must be positive')
    if int(cfg.steps) < 0 or int(cfg.max_events) < 0:
        raise ValueError('cfg.steps and cfg.max_events must be nonnegative')
    if float(cfg.dt) <= 0:
        raise ValueError('cfg.dt must be positive')
    if dump_every_checks < 1:
        raise ValueError('dump_every_checks must be >= 1')
    if (dump_file is None) != (dump_writer is None):
        raise ValueError('dump_file and dump_writer must be supplied together')
    if dump_file is not None and (atom_ids is None or atom_types is None):
        raise ValueError('atom_ids and atom_types are required for dumps')

    if reaction_mode == 'rate':
        handler = A2RateHandler(
            reaction=reaction, rng=rng,
            association_rate_ps=association_rate_ps,
            dissociation_rate_ps=dissociation_rate_ps,
            relaxation_dt_ps=relaxation_dt_ps,
            relaxation_time_ps=relaxation_time_ps,
            relaxation_tau_ps=relaxation_tau_ps,
        )
    else:
        handler = A2MetropolisHandler(
            reaction=reaction, rng=rng,
            association_proposal_probability=mc_association_proposal_probability,
        )
    observer = A2OutputObserver(
        reaction=reaction, cfg=cfg, box=box, masses=masses,
        rdf_observer=rdf_observer, dump_file=dump_file,
        dump_writer=dump_writer, atom_ids=atom_ids, atom_types=atom_types,
        dump_every_checks=dump_every_checks,
        dump_bonds_file=dump_bonds_file, dump_frames_file=dump_frames_file,
    )
    result = run_md_nvt(
        key, cfg=cfg, init_positions=init_positions, masses=masses,
        box=box, shift_fn=shift_fn, ff=ff, sys=sys,
        reaction_handler=handler, observers=(observer,),
    )
    return A2MDRunResult(
        final_md_state=result.final_md_state,
        ff=result.ff,
        sys=result.sys,
        accepted_events=result.accepted_events,
        steps_done=result.steps_done,
        event_history=tuple(observer.event_history),
        population_history=tuple(observer.population_history),
        samples=tuple(observer.samples),
        attempted_mc_moves=observer.attempted_mc_moves,
        rejected_mc_moves=observer.rejected_mc_moves,
    )
