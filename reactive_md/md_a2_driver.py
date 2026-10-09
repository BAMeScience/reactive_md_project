# reactive_md/md_a2_driver.py
"""Initial NVT driver for the A + A <-> A2 rate benchmark.

This is intentionally separate from the LiPF6 driver.  The published
benchmark's NPT thermostat/barostat will be addressed after this NVT
integration has been validated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate

from .config import SimConfig
from .forcefield import FFBundle
from .reaction import SystemState
from .reactions.a2 import A2Reaction
from .reactions.a2_rate import A2RateEvent, a2_rate_reactive_cycle


@dataclass(frozen=True)
class A2MDRecordedEvent:
    step: int
    event: A2RateEvent


@dataclass(frozen=True)
class A2MDSample:
    """A trajectory observation, taken after a reactive check/relaxation."""
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
    # (MD step, number of A2 molecules); includes step 0.
    population_history: tuple[tuple[int, int], ...]
    samples: tuple[A2MDSample, ...]


def _count_a2(system: SystemState, reaction: A2Reaction) -> int:
    a_atoms = set(int(i) for i in np.asarray(reaction.a_indices))
    bonds = np.asarray(system.bonds[0]).reshape(-1, 2)
    return sum(int(i) in a_atoms and int(j) in a_atoms for i, j in bonds)


def run_md_nvt_a2(
    key,
    *,
    cfg: SimConfig,
    init_positions,
    masses,
    box,
    shift_fn,
    ff: FFBundle,
    sys: SystemState,
    reaction: A2Reaction,
    association_rate_ps: float,
    dissociation_rate_ps: float,
    rng: np.random.Generator,
    relaxation_dt_ps: float = 0.00025,
    relaxation_time_ps: float = 1.0,
    relaxation_tau_ps: float = 0.01,
) -> A2MDRunResult:
    """Run NVT Nose-Hoover MD with rate-based A2 reaction checks.

    At every reactive check, accepted events are applied together and
    relaxed by the A2 rate cycle. The Nose-Hoover state is then
    reinitialized under the *new* force field, so its cached force is
    consistent with the new topology; the relaxed momenta are restored.

    ``cfg.dt`` and the reactive interval are in ps, rates in ps^-1.
    The relaxation time is additional to the counted normal-MD steps.
    """
    if int(cfg.check_every) <= 0:
        raise ValueError("cfg.check_every must be positive.")
    if int(cfg.steps) < 0:
        raise ValueError("cfg.steps must be non-negative.")
    if int(cfg.max_events) < 0:
        raise ValueError("cfg.max_events must be non-negative.")
    if float(cfg.dt) <= 0.0:
        raise ValueError("cfg.dt must be positive.")

    mass = jnp.asarray(masses)
    kT = cfg.kb_real * cfg.temperature_k

    def make_integrator(ff_current):
        def energy_scalar(R, neighbor):
            return ff_current.energy_fn(R, neighbor)["total"]

        return simulate.nvt_nose_hoover(
            energy_scalar,
            shift_fn,
            dt=cfg.dt,
            kT=kT,
            tau=cfg.tau_T,
            mass=mass,
        )

    def make_md_chunk(apply_nvt, neighbor_fn):
        @partial(jax.jit, static_argnames=("chunk",))
        def md_chunk(md_state, nlist, *, chunk):
            def body(carry, _):
                state, nl = carry
                nl = neighbor_fn.update(state.position, nl)
                state = apply_nvt(state, neighbor=nl)
                return (state, nl), None

            (state, nl), _ = jax.lax.scan(
                body, (md_state, nlist), xs=None, length=chunk
            )
            return state, nl

        return md_chunk

    init_nvt, apply_nvt = make_integrator(ff)
    md_chunk = make_md_chunk(apply_nvt, ff.neighbor_fn)

    # Ensure the neighbor list matches the actual initial positions.
    ff.nlist = ff.neighbor_fn.allocate(init_positions)
    key, sub = jax.random.split(key)
    md_state = init_nvt(sub, init_positions, neighbor=ff.nlist)

    accepted_events = 0
    steps_done = 0
    event_history = []
    population_history = [(0, _count_a2(sys, reaction))]
    samples = []
    n_association = 0
    n_dissociation = 0
    relaxation_elapsed_ps = 0.0

    def record_sample(step, n_assoc_candidates=0, n_diss_candidates=0):
        # These energies are in the units of the current force field
        # (kcal/mol for the LJ/OPLS setup). They are measured at the
        # current post-check configuration, not averaged over the chunk.
        potential = float(ff.energy_fn(md_state.position, ff.nlist)["total"])
        kinetic = float(
            0.5 * jnp.sum(mass[:, None] * md_state.velocity ** 2)
        )
        if not np.isfinite(potential) or not np.isfinite(kinetic):
            raise FloatingPointError(
                f"Nonfinite A2 MD energy at step {step}: "
                f"PE={potential}, KE={kinetic}"
            )
        md_time_ps = step * float(cfg.dt)
        samples.append(
            A2MDSample(
                step=step,
                md_time_ps=md_time_ps,
                cumulative_relaxation_time_ps=relaxation_elapsed_ps,
                elapsed_time_ps=md_time_ps + relaxation_elapsed_ps,
                n_a2=_count_a2(sys, reaction),
                n_association=n_association,
                n_dissociation=n_dissociation,
                potential_energy_kcal_mol=potential,
                kinetic_energy_kcal_mol=kinetic,
                n_association_candidates=n_assoc_candidates,
                n_dissociation_candidates=n_diss_candidates,
            )
        )

    record_sample(0)

    while steps_done < cfg.steps and accepted_events < cfg.max_events:
        chunk = min(int(cfg.check_every), int(cfg.steps - steps_done))
        md_state, ff.nlist = md_chunk(md_state, ff.nlist, chunk=chunk)
        steps_done += chunk

        cycle = a2_rate_reactive_cycle(
            key=key,
            R=md_state.position,
            velocities=md_state.velocity,
            box=box,
            system=sys,
            ff=ff,
            reaction=reaction,
            disp_fn=ff.disp_fn,
            shift_fn=shift_fn,
            masses=mass,
            association_rate_ps=association_rate_ps,
            dissociation_rate_ps=dissociation_rate_ps,
            reactive_interval_ps=chunk * cfg.dt,
            temperature_k=cfg.temperature_k,
            kb_real=cfg.kb_real,
            rng=rng,
            relaxation_dt_ps=relaxation_dt_ps,
            relaxation_time_ps=relaxation_time_ps,
            relaxation_tau_ps=relaxation_tau_ps,
        )
        key = cycle.key

        if cycle.events:
            accepted_events += len(cycle.events)
            n_association += sum(
                e.reaction_type == "association" for e in cycle.events
            )
            n_dissociation += sum(
                e.reaction_type == "dissociation" for e in cycle.events
            )
            # A single Langevin relaxation follows each accepted batch.
            relaxation_elapsed_ps += float(relaxation_time_ps)
            event_history.extend(
                A2MDRecordedEvent(step=steps_done, event=event)
                for event in cycle.events
            )

            ff = cycle.ff
            sys = cycle.system

            # The Langevin helper updates its neighbor list *before*
            # each step; allocate at the final relaxed coordinates.
            ff.nlist = ff.neighbor_fn.allocate(cycle.positions)

            # Reinitialize the NVT state to recompute the cached force
            # for the new topology. Keep the relaxed velocities instead
            # of the freshly sampled Nose-Hoover velocities.
            init_nvt, apply_nvt = make_integrator(ff)
            md_chunk = make_md_chunk(apply_nvt, ff.neighbor_fn)
            key, sub = jax.random.split(key)
            md_state = init_nvt(sub, cycle.positions, neighbor=ff.nlist)
            md_state = replace(
                md_state,
                momentum=jnp.asarray(cycle.velocities) * mass[:, None],
            )

        population_history.append((steps_done, _count_a2(sys, reaction)))
        record_sample(
            steps_done,
            cycle.n_association_candidates,
            cycle.n_dissociation_candidates,
        )

    return A2MDRunResult(
        final_md_state=md_state,
        ff=ff,
        sys=sys,
        accepted_events=accepted_events,
        steps_done=steps_done,
        event_history=tuple(event_history),
        population_history=tuple(population_history),
        samples=tuple(samples),
    )

