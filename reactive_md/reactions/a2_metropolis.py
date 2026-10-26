"""Fixed-coordinate, single-move Metropolis-Hastings A + A <-> A2.

This is an equilibrium *topology* sampler, not a kinetic reaction model.
The proposal chooses association/dissociation with fixed probabilities, then
chooses uniformly among eligible candidates of that type. A direction with
no candidates is a null proposal (its probability is NOT redistributed).
No deterministic coordinate translation or FIRE/Langevin minimization is used
in the Metropolis decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp
import numpy as np

from reactive_md.reaction import SystemState, build_trial_forcefield
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_rate import make_a2_system_state_from_trial


@dataclass(frozen=True)
class A2MCAttempt:
    reaction_type: str
    i: int
    j: int
    distance: float
    accepted: bool
    probability: float
    delta_energy: float
    log_proposal_ratio: float
    reason: str = ""


@dataclass(frozen=True)
class A2MCEvent:
    reaction_type: str
    i: int
    j: int
    distance: float
    probability: float
    delta_energy: float
    log_proposal_ratio: float


@dataclass(frozen=True)
class A2MCResult:
    key: object
    positions: object
    velocities: object
    ff: object
    system: SystemState
    events: tuple[A2MCEvent, ...]
    n_association_candidates: int
    n_dissociation_candidates: int
    attempts: tuple[A2MCAttempt, ...]


def _finalize_trial(trial: dict, reaction: A2Reaction, n_atoms: int) -> dict:
    """Update molecule connectivity and the *total* A2 energy offset."""
    trial = dict(trial)
    parents = np.arange(n_atoms, dtype=np.int32)

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = int(parents[i])
        return i

    a_set = set(int(i) for i in reaction.a_indices)
    n_a2 = 0
    for pair in np.asarray(trial["bonds"][0], dtype=np.int32).reshape(-1, 2):
        i, j = map(int, pair)
        parents[root(j)] = root(i)
        if i in a_set and j in a_set:
            n_a2 += 1
    trial["molecule_id"] = np.asarray([root(i) for i in range(n_atoms)], dtype=np.int32)
    trial["energy_offset"] = n_a2 * reaction.bond_energy_offset
    return trial


def _valid_nlist(ff, positions, max_retries=12):
    extra_capacity = 0
    for attempt in range(max_retries + 1):
        nl = ff.neighbor_fn.allocate(positions, extra_capacity=extra_capacity)
        if not bool(np.asarray(nl.did_buffer_overflow)):
            return nl
        extra_capacity = max(8, 2 * extra_capacity)
    raise RuntimeError("A2 MC trial neighbor-list allocation overflowed")


def a2_metropolis_reactive_cycle(
    *,
    key,
    R,
    velocities,
    box,
    system: SystemState,
    ff,
    reaction: A2Reaction,
    disp_fn,
    temperature_k: float,
    kb_real: float,
    rng: np.random.Generator,
    association_proposal_probability: float = 0.5,
) -> A2MCResult:
    """Attempt exactly one topology change at fixed positions.

    Target density is proportional to exp(-U/kT), with U from the force
    field including the A2 per-bond offset. No additional equilibrium
    constant/chemical-potential factor is included.
    """
    p_a = float(association_proposal_probability)
    if not (0.0 < p_a < 1.0) or not np.isfinite(p_a):
        raise ValueError("association_proposal_probability must lie strictly between 0 and 1")
    kT = float(kb_real) * float(temperature_k)
    if not np.isfinite(kT) or kT <= 0.0:
        raise ValueError("kb_real * temperature_k must be finite and positive")

    assoc = reaction.find_association_candidates(R, disp_fn, system=system)
    diss = reaction.find_dissociation_candidates(R, disp_fn, system=system)
    do_assoc = bool(rng.random() < p_a)
    candidates = assoc if do_assoc else diss
    kind = "association" if do_assoc else "dissociation"
    if not candidates:
        return A2MCResult(key, R, velocities, ff, system, (), len(assoc), len(diss), ())

    candidate = candidates[int(rng.integers(len(candidates)))]
    i, j = int(candidate.i), int(candidate.j)
    p_forward = p_a if do_assoc else (1.0 - p_a)
    p_reverse = (1.0 - p_a) if do_assoc else p_a
    trial = (reaction.build_association_trial(system, candidate) if do_assoc
             else reaction.build_dissociation_trial(system, candidate))
    trial = _finalize_trial(trial, reaction, len(system.molecule_id))
    trial_system = make_a2_system_state_from_trial(trial, system)

    # Check reverse reachability *at the same coordinates* before evaluating
    # energy. A long bond cannot be dissociated if its newly free pair would
    # be outside the association cutoff: q(reverse) would be zero.
    reverse_candidates = (
        reaction.find_dissociation_candidates(R, disp_fn, system=trial_system)
        if do_assoc else reaction.find_association_candidates(R, disp_fn, system=trial_system)
    )
    reverse_exists = any(
        {int(c.i), int(c.j)} == {i, j} for c in reverse_candidates
    )
    if not reverse_exists:
        attempt = A2MCAttempt(kind, i, j, float(candidate.distance), False,
                              0.0, float("nan"), float("-inf"), "reverse move unavailable")
        return A2MCResult(key, R, velocities, ff, system, (), len(assoc), len(diss), (attempt,))

    log_q_ratio = math.log(p_reverse) - math.log(len(reverse_candidates)) \
        - math.log(p_forward) + math.log(len(candidates))
    trial_ff = build_trial_forcefield(R, box, trial, ff)
    trial_ff.nlist = _valid_nlist(trial_ff, R)
    old_nlist = ff.neighbor_fn.update(R, ff.nlist)
    if bool(np.asarray(old_nlist.did_buffer_overflow)):
        old_nlist = _valid_nlist(ff, R)
    u_old = float(ff.energy_fn(R, old_nlist)["total"])
    u_new = float(trial_ff.energy_fn(R, trial_ff.nlist)["total"])
    delta_u = u_new - u_old
    if not np.isfinite(delta_u):
        probability = 0.0
    else:
        probability = math.exp(min(0.0, -delta_u / kT + log_q_ratio))
    accepted = bool(rng.random() < probability)
    attempt = A2MCAttempt(kind, i, j, float(candidate.distance), accepted,
                          probability, delta_u, log_q_ratio)
    if not accepted:
        return A2MCResult(key, R, velocities, ff, system, (), len(assoc), len(diss), (attempt,))

    event = A2MCEvent(kind, i, j, float(candidate.distance), probability,
                      delta_u, log_q_ratio)
    return A2MCResult(key, R, velocities, trial_ff, trial_system, (event,),
                      len(assoc), len(diss), (attempt,))
