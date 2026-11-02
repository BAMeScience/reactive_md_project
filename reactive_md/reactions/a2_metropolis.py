"""A2 implementation of rs@md[MC], Fig. 3a and Eq. (10) of Biedermann et al.

One candidate is selected uniformly from the combined eligible candidate list.
The topology is changed tentatively, the reacting atoms are translated, and a
Langevin MD relaxation is performed *before* Metropolis acceptance. Rejection
restores the original physical state; only the consumed RNG state is retained.

Energy convention: the default is the original program's local-potential
plus solvation-interaction mode. For an A2-only system this is evaluated as
U_full - U_spectators, separately before and after trial relaxation. The
reaction-energy correction V0 is carried in the topology-dependent full
force field, and enters the difference exactly once. The original code also
supports an optional full-system potential mode, exposed here as ``full``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp
import numpy as np

from reactive_md.reaction import SystemState, build_trial_forcefield
from reactive_md.relaxation import langevin_relax_with_nlist
from reactive_md.reactions.a2 import A2Reaction
from reactive_md.reactions.a2_energy import local_solvation_energy
from reactive_md.reactions.a2_rate import (
    make_a2_system_state_from_trial,
    translate_reacting_pair,
)


@dataclass(frozen=True)
class A2MCAttempt:
    reaction_type: str
    i: int
    j: int
    distance: float
    accepted: bool
    probability: float
    delta_energy: float
    reason: str = ""


@dataclass(frozen=True)
class A2MCEvent:
    reaction_type: str
    i: int
    j: int
    distance: float
    probability: float
    delta_energy: float


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
    """Rebuild connectivity and set the total PEL I dimer energy offset."""
    trial = dict(trial)
    parents = np.arange(n_atoms, dtype=np.int32)

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = int(parents[i])
        return i

    a_set = set(map(int, reaction.a_indices))
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
    for _ in range(max_retries + 1):
        nl = ff.neighbor_fn.allocate(positions, extra_capacity=extra_capacity)
        if not bool(np.asarray(nl.did_buffer_overflow)):
            return nl
        extra_capacity = max(8, 2 * extra_capacity)
    raise RuntimeError("A2 MC neighbor-list allocation overflowed")


def _potential(ff, positions, nlist):
    return float(np.asarray(ff.energy_fn(positions, nlist)["total"]))


def _metropolis_probability(delta_energy: float, kT: float) -> float:
    """The paper's Eq. (10), without a proposal-ratio multiplier."""
    if not np.isfinite(delta_energy):
        return 0.0
    return math.exp(min(0.0, -delta_energy / kT))


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
    shift_fn,
    masses,
    temperature_k: float,
    kb_real: float,
    rng: np.random.Generator,
    relaxation_dt_ps: float = 0.00025,
    relaxation_time_ps: float = 1.0,
    relaxation_tau_ps: float = 0.01,
    energy_mode: str = "local_solvation",
) -> A2MCResult:
    """Run one rs@md[MC] trial; commit only on acceptance.

    The original MD state and force-field objects are never mutated by a
    rejected trial. The returned PRNG key advances even if rejected because
    trial Langevin dynamics have consumed random numbers.
    """
    if energy_mode not in ("local_solvation", "full"):
        raise ValueError("energy_mode must be 'local_solvation' or 'full'")
    kT = float(kb_real) * float(temperature_k)
    if not np.isfinite(kT) or kT <= 0:
        raise ValueError("kb_real * temperature_k must be finite and positive")
    if not np.isfinite(relaxation_time_ps) or relaxation_time_ps <= 0:
        raise ValueError("MC trial relaxation_time_ps must be positive")

    assoc = reaction.find_association_candidates(R, disp_fn, system=system)
    diss = reaction.find_dissociation_candidates(R, disp_fn, system=system)
    candidates = [("association", c) for c in assoc] + [("dissociation", c) for c in diss]
    if not candidates:
        return A2MCResult(key, R, velocities, ff, system, (), len(assoc), len(diss), ())

    kind, candidate = candidates[int(rng.integers(len(candidates)))]
    i, j = int(candidate.i), int(candidate.j)
    trial = (reaction.build_association_trial(system, candidate)
             if kind == "association" else reaction.build_dissociation_trial(system, candidate))
    trial = _finalize_trial(trial, reaction, len(system.molecule_id))
    trial_system = make_a2_system_state_from_trial(trial, system)

    # Read the unmodified reactant energy *before* constructing/relaxing the
    # product. No pre-reaction state object or neighbor list is overwritten.
    pair = (i, j)
    u_before = (_potential(ff, R, _valid_nlist(ff, R))
                if energy_mode == "full" else
                local_solvation_energy(ff, R, system, pair, box))
    R_trial = translate_reacting_pair(
        R, i, j, disp_fn=disp_fn, shift_fn=shift_fn,
        displacement_per_atom=(reaction.association_displacement if kind == "association"
                               else reaction.dissociation_displacement),
        move_apart=(kind == "dissociation"),
    )
    trial_ff = build_trial_forcefield(R_trial, box, trial, ff)
    key_after, R_relaxed, v_relaxed, nl_relaxed = langevin_relax_with_nlist(
        key, R_trial, velocities, ff=trial_ff, shift_fn=shift_fn,
        masses=masses, temperature_k=temperature_k, kb_real=kb_real,
        dt_ps=relaxation_dt_ps, relaxation_time_ps=relaxation_time_ps,
        thermostat_tau_ps=relaxation_tau_ps,
    )
    if not bool(np.asarray(jnp.all(jnp.isfinite(R_relaxed)))):
        raise FloatingPointError("MC trial relaxation produced nonfinite positions")
    if not bool(np.asarray(jnp.all(jnp.isfinite(v_relaxed)))):
        raise FloatingPointError("MC trial relaxation produced nonfinite velocities")
    if bool(np.asarray(nl_relaxed.did_buffer_overflow)):
        raise RuntimeError("MC trial relaxation returned an overflowing neighbor list")

    u_after = (_potential(trial_ff, R_relaxed, nl_relaxed)
               if energy_mode == "full" else
               local_solvation_energy(trial_ff, R_relaxed, trial_system, pair, box))
    delta_u = u_after - u_before
    probability = _metropolis_probability(delta_u, kT)
    accepted = bool(rng.random() < probability)
    attempt = A2MCAttempt(kind, i, j, float(candidate.distance), accepted,
                          probability, delta_u,
                          "nonfinite trial energy" if not np.isfinite(delta_u) else "")
    if not accepted:
        return A2MCResult(key_after, R, velocities, ff, system, (),
                          len(assoc), len(diss), (attempt,))

    trial_ff.nlist = nl_relaxed
    event = A2MCEvent(kind, i, j, float(candidate.distance), probability, delta_u)
    return A2MCResult(key_after, R_relaxed, v_relaxed, trial_ff, trial_system,
                      (event,), len(assoc), len(diss), (attempt,))
