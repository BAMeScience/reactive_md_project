"""Rate-based A + A <-> A2 reactive cycles."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from reactive_md.reaction import SystemState, build_trial_forcefield
from reactive_md.relaxation import langevin_relax_with_nlist
from reactive_md.reactions.a2 import A2Reaction


def translate_reacting_pair(
    R,
    i,
    j,
    *,
    disp_fn,
    shift_fn,
    displacement_per_atom=0.0,
    move_apart=True,
):
    """Move each atom along the pair axis by the specified distance (angstrom).

    With ``move_apart=True``, increase separation (dissociation).
    With ``move_apart=False``, decrease separation (association).
    A zero displacement leaves positions unchanged.
    """
    delta = float(displacement_per_atom)
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("displacement_per_atom must be finite and nonnegative.")
    if delta == 0.0:
        return R

    d = disp_fn(R[i], R[j])
    r_before = float(jnp.linalg.norm(d))
    if r_before < 1.0e-8:
        raise ValueError("Cannot translate coincident reacting atoms.")

    u = d / r_before

    # Try both signs: displacement functions may use either orientation.
    def shifted(sign):
        return (
            R.at[i].set(shift_fn(R[i], -sign * delta * u))
            .at[j].set(shift_fn(R[j], +sign * delta * u))
        )

    R_plus = shifted(+1.0)
    R_minus = shifted(-1.0)
    r_plus = float(jnp.linalg.norm(disp_fn(R_plus[i], R_plus[j])))
    r_minus = float(jnp.linalg.norm(disp_fn(R_minus[i], R_minus[j])))

    if move_apart:
        R_new = R_plus if r_plus >= r_minus else R_minus
        r_after = max(r_plus, r_minus)
        if r_after <= r_before:
            raise ValueError("Pair translation did not increase separation.")
    else:
        R_new = R_plus if r_plus <= r_minus else R_minus
        r_after = min(r_plus, r_minus)
        if r_after >= r_before:
            raise ValueError("Pair translation did not decrease separation.")

    return R_new


@dataclass(frozen=True)
class A2RateEvent:
    """Accepted A/A2 reaction event."""

    reaction_type: str
    i: int
    j: int
    distance: float
    probability: float


@dataclass(frozen=True)
class A2RateResult:
    """Result of one rate-based A/A2 reactive cycle."""

    key: object
    positions: object
    velocities: object
    ff: object
    system: SystemState
    events: tuple[A2RateEvent, ...]
    n_association_candidates: int
    n_dissociation_candidates: int


def make_a2_system_state_from_trial(
    trial: dict,
    system_ref: SystemState,
) -> SystemState:
    """Convert an A2 trial dictionary to SystemState.

    The A/A2 benchmark does not use pf6_reacted, so it is preserved.
    """
    return SystemState(
        bonds=(
            jnp.array(trial["bonds"][0], dtype=int),
            jnp.array(trial["bonds"][1]),
            jnp.array(trial["bonds"][2]),
        ),
        angles=(
            jnp.array(trial["angles"][0], dtype=int),
            jnp.array(trial["angles"][1]),
            jnp.array(trial["angles"][2]),
        ),
        torsions=(
            jnp.array(trial["torsions"][0], dtype=int),
            jnp.array(trial["torsions"][1]),
            jnp.array(trial["torsions"][2]),
            jnp.array(trial["torsions"][3]),
        ),
        impropers=(
            jnp.array(trial["impropers"][0], dtype=int),
            jnp.array(trial["impropers"][1]),
            jnp.array(trial["impropers"][2]),
            jnp.array(trial["impropers"][3]),
        ),
        charges=jnp.array(trial["charges"]),
        sigmas=jnp.array(trial["sigmas"]),
        epsilons=jnp.array(trial["epsilons"]),
        molecule_id=jnp.array(trial["molecule_id"], dtype=int),
        pf6_reacted=jnp.array(system_ref.pf6_reacted),
    )


def a2_rate_reactive_cycle(
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
    association_rate_ps: float,
    dissociation_rate_ps: float,
    reactive_interval_ps: float,
    temperature_k: float,
    kb_real: float,
    rng: np.random.Generator,
    relaxation_dt_ps: float = 0.00025,
    relaxation_time_ps: float = 1.0,
    relaxation_tau_ps: float = 0.01,
) -> A2RateResult:
    """Perform one rate-based A + A <-> A2 reactive cycle.

    Candidate reactions are accepted with linear probabilities
    ``rate_ps * reactive_interval_ps``. Each atom may participate in at
    most one event per cycle. Accepted events update topology, translate
    reacting atoms, rebuild the force field, and trigger Langevin relaxation.
    """
    association_rate_ps = float(association_rate_ps)
    dissociation_rate_ps = float(dissociation_rate_ps)
    reactive_interval_ps = float(reactive_interval_ps)

    if association_rate_ps < 0.0:
        raise ValueError("association_rate_ps must be non-negative.")
    if dissociation_rate_ps < 0.0:
        raise ValueError("dissociation_rate_ps must be non-negative.")
    if reactive_interval_ps <= 0.0:
        raise ValueError("reactive_interval_ps must be positive.")

    p_assoc = association_rate_ps * reactive_interval_ps
    p_diss = dissociation_rate_ps * reactive_interval_ps
    if not 0.0 <= p_assoc <= 1.0:
        raise ValueError(
            "Association probability must satisfy 0 <= k_assoc * tau <= 1. "
            f"Got p_assoc={p_assoc}."
        )
    if not 0.0 <= p_diss <= 1.0:
        raise ValueError(
            "Dissociation probability must satisfy 0 <= k_diss * tau <= 1. "
            f"Got p_diss={p_diss}."
        )

    association_candidates = reaction.find_association_candidates(
        R, disp_fn, system=system
    )
    dissociation_candidates = reaction.find_dissociation_candidates(
        R, disp_fn, system=system
    )

    candidate_pool = (
        [("association", c, p_assoc) for c in association_candidates]
        + [("dissociation", c, p_diss) for c in dissociation_candidates]
    )
    if candidate_pool:
        order = rng.permutation(len(candidate_pool))
        candidate_pool = [candidate_pool[int(index)] for index in order]

    used_atoms = set()
    accepted_associations = []
    accepted_dissociations = []
    events = []

    for reaction_type, candidate, probability in candidate_pool:
        i = int(candidate.i)
        j = int(candidate.j)
        if i in used_atoms or j in used_atoms:
            continue
        if float(rng.random()) >= probability:
            continue

        used_atoms.add(i)
        used_atoms.add(j)
        if reaction_type == "association":
            accepted_associations.append(candidate)
        elif reaction_type == "dissociation":
            accepted_dissociations.append(candidate)
        else:
            raise RuntimeError(f"Unknown A2 reaction type: {reaction_type}")

        events.append(
            A2RateEvent(
                reaction_type=reaction_type,
                i=i,
                j=j,
                distance=float(candidate.distance),
                probability=float(probability),
            )
        )

    if not events:
        return A2RateResult(
            key=key,
            positions=R,
            velocities=velocities,
            ff=ff,
            system=system,
            events=(),
            n_association_candidates=len(association_candidates),
            n_dissociation_candidates=len(dissociation_candidates),
        )

    # Remove all accepted dissociation bonds in one operation, since
    # removing individual bonds would shift subsequent bond indices.
    bond_idx = np.asarray(system.bonds[0], dtype=np.int32)
    bond_k = np.asarray(system.bonds[1])
    bond_r0 = np.asarray(system.bonds[2])
    remove_indices = {int(c.bond_index) for c in accepted_dissociations}
    keep = np.array(
        [index not in remove_indices for index in range(bond_idx.shape[0])],
        dtype=bool,
    )
    new_bond_idx = bond_idx[keep]
    new_bond_k = bond_k[keep]
    new_bond_r0 = bond_r0[keep]

    if accepted_associations:
        association_indices = np.array(
            [[int(c.i), int(c.j)] for c in accepted_associations],
            dtype=np.int32,
        )
        association_k = np.full(
            (len(accepted_associations),),
            reaction.k_bond_jax,
            dtype=bond_k.dtype,
        )
        association_r0 = np.full(
            (len(accepted_associations),),
            reaction.r0_bond,
            dtype=bond_r0.dtype,
        )
        new_bond_idx = np.concatenate(
            [new_bond_idx, association_indices], axis=0
        )
        new_bond_k = np.concatenate([new_bond_k, association_k], axis=0)
        new_bond_r0 = np.concatenate([new_bond_r0, association_r0], axis=0)

    # In the A/A2 benchmark each unbonded atom is a separate molecule;
    # every A2 bond joins precisely two atoms.
    new_molecule_id = np.arange(len(system.molecule_id), dtype=np.int32)
    for i, j in new_bond_idx:
        new_molecule_id[int(j)] = new_molecule_id[int(i)]

    trial = {
        "bonds": (new_bond_idx, new_bond_k, new_bond_r0),
        "angles": tuple(np.asarray(value) for value in system.angles),
        "torsions": tuple(np.asarray(value) for value in system.torsions),
        "impropers": tuple(np.asarray(value) for value in system.impropers),
        "charges": np.asarray(system.charges).copy(),
        "sigmas": np.asarray(system.sigmas).copy(),
        "epsilons": np.asarray(system.epsilons).copy(),
        "molecule_id": new_molecule_id.copy(),
    }
    trial["energy_offset"] = (
        len(trial["bonds"][0]) * reaction.bond_energy_offset
    )
    new_system = make_a2_system_state_from_trial(trial, system)

    # Apply displacements before rebuilding the force field and relaxing.
    R_reaction = R
    for candidate in accepted_associations:
        R_reaction = translate_reacting_pair(
            R_reaction,
            int(candidate.i),
            int(candidate.j),
            disp_fn=disp_fn,
            shift_fn=shift_fn,
            displacement_per_atom=reaction.association_displacement,
            move_apart=False,
        )

    for candidate in accepted_dissociations:
        R_reaction = translate_reacting_pair(
            R_reaction,
            int(candidate.i),
            int(candidate.j),
            disp_fn=disp_fn,
            shift_fn=shift_fn,
            displacement_per_atom=reaction.dissociation_displacement,
            move_apart=True,
        )

    ff_trial = build_trial_forcefield(R_reaction, box, trial, ff)

    key, R_relaxed, velocities_relaxed, nlist_relaxed = (
        langevin_relax_with_nlist(
            key,
            R_reaction,
            velocities,
            ff=ff_trial,
            shift_fn=shift_fn,
            masses=masses,
            temperature_k=temperature_k,
            kb_real=kb_real,
            dt_ps=relaxation_dt_ps,
            relaxation_time_ps=relaxation_time_ps,
            thermostat_tau_ps=relaxation_tau_ps,
        )
    )

    if not bool(jnp.all(jnp.isfinite(R_relaxed))):
        raise RuntimeError(
            "A2 post-reaction Langevin relaxation produced non-finite positions."
        )
    if not bool(jnp.all(jnp.isfinite(velocities_relaxed))):
        raise RuntimeError(
            "A2 post-reaction Langevin relaxation produced non-finite velocities."
        )

    ff_trial.nlist = nlist_relaxed
    return A2RateResult(
        key=key,
        positions=R_relaxed,
        velocities=velocities_relaxed,
        ff=ff_trial,
        system=new_system,
        events=tuple(events),
        n_association_candidates=len(association_candidates),
        n_dissociation_candidates=len(dissociation_candidates),
    )
