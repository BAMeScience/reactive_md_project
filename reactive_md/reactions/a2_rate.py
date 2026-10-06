# reactive_md/reactions/a2_rate.py

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from reactive_md.reaction import (
    SystemState,
    build_trial_forcefield,
)
from reactive_md.relaxation import (
    langevin_relax_with_nlist,
)
from reactive_md.reactions.a2 import (
    A2Reaction,
)


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
    """Convert an A2 trial dictionary into SystemState.

    The A/A2 benchmark does not use pf6_reacted, so the value
    from the reference state is preserved unchanged.
    """

    return SystemState(
        bonds=(
            jnp.array(
                trial["bonds"][0],
                dtype=int,
            ),
            jnp.array(
                trial["bonds"][1],
            ),
            jnp.array(
                trial["bonds"][2],
            ),
        ),
        angles=(
            jnp.array(
                trial["angles"][0],
                dtype=int,
            ),
            jnp.array(
                trial["angles"][1],
            ),
            jnp.array(
                trial["angles"][2],
            ),
        ),
        torsions=(
            jnp.array(
                trial["torsions"][0],
                dtype=int,
            ),
            jnp.array(
                trial["torsions"][1],
            ),
            jnp.array(
                trial["torsions"][2],
            ),
            jnp.array(
                trial["torsions"][3],
            ),
        ),
        impropers=(
            jnp.array(
                trial["impropers"][0],
                dtype=int,
            ),
            jnp.array(
                trial["impropers"][1],
            ),
            jnp.array(
                trial["impropers"][2],
            ),
            jnp.array(
                trial["impropers"][3],
            ),
        ),
        charges=jnp.array(
            trial["charges"],
        ),
        sigmas=jnp.array(
            trial["sigmas"],
        ),
        epsilons=jnp.array(
            trial["epsilons"],
        ),
        molecule_id=jnp.array(
            trial["molecule_id"],
            dtype=int,
        ),
        pf6_reacted=jnp.array(
            system_ref.pf6_reacted,
        ),
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
    """Perform one complete rate-based A/A2 reactive cycle.

    Reaction:

        A + A <-> A2

    Association candidates are free A-A pairs within the
    geometric cutoff defined by A2Reaction.

    Dissociation candidates are existing A2 bonds.

    Reaction probabilities follow the linear rs@MD rule:

        p_assoc = k_assoc * tau

        p_diss = k_diss * tau

    Multiple non-conflicting reactions may be accepted during
    one reactive cycle.

    If at least one event is accepted:

        1. update topology,
        2. rebuild the force field,
        3. perform the short rs@MD Langevin relaxation.

    Parameters
    ----------
    key
        JAX PRNG key used by Langevin relaxation.

    R
        Current positions.

    velocities
        Current MD velocities.

    box
        Simulation box.

    system
        Current SystemState.

    ff
        Current force-field bundle.

    reaction
        A2Reaction instance.

    disp_fn
        Periodic displacement function.

    shift_fn
        Periodic shift function.

    masses
        Particle masses.

    association_rate_ps
        Association rate in ps^-1.

    dissociation_rate_ps
        Dissociation rate in ps^-1.

    reactive_interval_ps
        Time between reactive cycles in ps.

    temperature_k
        Temperature used during post-reaction relaxation.

    kb_real
        Boltzmann constant in kcal/mol/K.

    rng
        NumPy RNG used for stochastic reaction acceptance.

    relaxation_dt_ps
        Reduced Langevin timestep. Default corresponds to
        0.25 fs.

    relaxation_time_ps
        Duration of the post-reaction relaxation.

    relaxation_tau_ps
        Langevin thermostat time scale.

    Returns
    -------
    A2RateResult
        Updated positions, velocities, force field, topology,
        PRNG key, and accepted reaction information.
    """

    association_rate_ps = float(
        association_rate_ps
    )

    dissociation_rate_ps = float(
        dissociation_rate_ps
    )

    reactive_interval_ps = float(
        reactive_interval_ps
    )

    if association_rate_ps < 0.0:
        raise ValueError(
            "association_rate_ps must be non-negative."
        )

    if dissociation_rate_ps < 0.0:
        raise ValueError(
            "dissociation_rate_ps must be non-negative."
        )

    if reactive_interval_ps <= 0.0:
        raise ValueError(
            "reactive_interval_ps must be positive."
        )

    # ---------------------------------------------------------
    # Linear rs@MD reaction probabilities.
    # ---------------------------------------------------------

    p_assoc = (
        association_rate_ps
        * reactive_interval_ps
    )

    p_diss = (
        dissociation_rate_ps
        * reactive_interval_ps
    )

    if not 0.0 <= p_assoc <= 1.0:
        raise ValueError(
            "Association probability must satisfy "
            "0 <= k_assoc * tau <= 1. "
            f"Got p_assoc={p_assoc}."
        )

    if not 0.0 <= p_diss <= 1.0:
        raise ValueError(
            "Dissociation probability must satisfy "
            "0 <= k_diss * tau <= 1. "
            f"Got p_diss={p_diss}."
        )

    # ---------------------------------------------------------
    # Candidate detection from the topology at the beginning
    # of this reactive cycle.
    # ---------------------------------------------------------

    association_candidates = (
        reaction.find_association_candidates(
            R,
            disp_fn,
            system=system,
        )
    )

    dissociation_candidates = (
        reaction.find_dissociation_candidates(
            R,
            disp_fn,
            system=system,
        )
    )

    # ---------------------------------------------------------
    # Build one common candidate pool.
    # ---------------------------------------------------------

    candidate_pool = []

    for candidate in association_candidates:
        candidate_pool.append(
            (
                "association",
                candidate,
                p_assoc,
            )
        )

    for candidate in dissociation_candidates:
        candidate_pool.append(
            (
                "dissociation",
                candidate,
                p_diss,
            )
        )

    # Randomize candidate ordering so neither reaction direction
    # receives a systematic priority.
    if candidate_pool:
        order = rng.permutation(
            len(candidate_pool)
        )

        candidate_pool = [
            candidate_pool[
                int(index)
            ]
            for index in order
        ]

    # ---------------------------------------------------------
    # Stochastic acceptance.
    #
    # One A atom may participate in at most one accepted event
    # in the same reactive cycle.
    # ---------------------------------------------------------

    used_atoms = set()

    accepted_associations = []
    accepted_dissociations = []

    events = []

    for (
        reaction_type,
        candidate,
        probability,
    ) in candidate_pool:

        i = int(
            candidate.i
        )

        j = int(
            candidate.j
        )

        if (
            i in used_atoms
            or j in used_atoms
        ):
            continue

        u = float(
            rng.random()
        )

        if u >= probability:
            continue

        used_atoms.add(i)
        used_atoms.add(j)

        if reaction_type == "association":
            accepted_associations.append(
                candidate
            )

        elif reaction_type == "dissociation":
            accepted_dissociations.append(
                candidate
            )

        else:
            raise RuntimeError(
                "Unknown A2 reaction type: "
                f"{reaction_type}"
            )

        events.append(
            A2RateEvent(
                reaction_type=reaction_type,
                i=i,
                j=j,
                distance=float(
                    candidate.distance
                ),
                probability=float(
                    probability
                ),
            )
        )

    # ---------------------------------------------------------
    # No event accepted.
    #
    # Return the current state unchanged.
    # ---------------------------------------------------------

    if not events:
        return A2RateResult(
            key=key,
            positions=R,
            velocities=velocities,
            ff=ff,
            system=system,
            events=(),
            n_association_candidates=len(
                association_candidates
            ),
            n_dissociation_candidates=len(
                dissociation_candidates
            ),
        )

    # ---------------------------------------------------------
    # Build final bond topology.
    #
    # Do this in one operation because removing one bond can
    # otherwise shift later bond indices.
    # ---------------------------------------------------------

    bond_idx = np.asarray(
        system.bonds[0],
        dtype=np.int32,
    )

    bond_k = np.asarray(
        system.bonds[1],
    )

    bond_r0 = np.asarray(
        system.bonds[2],
    )

    remove_indices = {
        int(
            candidate.bond_index
        )
        for candidate
        in accepted_dissociations
    }

    keep = np.array(
        [
            index
            not in remove_indices
            for index
            in range(
                bond_idx.shape[0]
            )
        ],
        dtype=bool,
    )

    new_bond_idx = bond_idx[
        keep
    ]

    new_bond_k = bond_k[
        keep
    ]

    new_bond_r0 = bond_r0[
        keep
    ]

    # ---------------------------------------------------------
    # Add newly associated A-A bonds.
    # ---------------------------------------------------------

    if accepted_associations:

        association_indices = np.array(
            [
                [
                    int(
                        candidate.i
                    ),
                    int(
                        candidate.j
                    ),
                ]
                for candidate
                in accepted_associations
            ],
            dtype=np.int32,
        )

        association_k = np.full(
            (
                len(
                    accepted_associations
                ),
            ),
            reaction.k_bond_jax,
            dtype=bond_k.dtype,
        )

        association_r0 = np.full(
            (
                len(
                    accepted_associations
                ),
            ),
            reaction.r0_bond,
            dtype=bond_r0.dtype,
        )

        new_bond_idx = np.concatenate(
            [
                new_bond_idx,
                association_indices,
            ],
            axis=0,
        )

        new_bond_k = np.concatenate(
            [
                new_bond_k,
                association_k,
            ],
            axis=0,
        )

        new_bond_r0 = np.concatenate(
            [
                new_bond_r0,
                association_r0,
            ],
            axis=0,
        )

    # ---------------------------------------------------------
    # Construct trial dictionary.
    # ---------------------------------------------------------

    trial = {
        "bonds": (
            new_bond_idx,
            new_bond_k,
            new_bond_r0,
        ),
        "angles": tuple(
            np.asarray(
                value
            )
            for value
            in system.angles
        ),
        "torsions": tuple(
            np.asarray(
                value
            )
            for value
            in system.torsions
        ),
        "impropers": tuple(
            np.asarray(
                value
            )
            for value
            in system.impropers
        ),
        "charges": np.asarray(
            system.charges
        ).copy(),
        "sigmas": np.asarray(
            system.sigmas
        ).copy(),
        "epsilons": np.asarray(
            system.epsilons
        ).copy(),
        "molecule_id": np.asarray(
            system.molecule_id,
            dtype=np.int32,
        ).copy(),
    }

    # ---------------------------------------------------------
    # Convert topology trial to SystemState.
    # ---------------------------------------------------------

    new_system = (
        make_a2_system_state_from_trial(
            trial,
            system,
        )
    )

    # ---------------------------------------------------------
    # Rebuild force field for the new topology.
    # ---------------------------------------------------------

    ff_trial = (
        build_trial_forcefield(
            R,
            box,
            trial,
            ff,
        )
    )

    # ---------------------------------------------------------
    # Short rs@MD Langevin relaxation.
    #
    # Benchmark defaults:
    #
    #   dt = 0.25 fs
    #   t_relax = 1 ps
    #   tau_T = 0.01 ps
    # ---------------------------------------------------------

    (
        key,
        R_relaxed,
        velocities_relaxed,
        nlist_relaxed,
    ) = langevin_relax_with_nlist(
        key,
        R,
        velocities,
        ff=ff_trial,
        shift_fn=shift_fn,
        masses=masses,
        temperature_k=temperature_k,
        kb_real=kb_real,
        dt_ps=relaxation_dt_ps,
        relaxation_time_ps=(
            relaxation_time_ps
        ),
        thermostat_tau_ps=(
            relaxation_tau_ps
        ),
    )

    if not bool(
        jnp.all(
            jnp.isfinite(
                R_relaxed
            )
        )
    ):
        raise RuntimeError(
            "A2 post-reaction Langevin relaxation "
            "produced non-finite positions."
        )

    if not bool(
        jnp.all(
            jnp.isfinite(
                velocities_relaxed
            )
        )
    ):
        raise RuntimeError(
            "A2 post-reaction Langevin relaxation "
            "produced non-finite velocities."
        )

    ff_trial.nlist = (
        nlist_relaxed
    )

    # ---------------------------------------------------------
    # Return complete accepted state.
    # ---------------------------------------------------------

    return A2RateResult(
        key=key,
        positions=R_relaxed,
        velocities=(
            velocities_relaxed
        ),
        ff=ff_trial,
        system=new_system,
        events=tuple(
            events
        ),
        n_association_candidates=len(
            association_candidates
        ),
        n_dissociation_candidates=len(
            dissociation_candidates
        ),
    )
