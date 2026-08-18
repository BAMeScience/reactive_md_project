# reactive_md/reaction.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from jax_md.minimize import fire_descent

from .reactions import lipf6
from .topology_opls import remove_terms_in_molid
from .forcefield import FFBundle, build_forcefield


# Constants to compute rate constants from activation energies in eV.
K_B_SI = 1.380649e-23
H_SI = 6.62607015e-34
K_B_EV = 8.617333262145e-5


def tst_rate_ps(
    *,
    temperature_k: float,
    activation_energy_eV: float,
    prefactor_ps: float | None = None,
) -> float:
    """Transition-state-theory rate in ps^-1 from an activation energy."""
    if prefactor_ps is None:
        prefactor_ps = (K_B_SI * temperature_k / H_SI) * 1.0e-12

    return float(
        prefactor_ps
        * np.exp(-activation_energy_eV / (K_B_EV * temperature_k))
    )


def resolve_rate_ps(
    *,
    reaction_rate_ps: float | None,
    activation_energy_eV: float | None,
    temperature_k: float,
    prefactor_ps: float | None = None,
) -> float:
    """Resolve either an explicitly supplied rate or a TST-derived rate."""
    if reaction_rate_ps is not None and activation_energy_eV is not None:
        raise ValueError(
            "Specify either reaction_rate_ps or activation_energy_eV, not both."
        )

    if reaction_rate_ps is not None:
        return float(reaction_rate_ps)

    if activation_energy_eV is not None:
        return tst_rate_ps(
            temperature_k=temperature_k,
            activation_energy_eV=activation_energy_eV,
            prefactor_ps=prefactor_ps,
        )

    return 0.0


@dataclass
class SystemState:
    bonds: tuple
    angles: tuple
    torsions: tuple
    impropers: tuple
    charges: Any
    sigmas: Any
    epsilons: Any
    molecule_id: Any
    pf6_reacted: Any

def reaction_probability(
    sigma: float,
    *,
    midpoint: float = 0.0,
    width: float = 0.2,
) -> float:
    """Smooth sigma-dependent kinetic probability/rate factor.

    The value is used as the kinetic accessibility factor p_sigma.

    In hybrid Metropolis mode:
        p_total = p_sigma(sigma) * p_metropolis(Delta E)

    In rate mode:
        k_eff = k_base * p_sigma(sigma)
    """
    if width <= 0.0:
        raise ValueError("Sigma width must be positive.")

    z = (float(sigma) - float(midpoint)) / float(width)
    z = float(np.clip(z, -700.0, 700.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def rate_probability_from_reaction_coordinate(
    *,
    sigma: float,
    base_rate_ps: float,
    reactive_interval_ps: float,
    midpoint: float = 0.0,
    width: float = 0.2,
) -> tuple[float, float, float]:
    """Convert sigma into a rate-mode event probability.

    Returns
    -------
    p_react:
        Probability for the reactive step during one reactive interval.
    k_eff_ps:
        Effective rate in ps^-1.
    sigma_factor:
        Dimensionless factor between 0 and 1.
    """
    sigma_factor = reaction_probability(
        sigma,
        midpoint=midpoint,
        width=width,
    )
    k_eff_ps = float(base_rate_ps) * sigma_factor
    p_react = 1.0 - float(np.exp(-k_eff_ps * float(reactive_interval_ps)))
    return p_react, k_eff_ps, sigma_factor




def candidate_records_from_reaction_candidates(
    candidates: list[lipf6.ReactionCandidate],
    *,
    top_n: int = 10,
) -> list[dict]:
    """Convert sigma-ranked candidates into dictionaries for logging."""
    records = []
    for rank, cand in enumerate(candidates[: int(top_n)]):
        sigma = lipf6.reaction_coordinate(d_pf=cand.d_pf, d_lif=cand.d_lif)
        records.append(
            {
                "rank": int(rank),
                "k_pf6": cand.k_pf6,
                "li_idx": cand.li_idx,
                "leave_F": cand.leave_F,
                "d_lif": cand.d_lif,
                "d_pf": cand.d_pf,
                "sigma": sigma,
            }
        )
    return records


def _candidate_info(cand: lipf6.ReactionCandidate) -> dict:
    sigma = lipf6.reaction_coordinate(d_pf=cand.d_pf, d_lif=cand.d_lif)
    return {
        "k_pf6": cand.k_pf6,
        "li_idx": cand.li_idx,
        "leave_F": cand.leave_F,
        "d_lif": cand.d_lif,
        "d_pf": cand.d_pf,
        "sigma": sigma,
    }



def make_system_state_from_trial(
    trial: dict,
    pf6_reacted_np: np.ndarray,
) -> SystemState:
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
        pf6_reacted=jnp.array(pf6_reacted_np),
    )


def build_trial_forcefield(R, box, trial: dict, ff_ref: FFBundle):
    return build_forcefield(
        R=R,
        box=box,
        bond_idx=trial["bonds"][0],
        k_b=trial["bonds"][1],
        r0=trial["bonds"][2],
        angle_idx=trial["angles"][0],
        k_theta=trial["angles"][1],
        theta0=trial["angles"][2],
        torsions=trial["torsions"],
        impropers=trial["impropers"],
        charges=trial["charges"],
        sigmas=trial["sigmas"],
        epsilons=trial["epsilons"],
        molecule_id=trial["molecule_id"],
        r_cut=float(ff_ref.nb_options.r_cut),
        dr_threshold=float(ff_ref.nb_options.dr_threshold),
    )


def accept_reject(key, *, dE: float, beta: float):
    if not np.isfinite(dE):
        return key, False, 0.0

    exponent = float(np.clip(-beta * float(dE), -700.0, 700.0))
    p_metropolis = min(1.0, float(np.exp(exponent)))

    key, sub = jax.random.split(key)
    u = float(jax.random.uniform(sub))

    return key, (u < p_metropolis), p_metropolis

def maybe_react_one_event(
    key,
    R,
    box,
    *,
    shift_fn,
    ff: FFBundle,
    sys: SystemState,
    reaction: lipf6.LiPF6Reaction,
    beta: float,
    sigma_mid: float = 0.0,
    sigma_width: float = 0.2,
    mc_energy_evaluator=None,
    candidate_log_top_n: int = 10,
):

    pf6_reacted_np = np.array(sys.pf6_reacted, dtype=bool)

    candidates = reaction.find_candidates(
        R,
        ff.disp_fn,
        pf6_reacted=pf6_reacted_np,
    )
    
    candidate_records = candidate_records_from_reaction_candidates(
        candidates,
        top_n=candidate_log_top_n,
    )

    if not candidates:
        return key, False, ff, sys, {
            "mode": "metropolis",
            "reason": "no_candidates",
            "candidate_records": candidate_records,
        }, R

    cand = candidates[0]

    candidate_info = {
        "k_pf6": cand.k_pf6,
        "li_idx": cand.li_idx,
        "leave_F": cand.leave_F,
        "d_lif": cand.d_lif,
        "d_pf": cand.d_pf,
        "sigma": lipf6.reaction_coordinate(d_pf=cand.d_pf, d_lif=cand.d_lif),
    }

    sigma = candidate_info["sigma"]

    # Kinetic sigma gate BEFORE expensive topology/FIRE/energy work.
    p_sigma = reaction_probability(
        sigma,
        midpoint=sigma_mid,
        width=sigma_width,
    )

    key, sub = jax.random.split(key)
    u_sigma = float(jax.random.uniform(sub))

    if u_sigma >= p_sigma:
       return key, False, ff, sys, {
         "mode": "metropolis",
         "candidate": candidate_info,
         "reason": "sigma_gate_rejected",
         "p_sigma": p_sigma,
         "p_metropolis": 0.0,
         "p_total": 0.0,
         "p_acc": 0.0,
         "u_sigma": u_sigma,
         "candidate_records": candidate_records,
       }, R

    if mc_energy_evaluator is None:
        nlist_before = ff.neighbor_fn.update(R, ff.nlist)
        E_before_arr = ff.energy_fn(R, nlist_before)["total"]
        E_before_arr.block_until_ready()
        E_before = float(E_before_arr)
    else:
        E_before = mc_energy_evaluator.energy(R)

    trial, _pf6_molid = reaction.build_trial(sys, cand)

    if trial is None:
        return key, False, ff, sys, {
            "mode": "metropolis",
            "reason": "type_sanity_failed",
            "candidate": candidate_info,
            "p_sigma": p_sigma,
            "p_metropolis": "",
            "p_total": 0.0,
            "p_acc": 0.0,
            "candidate_records": candidate_records,
        }, R

    ff_trial = build_trial_forcefield(R, box, trial, ff)

    R_probe = reaction.prepare_probe(
        R,
        cand,
        sigma_p=float(trial["sigmas"][reaction.phosphorus_index(cand)]),
        sigma_f=float(trial["sigmas"][cand.leave_F]),
        disp_fn=ff_trial.disp_fn,
        shift_fn=shift_fn,
    )

    d_lif_after_probe = float(
          jnp.linalg.norm(
              ff_trial.disp_fn(
              R_probe[cand.li_idx],
              R_probe[cand.leave_F],
           )
          )
    )

    R_relaxed, nlist_relaxed = fire_relax_with_nlist(
        R_probe,
        ff_trial=ff_trial,
        shift_fn=shift_fn,
        n_steps=30,
        dt_start=1.0e-3,
        f_inc=1.01,
        dt_max=1.0e-2,
        n_min=2,
    )

    if not bool(jnp.all(jnp.isfinite(R_relaxed))):
        return key, False, ff, sys, {
            "mode": "metropolis",
            "reason": "nonfinite_relaxed_geometry",
            "candidate": candidate_info,
            "dE": float("nan"),
            "p_sigma": p_sigma,
            "p_metropolis": 0.0,
            "p_total": 0.0,
            "p_acc": 0.0,
            "candidate_records": candidate_records,
        }, R

    if mc_energy_evaluator is None:
        E_after_arr = ff_trial.energy_fn(R_relaxed, nlist_relaxed)["total"]
        E_after_arr.block_until_ready()
        E_after = float(E_after_arr)
    else:
        E_after = mc_energy_evaluator.energy(R_relaxed)

    dE = E_after - E_before

    key, accepted, p_metropolis = accept_reject(
        key,
        dE=dE,
        beta=beta,
    )

    p_total = float(p_sigma * p_metropolis)

    info = {
        "mode": "metropolis",
        "candidate": candidate_info,
        "dE": dE,
        "p_acc": p_total,
        "p_total": p_total,
        "p_sigma": p_sigma,
        "p_metropolis": p_metropolis,
        "sigma_mid": sigma_mid,
        "sigma_width": sigma_width,
        "candidate_records": candidate_records,
    }

    if not accepted:
        return key, False, ff, sys, info, R

    pf6_reacted_np[cand.k_pf6] = True
    sys_new = make_system_state_from_trial(trial, pf6_reacted_np)

    info["accepted_event"] = candidate_info

    return key, True, ff_trial, sys_new, info, R_relaxed


def maybe_react_rate_events(
    key,
    R,
    box,
    *,
    shift_fn,
    ff: FFBundle,
    sys: SystemState,
    reaction: lipf6.LiPF6Reaction,
    reaction_rate_ps: float | None,
    activation_energy_eV: float | None,
    temperature_k: float,
    prefactor_ps: float | None,
    reactive_interval_ps: float,
    max_reactions_per_check: int = 1,
    candidate_log_top_n: int = 10,
    sigma_mid: float = 0.0,
    sigma_width: float = 0.2,
):
    """Attempt rate-based reactions.

    Candidates are first evaluated using the reaction-coordinate
    probability p_sigma. Candidates passing this geometric gate are
    subsequently evaluated using the kinetic probability

        p_rate = 1 - exp(-k * dt)

    where k is either supplied directly or derived from an activation
    energy.

    Accepted, non-conflicting topology changes and probe placements are
    accumulated, followed by one final FIRE relaxation.
    """

    pf6_reacted_np = np.asarray(
        sys.pf6_reacted,
        dtype=bool,
    ).copy()

    # ---------------------------------------------------------
    # Resolve the physical rate constant.
    # ---------------------------------------------------------

    base_rate_ps = resolve_rate_ps(
        reaction_rate_ps=reaction_rate_ps,
        activation_energy_eV=activation_energy_eV,
        temperature_k=temperature_k,
        prefactor_ps=prefactor_ps,
    )

    # ---------------------------------------------------------
    # Kinetic probability.
    #
    # This does not depend on the individual candidate, so calculate
    # it only once per reaction check.
    # ---------------------------------------------------------

    p_rate = float(
        1.0
        - np.exp(
            -base_rate_ps * reactive_interval_ps
        )
    )

    # ---------------------------------------------------------
    # Find all reaction candidates from the same reference geometry.
    # ---------------------------------------------------------

    candidates = reaction.find_candidates(
        R,
        ff.disp_fn,
        pf6_reacted=pf6_reacted_np,
    )

    candidate_records = candidate_records_from_reaction_candidates(
        candidates,
        top_n=candidate_log_top_n,
    )

    if not candidates:
        return key, False, ff, sys, {
            "mode": "rate",
            "sigma_mid": sigma_mid,
            "sigma_width": sigma_width,
            "n_candidates": 0,
            "n_accepted_this_check": 0,
            "p_sigma": 0.0,
            "p_rate": 0.0,
            "p_total": 0.0,
            "k_rate_ps": base_rate_ps,
            "dt_reactive_ps": reactive_interval_ps,
            "candidate_records": candidate_records,
        }, R

    accepted_events = []
    accepted_candidates = []
    accepted_probe_parameters = []

    R_product = R
    ff_current = ff
    sys_current = sys

    used_li: set[int] = set()

    last_p_sigma = 0.0
    last_p_total = 0.0

    # ---------------------------------------------------------
    # Candidate loop.
    # ---------------------------------------------------------

    for cand in candidates:
        if len(accepted_events) >= max_reactions_per_check:
            break

        if pf6_reacted_np[cand.k_pf6]:
            continue

        if cand.li_idx in used_li:
            continue

        # -----------------------------------------------------
        # Reaction-coordinate probability.
        # -----------------------------------------------------

        sigma = lipf6.reaction_coordinate(
            d_pf=cand.d_pf,
            d_lif=cand.d_lif,
        )

        p_sigma = reaction_probability(
            sigma,
            midpoint=sigma_mid,
            width=sigma_width,
        )

        p_total = p_sigma * p_rate

        last_p_sigma = p_sigma
        last_p_total = p_total

        # -----------------------------------------------------
        # Gate 1: reaction-coordinate / geometric gate.
        # -----------------------------------------------------

        key, sub = jax.random.split(key)
        u_sigma = float(jax.random.uniform(sub))

        if u_sigma >= p_sigma:
            continue

        # -----------------------------------------------------
        # Gate 2: kinetic rate gate.
        #
        # Only candidates that passed the sigma gate reach this
        # second random draw.
        # -----------------------------------------------------

        key, sub = jax.random.split(key)
        u_rate = float(jax.random.uniform(sub))

        if u_rate >= p_rate:
            continue

        # -----------------------------------------------------
        # Both gates passed. Only now perform topology-changing
        # work.
        # -----------------------------------------------------

        trial, _pf6_molid = reaction.build_trial(
            sys_current,
            cand,
        )

        if trial is None:
            continue

        ff_trial = build_trial_forcefield(
            R,
            box,
            trial,
            ff_current,
        )

        pf6_reacted_np[cand.k_pf6] = True
        used_li.add(cand.li_idx)

        sys_current = make_system_state_from_trial(
            trial,
            pf6_reacted_np,
        )

        ff_current = ff_trial

        # -----------------------------------------------------
        # Event logging.
        # -----------------------------------------------------

        event_info = _candidate_info(cand)

        event_info.update(
            {
                "p_sigma": p_sigma,
                "p_rate": p_rate,
                "p_total": p_total,
                "u_sigma": u_sigma,
                "u_rate": u_rate,
                "k_rate_ps": base_rate_ps,
                "dt_reactive_ps": reactive_interval_ps,
                "sigma_mid": sigma_mid,
                "sigma_width": sigma_width,
            }
        )

        accepted_events.append(event_info)
        accepted_candidates.append(cand)

        accepted_probe_parameters.append(
            (
                float(
                    trial["sigmas"][
                        reaction.phosphorus_index(cand)
                    ]
                ),
                float(
                    trial["sigmas"][cand.leave_F]
                ),
            )
        )

    # ---------------------------------------------------------
    # No reaction accepted.
    # ---------------------------------------------------------

    if not accepted_events:
        return key, False, ff, sys, {
            "mode": "rate",
            "sigma_mid": sigma_mid,
            "sigma_width": sigma_width,
            "n_candidates": len(candidates),
            "n_accepted_this_check": 0,
            "p_sigma": last_p_sigma,
            "p_rate": p_rate,
            "p_total": last_p_total,
            "k_rate_ps": base_rate_ps,
            "dt_reactive_ps": reactive_interval_ps,
            "candidate_records": candidate_records,
        }, R

    # ---------------------------------------------------------
    # Apply the probe placements for all accepted candidates.
    # ---------------------------------------------------------

    R_product = R

    for cand, probe_parameters in zip(
        accepted_candidates,
        accepted_probe_parameters,
    ):
        sigma_p, sigma_f = probe_parameters

        R_probe_single = reaction.prepare_probe(
            R,
            cand,
            sigma_p=sigma_p,
            sigma_f=sigma_f,
            disp_fn=ff_current.disp_fn,
            shift_fn=shift_fn,
        )

        R_product = R_product.at[cand.leave_F].set(
            R_probe_single[cand.leave_F]
        )

    # ---------------------------------------------------------
    # Single relaxation after all accepted topology changes.
    # ---------------------------------------------------------

    R_relaxed, nlist_relaxed = fire_relax_with_nlist(
        R_product,
        ff_trial=ff_current,
        shift_fn=shift_fn,
        n_steps=30,
        dt_start=1.0e-3,
        f_inc=1.01,
        dt_max=1.0e-2,
        n_min=2,
    )

    if not bool(jnp.all(jnp.isfinite(R_relaxed))):
        return key, False, ff, sys, {
            "mode": "rate",
            "reason": "nonfinite_relaxed_geometry",
            "sigma_mid": sigma_mid,
            "sigma_width": sigma_width,
            "n_candidates": len(candidates),
            "n_accepted_this_check": 0,
            "p_rate": p_rate,
            "k_rate_ps": base_rate_ps,
            "dt_reactive_ps": reactive_interval_ps,
            "candidate_records": candidate_records,
        }, R

    ff_current.nlist = nlist_relaxed

    first_event = accepted_events[0]

    return key, True, ff_current, sys_current, {
        "mode": "rate",
        "sigma_mid": sigma_mid,
        "sigma_width": sigma_width,
        "accepted_events": accepted_events,
        "accepted_event": first_event,
        "n_candidates": len(candidates),
        "n_accepted_this_check": len(accepted_events),
        "p_sigma": first_event["p_sigma"],
        "p_rate": first_event["p_rate"],
        "p_total": first_event["p_total"],
        "k_rate_ps": base_rate_ps,
        "activation_energy_eV": activation_energy_eV,
        "temperature_k": temperature_k,
        "prefactor_ps": prefactor_ps,
        "dt_reactive_ps": reactive_interval_ps,
        "candidate_records": candidate_records,
    }, R_relaxed


def fire_relax_with_nlist(
    R0: jnp.ndarray,
    *,
    ff_trial,
    shift_fn,
    n_steps: int = 30,
    dt_start: float = 1.0e-3,
    f_inc: float = 1.01,
    dt_max: float = 1.0e-2,
    n_min: int = 2,
):
    def energy_scalar(R, *, nlist):
        return ff_trial.energy_fn(R, nlist)["total"]

    fire_init, fire_apply = fire_descent(
        energy_scalar,
        shift_fn,
        dt_start=dt_start,
        f_inc=f_inc,
        dt_max=dt_max,
        n_min=n_min,
    )

    fire_apply = jax.jit(fire_apply)
    update_nlist = jax.jit(ff_trial.neighbor_fn.update)

    nlist = ff_trial.neighbor_fn.allocate(R0)
    fire_state = fire_init(R0, nlist=nlist)

    @jax.jit
    def step_fire_fn(i, carry):
        st, nl = carry
        nl = update_nlist(st.position, nl)
        st = fire_apply(st, nlist=nl)
        return st, nl

    fire_state, nlist = jax.lax.fori_loop(
        0,
        int(n_steps),
        step_fire_fn,
        (fire_state, nlist),
    )

    fire_state.position.block_until_ready()
    return fire_state.position, nlist

