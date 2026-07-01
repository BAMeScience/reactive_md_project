# reactive_md/reaction.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from jax_md.minimize import fire_descent

from .reactions.templates_pf5 import PF5Template, LiFTemplate
from .reactions.lipf6 import embed_pf5_into_pf6
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


@dataclass(frozen=True)
class ReactionCandidate:
    """One possible PF6 -> PF5 + Li/F reaction event."""

    k_pf6: int
    li_idx: int
    leave_F: int
    d_lif: float
    d_pf: float


def _distance(disp_fn, Rj, i: int, j: int) -> float:
    dr = np.asarray(disp_fn(Rj[int(i)], Rj[int(j)]))
    return float(np.linalg.norm(dr))


def reaction_coordinate(*, d_pf: float, d_lif: float) -> float:

    """
    Reaction coordinate from Fattebert et al. Journal of The Electrochemical Society, 2024 171 080505 for LiPF6 decomposition.

    σ = d(P–F) - d(Li–F)

    σ < 0 : reactant-like configuration
    σ ≈ 0 : transition region
    σ > 0 : product-like configuration
    """
    return float(d_pf - d_lif)


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


def find_reaction_candidates(
    R,
    pf6_atoms_np: np.ndarray,
    li_atoms_np: np.ndarray,
    disp_fn,
    *,
    pf6_reacted_np: np.ndarray,
) -> list[ReactionCandidate]:
    """Return all possible reaction candidates, ranked by sigma.

    Every unreacted PF6 fluorine and every Li ion is considered. No independent
    Li-F or P-F hard cutoff is used. The only ordering variable is

        sigma = d(P-F) - d(Li-F)
    """
    Rj = jnp.asarray(R)
    candidates: list[ReactionCandidate] = []

    for k in range(pf6_atoms_np.shape[0]):
        if pf6_reacted_np[k]:
            continue

        P_atom = int(pf6_atoms_np[k, 0])
        Fs = pf6_atoms_np[k, 1:]

        for li in li_atoms_np:
            for f in Fs:
                li_idx = int(li)
                f_idx = int(f)

                d_lif = _distance(disp_fn, Rj, li_idx, f_idx)
                d_pf = _distance(disp_fn, Rj, P_atom, f_idx)

                candidates.append(
                    ReactionCandidate(
                        k_pf6=int(k),
                        li_idx=li_idx,
                        leave_F=f_idx,
                        d_lif=float(d_lif),
                        d_pf=float(d_pf),
                    )
                )

    candidates.sort(
        key=lambda c: reaction_coordinate(d_pf=c.d_pf, d_lif=c.d_lif),
        reverse=True,
    )
    return candidates


def candidate_records_from_reaction_candidates(
    candidates: list[ReactionCandidate],
    *,
    top_n: int = 10,
) -> list[dict]:
    """Convert sigma-ranked candidates into dictionaries for logging."""
    records = []
    for rank, cand in enumerate(candidates[: int(top_n)]):
        sigma = reaction_coordinate(d_pf=cand.d_pf, d_lif=cand.d_lif)
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


def _candidate_info(cand: ReactionCandidate) -> dict:
    sigma = reaction_coordinate(d_pf=cand.d_pf, d_lif=cand.d_lif)
    return {
        "k_pf6": cand.k_pf6,
        "li_idx": cand.li_idx,
        "leave_F": cand.leave_F,
        "d_lif": cand.d_lif,
        "d_pf": cand.d_pf,
        "sigma": sigma,
    }


def make_probe_geometry(
    R,
    *,
    P_atom: int,
    leave_F: int,
    disp_fn,
    shift_fn,
    r_pf_probe: float = 4.0,
):
    """Move the leaving F away from P before product-side relaxation.

    This is a geometry preparation step, not a reaction criterion.
    """
    rP = R[P_atom]
    rF = R[leave_F]

    PF_vec = disp_fn(rP, rF)
    PF_dist = jnp.linalg.norm(PF_vec) + 1.0e-12
    uPF = PF_vec / PF_dist

    rF_new = rP + r_pf_probe * uPF
    drF = rF_new - rF

    return R.at[leave_F].set(shift_fn(rF, drF))


def propose_reaction_trial(
    sys: SystemState,
    cand: ReactionCandidate,
    *,
    pf6_atoms_np: np.ndarray,
    atom_types_np: np.ndarray,
    pf5: PF5Template,
    lif: LiFTemplate,
    p_type: int,
    f_type: int,
    li_type: int,
):
    k_pf6 = cand.k_pf6
    li_idx = cand.li_idx
    leave_F = cand.leave_F

    P_atom = int(pf6_atoms_np[k_pf6, 0])

    if (
        atom_types_np[P_atom] != p_type
        or atom_types_np[leave_F] != f_type
        or atom_types_np[li_idx] != li_type
    ):
        return None, None

    bond_idx, k_b, r0 = (
        np.array(sys.bonds[0], dtype=np.int32),
        np.array(sys.bonds[1], dtype=np.float32),
        np.array(sys.bonds[2], dtype=np.float32),
    )

    angle_idx, k_theta, theta0 = (
        np.array(sys.angles[0], dtype=np.int32),
        np.array(sys.angles[1], dtype=np.float32),
        np.array(sys.angles[2], dtype=np.float32),
    )

    tors_idx, tors_k, tors_n, tors_gamma = (
        np.array(sys.torsions[0], dtype=np.int32),
        np.array(sys.torsions[1], dtype=np.float32),
        np.array(sys.torsions[2], dtype=np.int32),
        np.array(sys.torsions[3], dtype=np.float32),
    )

    impr_idx, impr_k, impr_n, impr_gamma = (
        np.array(sys.impropers[0], dtype=np.int32),
        np.array(sys.impropers[1], dtype=np.float32),
        np.array(sys.impropers[2], dtype=np.int32),
        np.array(sys.impropers[3], dtype=np.float32),
    )

    charges_np = np.array(sys.charges, dtype=np.float32)
    sigmas_np = np.array(sys.sigmas, dtype=np.float32)
    eps_np = np.array(sys.epsilons, dtype=np.float32)
    molecule_id_np = np.array(sys.molecule_id, dtype=np.int32)

    pf6_molid = int(molecule_id_np[P_atom])

    bond_idx2, (k_b2, r0_2) = remove_terms_in_molid(
        bond_idx,
        [k_b, r0],
        molecule_id_np,
        pf6_molid,
    )

    angle_idx2, (k_th2, th0_2) = remove_terms_in_molid(
        angle_idx,
        [k_theta, theta0],
        molecule_id_np,
        pf6_molid,
    )

    tors_idx2, (tors_k2, tors_n2, tors_g2) = remove_terms_in_molid(
        tors_idx,
        [tors_k, tors_n, tors_gamma],
        molecule_id_np,
        pf6_molid,
    )

    impr_idx2, (impr_k2, impr_n2, impr_g2) = remove_terms_in_molid(
        impr_idx,
        [impr_k, impr_n, impr_gamma],
        molecule_id_np,
        pf6_molid,
    )

    pf5_glob, pf5_bonds_g, pf5_angles_g = embed_pf5_into_pf6(
        pf6_atoms_np[k_pf6],
        leave_F,
        pf5_bond_idx_local=pf5.bond_idx_local,
        pf5_angle_idx_local=pf5.angle_idx_local,
    )

    bond_idx2 = np.concatenate([bond_idx2, pf5_bonds_g], axis=0)
    k_b2 = np.concatenate([k_b2, pf5.k_b], axis=0)
    r0_2 = np.concatenate([r0_2, pf5.r0], axis=0)

    angle_idx2 = np.concatenate([angle_idx2, pf5_angles_g], axis=0)
    k_th2 = np.concatenate([k_th2, pf5.k_theta], axis=0)
    th0_2 = np.concatenate([th0_2, pf5.theta0], axis=0)

    P_pf5 = int(pf5_glob[0])
    Fs_pf5 = pf5_glob[1:]

    charges_np[P_pf5] = pf5.q["P"]
    charges_np[Fs_pf5] = pf5.q["F"]

    eps_np[P_pf5], sigmas_np[P_pf5] = pf5.pair["P"]
    eps_np[Fs_pf5], sigmas_np[Fs_pf5] = pf5.pair["F"]

    charges_np[leave_F] = lif.nb["F"]["q"]
    sigmas_np[leave_F] = lif.nb["F"]["sigma"]
    eps_np[leave_F] = lif.nb["F"]["eps"]

    charges_np[li_idx] = lif.nb["Li"]["q"]
    sigmas_np[li_idx] = lif.nb["Li"]["sigma"]
    eps_np[li_idx] = lif.nb["Li"]["eps"]

    molecule_id2 = molecule_id_np.copy()
    new_molid = int(molecule_id2.max()) + 1
    molecule_id2[leave_F] = new_molid

    return {
        "bonds": (bond_idx2, k_b2, r0_2),
        "angles": (angle_idx2, k_th2, th0_2),
        "torsions": (tors_idx2, tors_k2, tors_n2, tors_g2),
        "impropers": (impr_idx2, impr_k2, impr_n2, impr_g2),
        "charges": charges_np,
        "sigmas": sigmas_np,
        "epsilons": eps_np,
        "molecule_id": molecule_id2,
    }, pf6_molid


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
    pf6_atoms,
    li_atoms,
    atom_types,
    pf5: PF5Template,
    lif: LiFTemplate,
    p_type: int,
    f_type: int,
    li_type: int,
    r_pf_probe: float,
    beta: float,
    sigma_mid: float = 0.0,
    sigma_width: float = 0.2,
    mc_energy_evaluator=None,
    candidate_log_top_n: int = 10,
):
    pf6_atoms_np = np.array(pf6_atoms, dtype=np.int32)
    li_atoms_np = np.array(li_atoms, dtype=np.int32)
    pf6_reacted_np = np.array(sys.pf6_reacted, dtype=bool)
    atom_types_np = np.array(atom_types)

    candidates = find_reaction_candidates(
        R,
        pf6_atoms_np,
        li_atoms_np,
        ff.disp_fn,
        pf6_reacted_np=pf6_reacted_np,
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
        "sigma": reaction_coordinate(d_pf=cand.d_pf, d_lif=cand.d_lif),
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
            "reason": "sigma_gate_rejected",
            "candidate": candidate_info,
            "p_sigma": p_sigma,
            "p_metropolis": "",
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

    trial, _pf6_molid = propose_reaction_trial(
        sys,
        cand,
        pf6_atoms_np=pf6_atoms_np,
        atom_types_np=atom_types_np,
        pf5=pf5,
        lif=lif,
        p_type=p_type,
        f_type=f_type,
        li_type=li_type,
    )

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

    P_atom = int(pf6_atoms_np[cand.k_pf6, 0])
    R_probe = make_probe_geometry(
        R,
        P_atom=P_atom,
        leave_F=cand.leave_F,
        disp_fn=ff.disp_fn,
        shift_fn=shift_fn,
        r_pf_probe=r_pf_probe,
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
    pf6_atoms,
    li_atoms,
    atom_types,
    pf5: PF5Template,
    lif: LiFTemplate,
    p_type: int,
    f_type: int,
    li_type: int,
    r_pf_probe: float,
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

    In rate mode, sigma contributes to the effective rate via
    reaction_probability().
    """
    pf6_atoms_np = np.array(pf6_atoms, dtype=np.int32)
    li_atoms_np = np.array(li_atoms, dtype=np.int32)
    pf6_reacted_np = np.array(sys.pf6_reacted, dtype=bool)
    atom_types_np = np.array(atom_types)

    base_rate_ps = resolve_rate_ps(
        reaction_rate_ps=reaction_rate_ps,
        activation_energy_eV=activation_energy_eV,
        temperature_k=temperature_k,
        prefactor_ps=prefactor_ps,
    )

    candidates = find_reaction_candidates(
        R,
        pf6_atoms_np,
        li_atoms_np,
        ff.disp_fn,
        pf6_reacted_np=pf6_reacted_np,
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
            "p_rate": 0.0,
            "k_rate_ps": base_rate_ps,
            "k_eff_ps": 0.0,
            "pf_rate_factor": 0.0,
            "dt_reactive_ps": reactive_interval_ps,
            "candidate_records": candidate_records,
        }, R

    accepted_events = []
    R_current = R
    ff_current = ff
    sys_current = sys

    last_p_rate = 0.0
    last_k_eff = 0.0
    last_pf_factor = 0.0

    for cand in candidates:
        if len(accepted_events) >= max_reactions_per_check:
            break

        if pf6_reacted_np[cand.k_pf6]:
            continue

        sigma = reaction_coordinate(d_pf=cand.d_pf, d_lif=cand.d_lif)
        p_react, k_eff, pf_factor = rate_probability_from_reaction_coordinate(
            sigma=sigma,
            base_rate_ps=base_rate_ps,
            reactive_interval_ps=reactive_interval_ps,
            midpoint=sigma_mid,
            width=sigma_width,
        )

        last_p_rate = p_react
        last_k_eff = k_eff
        last_pf_factor = pf_factor

        key, sub = jax.random.split(key)
        u = float(jax.random.uniform(sub))

        if u >= p_react:
            continue

        trial, _pf6_molid = propose_reaction_trial(
            sys_current,
            cand,
            pf6_atoms_np=pf6_atoms_np,
            atom_types_np=atom_types_np,
            pf5=pf5,
            lif=lif,
            p_type=p_type,
            f_type=f_type,
            li_type=li_type,
        )

        if trial is None:
            continue

        ff_trial = build_trial_forcefield(R_current, box, trial, ff_current)

        P_atom = int(pf6_atoms_np[cand.k_pf6, 0])
        R_probe = make_probe_geometry(
            R_current,
            P_atom=P_atom,
            leave_F=cand.leave_F,
            disp_fn=ff_current.disp_fn,
            shift_fn=shift_fn,
            r_pf_probe=r_pf_probe,
        )

        R_relaxed, _nlist_relaxed = fire_relax_with_nlist(
            R_probe,
            ff_trial=ff_trial,
            shift_fn=shift_fn,
            n_steps=30,
            dt_start=1.0e-3,
            f_inc=1.01,
            dt_max=1.0e-2,
            n_min=2,
        )

        pf6_reacted_np[cand.k_pf6] = True
        sys_current = make_system_state_from_trial(trial, pf6_reacted_np)

        ff_current = ff_trial
        ff_current.nlist = ff_current.neighbor_fn.allocate(R_relaxed)
        R_current = R_relaxed

        event_info = _candidate_info(cand)
        event_info.update(
            {
                "p_rate": p_react,
                "k_rate_ps": base_rate_ps,
                "k_eff_ps": k_eff,
                "pf_rate_factor": pf_factor,
                "dt_reactive_ps": reactive_interval_ps,
                "sigma_mid": sigma_mid,
                "sigma_width": sigma_width,
            }
        )
        accepted_events.append(event_info)

    if not accepted_events:
        return key, False, ff, sys, {
            "mode": "rate",
            "sigma_mid": sigma_mid,
            "sigma_width": sigma_width,
            "n_candidates": len(candidates),
            "n_accepted_this_check": 0,
            "p_rate": last_p_rate,
            "k_rate_ps": base_rate_ps,
            "k_eff_ps": last_k_eff,
            "pf_rate_factor": last_pf_factor,
            "dt_reactive_ps": reactive_interval_ps,
            "candidate_records": candidate_records,
        }, R

    first_event = accepted_events[0]

    return key, True, ff_current, sys_current, {
        "mode": "rate",
        "sigma_mid": sigma_mid,
        "sigma_width": sigma_width,
        "accepted_events": accepted_events,
        "accepted_event": first_event,
        "n_candidates": len(candidates),
        "n_accepted_this_check": len(accepted_events),
        "p_rate": first_event["p_rate"],
        "k_rate_ps": base_rate_ps,
        "activation_energy_eV": activation_energy_eV,
        "temperature_k": temperature_k,
        "prefactor_ps": prefactor_ps,
        "k_eff_ps": first_event["k_eff_ps"],
        "pf_rate_factor": first_event["pf_rate_factor"],
        "dt_reactive_ps": reactive_interval_ps,
        "candidate_records": candidate_records,
    }, R_current


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

