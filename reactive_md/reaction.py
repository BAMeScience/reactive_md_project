# reactive_md/reaction.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax_md.minimize import fire_descent

from .templates_pf5 import PF5Template, LiFTemplate
from .topology_opls import embed_pf5_into_pf6, remove_terms_in_molid
from .forcefield import FFBundle, build_forcefield


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
class Candidate:
    k_pf6: int
    li_idx: int
    leave_F: int
    d_lif: float
    d_pf: float


def _distance(disp_fn, Rj, i: int, j: int) -> float:
    dr = np.asarray(disp_fn(Rj[int(i)], Rj[int(j)]))
    return float(np.linalg.norm(dr))


def find_best_candidate(
    R,
    pf6_atoms_np: np.ndarray,
    li_atoms_np: np.ndarray,
    disp_fn,
    *,
    r_lif_on: float,
    r_pf_break: float,
    pf6_reacted_np: np.ndarray,
) -> Optional[Candidate]:
    """
    Find the best physically plausible PF6 + Li reaction candidate.

    Required gates:
    - Li-F must be a close contact: d_LiF < r_lif_on
    - The leaving P-F bond must already be stretched: d_PF > r_pf_break

    This prevents accepting distant Li-F pairs merely because FIRE relaxation
    can later repair a trial product geometry.
    """
    Rj = jnp.asarray(R)

    best: Optional[Candidate] = None
    best_score = 1.0e30

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

                if d_lif >= r_lif_on:
                    continue
                if d_pf <= r_pf_break:
                    continue

                # Prefer closer Li-F contacts, with a small preference for
                # more stretched P-F bonds.
                score = d_lif - 0.1 * d_pf

                if score < best_score:
                    best_score = score
                    best = Candidate(
                        k_pf6=int(k),
                        li_idx=li_idx,
                        leave_F=f_idx,
                        d_lif=float(d_lif),
                        d_pf=float(d_pf),
                    )

    return best

def find_all_candidates(
    R,
    pf6_atoms_np,
    li_atoms_np,
    disp_fn,
    *,
    r_lif_on,
    r_pf_break,
    pf6_reacted_np,
):
    Rj = jnp.asarray(R)
    candidates = []

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

                if d_lif < r_lif_on and d_pf > r_pf_break:
                    candidates.append(
                        Candidate(
                            k_pf6=int(k),
                            li_idx=li_idx,
                            leave_F=f_idx,
                            d_lif=float(d_lif),
                            d_pf=float(d_pf),
                        )
                    )

    candidates.sort(key=lambda c: (c.d_lif, -c.d_pf))
    return candidates

def find_closest_lif_pair(
    R,
    pf6_atoms_np: np.ndarray,
    li_atoms_np: np.ndarray,
    disp_fn,
    *,
    pf6_reacted_np: np.ndarray,
) -> Optional[Candidate]:
    """
    Diagnostic helper: return the closest Li-F pair, even if it fails gates.
    """
    Rj = jnp.asarray(R)

    best: Optional[Candidate] = None
    best_d = 1.0e30

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

                if d_lif < best_d:
                    best_d = d_lif
                    best = Candidate(
                        k_pf6=int(k),
                        li_idx=li_idx,
                        leave_F=f_idx,
                        d_lif=float(d_lif),
                        d_pf=float(d_pf),
                    )

    return best


def make_probe_geometry(
    R,
    *,
    P_atom: int,
    leave_F: int,
    disp_fn,
    shift_fn,
    r_pf_probe: float = 4.0,
):
    """
    Push leaving F away from P along the old P->F direction.
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
    cand: Candidate,
    *,
    pf6_atoms_np: np.ndarray,
    atom_types_np: np.ndarray,
    pf5: PF5Template,
    lif: LiFTemplate,
    p_type: int,
    f_type: int,
    li_type: int,
):
    """
    Build trial topology and nonbonded arrays for PF6 -> PF5 + LiF.
    """
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


def accept_reject(key, dE: float, beta: float):
    p_acc = min(1.0, float(np.exp(-beta * dE)))
    key, sub = jax.random.split(key)
    u = float(jax.random.uniform(sub))
    return key, (u < p_acc), p_acc


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
    r_lif_on: float,
    r_pf_break: float,
    r_pf_probe: float,
    beta: float,
    mc_energy_evaluator=None,
):
    """
    Attempt one PF6 -> PF5 + LiF reaction.

    A trial topology is only built after the geometry gate passes. This avoids
    repeatedly compiling FIRE relaxation for clearly unphysical distant pairs.
    """
    pf6_atoms_np = np.array(pf6_atoms, dtype=np.int32)
    li_atoms_np = np.array(li_atoms, dtype=np.int32)
    pf6_reacted_np = np.array(sys.pf6_reacted, dtype=bool)
    atom_types_np = np.array(atom_types)

    cand = find_best_candidate(
        R,
        pf6_atoms_np,
        li_atoms_np,
        ff.disp_fn,
        r_lif_on=r_lif_on,
        r_pf_break=r_pf_break,
        pf6_reacted_np=pf6_reacted_np,
    )

    if cand is None:
        closest = find_closest_lif_pair(
            R,
            pf6_atoms_np,
            li_atoms_np,
            ff.disp_fn,
            pf6_reacted_np=pf6_reacted_np,
        )

        info = {}
        if closest is not None:
            info["closest"] = {
                "k_pf6": closest.k_pf6,
                "li_idx": closest.li_idx,
                "leave_F": closest.leave_F,
                "d_lif": closest.d_lif,
                "d_pf": closest.d_pf,
            }

        return key, False, ff, sys, info, R

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
        return key, False, ff, sys, {"reason": "type_sanity_failed"}, R

    ff_trial = build_forcefield(
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
        r_cut=float(ff.nb_options.r_cut),
        dr_threshold=float(ff.nb_options.dr_threshold),
    )

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

    if mc_energy_evaluator is None:
       E_after_arr = ff_trial.energy_fn(R_relaxed, nlist_relaxed)["total"]
       E_after_arr.block_until_ready()
       E_after = float(E_after_arr)
    else:
       E_after = mc_energy_evaluator.energy(R_relaxed)

    dE = E_after - E_before
    key, accepted, p_acc = accept_reject(key, dE, beta)

    candidate_info = {
        "k_pf6": cand.k_pf6,
        "li_idx": cand.li_idx,
        "leave_F": cand.leave_F,
        "d_lif": cand.d_lif,
        "d_pf": cand.d_pf,
    }

    if not accepted:
        return key, False, ff, sys, {
            "candidate": candidate_info,
            "dE": dE,
            "p_acc": p_acc,
        }, R

    pf6_reacted_np[cand.k_pf6] = True

    sys_new = SystemState(
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

    return key, True, ff_trial, sys_new, {
        "accepted_event": candidate_info,
        "dE": dE,
        "p_acc": p_acc,
    }, R_relaxed


def maybe_react_rate_events(
    key,
    R,
    box,
    *,
    shift_fn,
    ff,
    sys,
    pf6_atoms,
    li_atoms,
    atom_types,
    pf5,
    lif,
    p_type,
    f_type,
    li_type,
    r_lif_on,
    r_pf_break,
    r_pf_probe,
    reaction_rate_ps,
    reactive_interval_ps,
    max_reactions_per_check=1,
):
    pf6_atoms_np = np.array(pf6_atoms, dtype=np.int32)
    li_atoms_np = np.array(li_atoms, dtype=np.int32)
    pf6_reacted_np = np.array(sys.pf6_reacted, dtype=bool)
    atom_types_np = np.array(atom_types)

    candidates = find_all_candidates(
        R,
        pf6_atoms_np,
        li_atoms_np,
        ff.disp_fn,
        r_lif_on=r_lif_on,
        r_pf_break=r_pf_break,
        pf6_reacted_np=pf6_reacted_np,
    )

    if not candidates:
        closest = find_closest_lif_pair(
            R,
            pf6_atoms_np,
            li_atoms_np,
            ff.disp_fn,
            pf6_reacted_np=pf6_reacted_np,
        )
        info = {"mode": "rate", "n_candidates": 0}
        if closest is not None:
            info["closest"] = {
                "k_pf6": closest.k_pf6,
                "li_idx": closest.li_idx,
                "leave_F": closest.leave_F,
                "d_lif": closest.d_lif,
                "d_pf": closest.d_pf,
            }
        return key, False, ff, sys, info, R

    p_react = 1.0 - float(np.exp(-reaction_rate_ps * reactive_interval_ps))

    accepted_events = []
    R_current = R
    ff_current = ff
    sys_current = sys

    for cand in candidates:
        if len(accepted_events) >= max_reactions_per_check:
            break

        if pf6_reacted_np[cand.k_pf6]:
            continue

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

        ff_trial = build_forcefield(
            R=R_current,
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
            r_cut=float(ff_current.nb_options.r_cut),
            dr_threshold=float(ff_current.nb_options.dr_threshold),
        )

        P_atom = int(pf6_atoms_np[cand.k_pf6, 0])
        R_probe = make_probe_geometry(
            R_current,
            P_atom=P_atom,
            leave_F=cand.leave_F,
            disp_fn=ff_current.disp_fn,
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

        pf6_reacted_np[cand.k_pf6] = True

        sys_current = SystemState(
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

        ff_current = ff_trial
        ff_current.nlist = ff_current.neighbor_fn.allocate(R_relaxed)
        R_current = R_relaxed

        accepted_events.append(
            {
                "k_pf6": cand.k_pf6,
                "li_idx": cand.li_idx,
                "leave_F": cand.leave_F,
                "d_lif": cand.d_lif,
                "d_pf": cand.d_pf,
                "p_rate": p_react,
                "k_rate_ps": reaction_rate_ps,
                "dt_reactive_ps": reactive_interval_ps,
            }
        )

    if not accepted_events:
        return key, False, ff, sys, {
            "mode": "rate",
            "n_candidates": len(candidates),
            "p_rate": p_react,
        }, R

    return key, True, ff_current, sys_current, {
        "mode": "rate",
        "accepted_events": accepted_events,
        "accepted_event": accepted_events[0],
        "n_candidates": len(candidates),
        "p_rate": p_react,
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
    """
    Run FIRE minimization under trial topology, updating the neighbor list.
    """
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
