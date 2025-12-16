# reactive_md/reaction.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
import jax
import jax.numpy as jnp

from .templates_pf5 import PF5Template, LiFTemplate
from .topology_opls import embed_pf5_into_pf6, remove_terms_in_molid
from .forcefield import FFBundle, build_forcefield

@dataclass
class SystemState:
    bonds: tuple  # (bond_idx, k_b, r0)
    angles: tuple # (angle_idx, k_theta, theta0)
    torsions: tuple  # (idx, k, n, gamma)
    impropers: tuple # (idx, k, n, gamma)
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
    dmin: float

def find_best_candidate(R, pf6_atoms_np, li_atoms_np, disp_fn, r_on, pf6_reacted_np) -> Optional[Candidate]:
    """
    Brute-force best Li–F candidate across PF6 blocks & Li atoms.
    Returns Candidate or None if best_d > r_on.
    (Kept structurally similar to your original.)
    """
    Rj = jnp.asarray(R)
    best = None
    best_d = 1e30
    for k in range(pf6_atoms_np.shape[0]):
        if pf6_reacted_np[k]:
            continue
        Fs = pf6_atoms_np[k, 1:]
        for li in li_atoms_np:
            dmin = 1e30
            best_f = -1
            for f in Fs:
                dr = np.array(disp_fn(Rj[int(li)], Rj[int(f)]))
                d = float(np.linalg.norm(dr))
                if d < dmin:
                    dmin = d
                    best_f = int(f)
            if dmin < best_d:
                best_d = dmin
                best = Candidate(int(k), int(li), int(best_f), float(dmin))
    if best is None or best_d > r_on:
        return None
    return best

def make_probe_geometry(R, *, P_atom: int, leave_F: int, disp_fn, shift_fn, r_pf_probe: float = 4.0):
    """
    Push leaving F away from P along the old P->F direction (PBC-aware).
    Same probe idea as your script.
    """
    rP = R[P_atom]
    rF = R[leave_F]

    PF_vec = disp_fn(rP, rF)                 # vector from P -> F (PBC-aware)
    PF_dist = jnp.linalg.norm(PF_vec) + 1e-12
    uPF = PF_vec / PF_dist

    rF_new = rP + r_pf_probe * uPF
    drF = rF_new - rF
    R_probe = R.at[leave_F].set(shift_fn(rF, drF))
    return R_probe

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
    Build trial arrays (bonded terms swapped; charges/sigma/epsilon updated; leaving F molid changed).
    Mirrors your original block.
    Returns (trial_arrays_dict, pf6_reacted_np_updated).
    """
    k_pf6, li_idx, leave_F = cand.k_pf6, cand.li_idx, cand.leave_F

    P_atom = int(pf6_atoms_np[k_pf6, 0])
    if atom_types_np[P_atom] != p_type or atom_types_np[leave_F] != f_type or atom_types_np[li_idx] != li_type:
        return None, None  # sanity fail

    # Unpack current topology to numpy
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

    # PF6 molecule id
    pf6_molid = int(molecule_id_np[P_atom])

    # 1) Remove PF6 internal bonded terms
    bond_idx2, (k_b2, r0_2) = remove_terms_in_molid(bond_idx, [k_b, r0], molecule_id_np, pf6_molid)
    angle_idx2, (k_th2, th0_2) = remove_terms_in_molid(angle_idx, [k_theta, theta0], molecule_id_np, pf6_molid)
    tors_idx2, (tors_k2, tors_n2, tors_g2) = remove_terms_in_molid(
        tors_idx, [tors_k, tors_n, tors_gamma], molecule_id_np, pf6_molid
    )
    impr_idx2, (impr_k2, impr_n2, impr_g2) = remove_terms_in_molid(
        impr_idx, [impr_k, impr_n, impr_gamma], molecule_id_np, pf6_molid
    )

    # 2) Embed PF5 bonded terms
    pf5_glob, pf5_bonds_g, pf5_angles_g = embed_pf5_into_pf6(
        pf6_atoms_np[k_pf6], leave_F,
        pf5_bond_idx_local=pf5.bond_idx_local,
        pf5_angle_idx_local=pf5.angle_idx_local,
    )

    bond_idx2 = np.concatenate([bond_idx2, pf5_bonds_g], axis=0)
    k_b2 = np.concatenate([k_b2, pf5.k_b], axis=0)
    r0_2 = np.concatenate([r0_2, pf5.r0], axis=0)

    angle_idx2 = np.concatenate([angle_idx2, pf5_angles_g], axis=0)
    k_th2 = np.concatenate([k_th2, pf5.k_theta], axis=0)
    th0_2 = np.concatenate([th0_2, pf5.theta0], axis=0)

    # 3) Update PF5 nonbonded
    P_pf5 = int(pf5_glob[0])
    Fs_pf5 = pf5_glob[1:]

    charges_np[P_pf5] = pf5.q["P"]
    charges_np[Fs_pf5] = pf5.q["F"]

    eps_np[P_pf5], sigmas_np[P_pf5] = pf5.pair["P"]
    eps_np[Fs_pf5], sigmas_np[Fs_pf5] = pf5.pair["F"]

    # 4) LiF nonbonded for leaving F and Li
    charges_np[leave_F] = lif.nb["F"]["q"]
    sigmas_np[leave_F]  = lif.nb["F"]["sigma"]
    eps_np[leave_F]     = lif.nb["F"]["eps"]

    charges_np[li_idx] = lif.nb["Li"]["q"]
    sigmas_np[li_idx]  = lif.nb["Li"]["sigma"]
    eps_np[li_idx]     = lif.nb["Li"]["eps"]

    # 5) Leave F as separate molecule
    molecule_id2 = molecule_id_np.copy()
    new_molid = int(molecule_id2.max()) + 1
    molecule_id2[leave_F] = new_molid

    return dict(
        bonds=(bond_idx2, k_b2, r0_2),
        angles=(angle_idx2, k_th2, th0_2),
        torsions=(tors_idx2, tors_k2, tors_n2, tors_g2),
        impropers=(impr_idx2, impr_k2, impr_n2, impr_g2),
        charges=charges_np,
        sigmas=sigmas_np,
        epsilons=eps_np,
        molecule_id=molecule_id2,
    ), pf6_molid

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
    r_on: float,
    beta: float,
):
    """
    Performs one reaction attempt (0 or 1 event), returns updated (key, accepted, ff, sys, info).
    Kept aligned with your current behavior.
    """
    pf6_atoms_np = np.array(pf6_atoms, dtype=np.int32)
    li_atoms_np = np.array(li_atoms, dtype=np.int32)
    pf6_reacted_np = np.array(sys.pf6_reacted, dtype=bool)
    atom_types_np = np.array(atom_types)

    cand = find_best_candidate(R, pf6_atoms_np, li_atoms_np, ff.disp_fn, r_on, pf6_reacted_np)
    if cand is None:
        # debug: closest Li–F overall (even if > r_on)
        best_any = find_best_candidate(R, pf6_atoms_np, li_atoms_np, ff.disp_fn, 1e9, pf6_reacted_np)
        if best_any is not None:
            print(
                f"  [debug] No candidate under r_on={r_on:.3f}. "
                f"Closest Li–F: Li={best_any.li_idx}, F={best_any.leave_F}, d={best_any.dmin:.3f} Å"
            )
        else:
            print("  [debug] No Li–F pairs found at all (unexpected).")
        return key, False, ff, sys, {}

    # probe geometry
    P_atom = int(pf6_atoms_np[cand.k_pf6, 0])
    R_probe = make_probe_geometry(R, P_atom=P_atom, leave_F=cand.leave_F,
                                  disp_fn=ff.disp_fn, shift_fn=shift_fn, r_pf_probe=4.0)

    # Energy before (NOTE: kept compatible with your original flow)
    E_before = float(ff.energy_fn(R_probe, ff.nlist)["total"])

    trial, _pf6_molid = propose_reaction_trial(
        sys, cand,
        pf6_atoms_np=pf6_atoms_np,
        atom_types_np=atom_types_np,
        pf5=pf5,
        lif=lif,
        p_type=p_type, f_type=f_type, li_type=li_type,
    )
    if trial is None:
        print("  [debug] Type sanity failed for candidate; skipping.")
        return key, False, ff, sys, {}

    # Rebuild trial force field
    ff_trial = build_forcefield(
        R=R,
        box=box,
        bond_idx=trial["bonds"][0], k_b=trial["bonds"][1], r0=trial["bonds"][2],
        angle_idx=trial["angles"][0], k_theta=trial["angles"][1], theta0=trial["angles"][2],
        torsions=trial["torsions"],
        impropers=trial["impropers"],
        charges=trial["charges"],
        sigmas=trial["sigmas"],
        epsilons=trial["epsilons"],
        molecule_id=trial["molecule_id"],
        r_cut=float(ff.nb_options.r_cut),
        dr_threshold=float(ff.nb_options.dr_threshold),
    )

    E_after = float(ff_trial.energy_fn(R_probe, ff_trial.nlist)["total"])
    dE = E_after - E_before

    key, accepted, p_acc = accept_reject(key, dE, beta)
    if not accepted:
        return key, False, ff, sys, {
            "candidate": (cand.k_pf6, cand.li_idx, cand.leave_F, cand.dmin),
            "dE": dE,
            "p_acc": p_acc,
        }

    # accepted: update sys and pf6_reacted flag
    pf6_reacted_np[cand.k_pf6] = True
    sys_new = SystemState(
        bonds=(jnp.array(trial["bonds"][0], dtype=int), jnp.array(trial["bonds"][1]), jnp.array(trial["bonds"][2])),
        angles=(jnp.array(trial["angles"][0], dtype=int), jnp.array(trial["angles"][1]), jnp.array(trial["angles"][2])),
        torsions=(jnp.array(trial["torsions"][0], dtype=int), jnp.array(trial["torsions"][1]), jnp.array(trial["torsions"][2]), jnp.array(trial["torsions"][3])),
        impropers=(jnp.array(trial["impropers"][0], dtype=int), jnp.array(trial["impropers"][1]), jnp.array(trial["impropers"][2]), jnp.array(trial["impropers"][3])),
        charges=jnp.array(trial["charges"]),
        sigmas=jnp.array(trial["sigmas"]),
        epsilons=jnp.array(trial["epsilons"]),
        molecule_id=jnp.array(trial["molecule_id"], dtype=int),
        pf6_reacted=jnp.array(pf6_reacted_np),
    )

    return key, True, ff_trial, sys_new, {
        "accepted_event": (cand.k_pf6, cand.li_idx, cand.leave_F, cand.dmin),
        "dE": dE,
        "p_acc": p_acc,
    }

