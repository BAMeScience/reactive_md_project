# reactive_md/topology_ops.py
from __future__ import annotations
import numpy as np

def discover_pf6_and_li(atom_types, molecule_id, *, p_type: int, f_type: int, li_type: int):
    """
    Infer PF6 molecules and Li atoms from atom_types and molecule_id.

    Returns:
      pf6_atoms: (n_pf6, 7) int array [P, F1..F6] (global atom indices)
      li_atoms: (n_li,) int array of Li atom indices
    """
    at = np.array(atom_types)
    mol = np.array(molecule_id)

    unique_molids = np.unique(mol)
    pf6_blocks = []

    for m in unique_molids:
        idx = np.where(mol == m)[0]
        types_m = at[idx]

        p_idx = idx[types_m == p_type]
        f_idx = idx[types_m == f_type]

        if p_idx.size == 1 and f_idx.size == 6:
            P_atom = int(p_idx[0])
            Fs = np.sort(f_idx).astype(int)
            block = np.concatenate([[P_atom], Fs])
            pf6_blocks.append(block)

    pf6_blocks = np.array(pf6_blocks, dtype=np.int32)
    li_atoms = np.where(at == li_type)[0].astype(np.int32)

    print(f"[discover_pf6_and_li] Found {pf6_blocks.shape[0]} PF6 molecules, {li_atoms.shape[0]} Li atoms.")
    return pf6_blocks, li_atoms

def embed_pf5_into_pf6(pf6_atoms_row: np.ndarray, leave_F: int,
                       pf5_bond_idx_local: np.ndarray,
                       pf5_angle_idx_local: np.ndarray):
    """Map PF5 local indices onto a PF6 block, given the leaving F atom."""
    P = int(pf6_atoms_row[0])
    Fs = [int(x) for x in pf6_atoms_row[1:]]
    leave_F = int(leave_F)
    Fs_remain = [f for f in Fs if f != leave_F]
    if len(Fs_remain) != 5:
        raise ValueError("leave_F is not one of the PF6 F atoms for this block.")
    pf5_glob = np.array([P] + Fs_remain, dtype=np.int32)
    bonds_g = pf5_glob[pf5_bond_idx_local]
    angles_g = pf5_glob[pf5_angle_idx_local]
    return pf5_glob, bonds_g, angles_g

def remove_terms_in_molid(term_idx: np.ndarray, params_list: list[np.ndarray],
                          molecule_id_np: np.ndarray, molid: int):
    """
    Remove all bonded terms whose atoms all lie in molecule 'molid'.
    term_idx: (n_terms, n_atoms_in_term)
    params_list: [param1, param2, ...] each of length n_terms
    """
    inside = np.all(molecule_id_np[term_idx] == molid, axis=1)
    keep = ~inside
    term_idx2 = term_idx[keep]
    params2 = [p[keep] for p in params_list]
    return term_idx2, params2

