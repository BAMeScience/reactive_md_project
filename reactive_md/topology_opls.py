# reactive_md/topology_opls.py
from __future__ import annotations
import numpy as np

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

