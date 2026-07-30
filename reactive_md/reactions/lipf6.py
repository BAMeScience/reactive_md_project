# reactive_md/reactions/lipf6.py

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ReactionCandidate:
    """One possible PF6 -> PF5 + LiF reaction event."""

    k_pf6: int
    li_idx: int
    leave_F: int
    d_lif: float
    d_pf: float


def _distance(disp_fn, Rj, i: int, j: int) -> float:
    """Return the minimum-image distance between atoms i and j."""
    dr = np.asarray(disp_fn(Rj[int(i)], Rj[int(j)]))
    return float(np.linalg.norm(dr))


def reaction_coordinate(*, d_pf: float, d_lif: float) -> float:
    """Return the LiPF6 decomposition reaction coordinate.

    Based on Fattebert et al., Journal of The Electrochemical Society,
    2024, 171, 080505.

        sigma = d(P-F) - d(Li-F)

    sigma < 0:
        Reactant-like configuration.

    sigma approximately 0:
        Transition region.

    sigma > 0:
        Product-like configuration.
    """
    return float(d_pf - d_lif)


def find_reaction_candidates(
    R,
    pf6_atoms_np: np.ndarray,
    li_atoms_np: np.ndarray,
    disp_fn,
    *,
    pf6_reacted_np: np.ndarray,
) -> list[ReactionCandidate]:
    """Return all possible reaction candidates, ranked by sigma.

    Every fluorine belonging to an unreacted PF6 molecule is paired with every
    Li ion. No independent Li-F or P-F hard cutoff is applied.

    Candidates are ranked by

        sigma = d(P-F) - d(Li-F)

    in descending order.
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

                d_lif = _distance(
                    disp_fn,
                    Rj,
                    li_idx,
                    f_idx,
                )
                d_pf = _distance(
                    disp_fn,
                    Rj,
                    P_atom,
                    f_idx,
                )

                candidates.append(
                    ReactionCandidate(
                        k_pf6=int(k),
                        li_idx=li_idx,
                        leave_F=f_idx,
                        d_lif=d_lif,
                        d_pf=d_pf,
                    )
                )

    candidates.sort(
        key=lambda candidate: reaction_coordinate(
            d_pf=candidate.d_pf,
            d_lif=candidate.d_lif,
        ),
        reverse=True,
    )

    return candidates


def prepare_probe_geometry(
    R,
    *,
    P_atom: int,
    leave_F: int,
    li_idx: int,
    sigma_p: float,
    sigma_f: float,
    disp_fn,
    shift_fn,
    eps: float = 1.0e-8,
):
    """Move the departing F from P toward the reacting Li."""
    r_p = R[P_atom]
    r_f = R[leave_F]
    r_li = R[li_idx]

    r_pf_target = (
        2.0 ** (1.0 / 6.0)
        * 0.5
        * (sigma_p + sigma_f)
    )

    p_to_li = disp_fn(r_p, r_li)
    p_to_li_distance = jnp.linalg.norm(p_to_li)

    direction_to_li = (
        p_to_li
        / jnp.maximum(p_to_li_distance, eps)
    )

    f_probe = shift_fn(
        r_p,
        r_pf_target * direction_to_li,
    )

    geometry_is_valid = p_to_li_distance > r_pf_target

    new_f = jnp.where(
        geometry_is_valid,
        f_probe,
        r_f,
    )

    return R.at[leave_F].set(new_f)


def discover_pf6_and_li(
    atom_types,
    molecule_id,
    *,
    p_type: int,
    f_type: int,
    li_type: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer PF6 molecules and Li atoms from atom types and molecule IDs.

    Returns
    -------
    pf6_atoms
        Integer array with shape ``(n_pf6, 7)``. Each row contains
        ``[P, F1, F2, F3, F4, F5, F6]`` using global atom indices.

    li_atoms
        Integer array with shape ``(n_li,)`` containing global Li indices.
    """
    atom_types_np = np.asarray(atom_types)
    molecule_id_np = np.asarray(molecule_id)

    pf6_blocks: list[np.ndarray] = []

    for molecule in np.unique(molecule_id_np):
        indices = np.where(molecule_id_np == molecule)[0]
        molecule_types = atom_types_np[indices]

        p_indices = indices[molecule_types == p_type]
        f_indices = indices[molecule_types == f_type]

        if p_indices.size == 1 and f_indices.size == 6:
            P_atom = int(p_indices[0])
            fluorines = np.sort(f_indices).astype(np.int32)

            block = np.concatenate(
                [
                    np.array([P_atom], dtype=np.int32),
                    fluorines,
                ]
            )
            pf6_blocks.append(block)

    if pf6_blocks:
        pf6_atoms = np.stack(pf6_blocks).astype(np.int32)
    else:
        pf6_atoms = np.empty((0, 7), dtype=np.int32)

    li_atoms = np.where(
        atom_types_np == li_type
    )[0].astype(np.int32)

    print(
        "[discover_pf6_and_li] "
        f"Found {pf6_atoms.shape[0]} PF6 molecules, "
        f"{li_atoms.shape[0]} Li atoms."
    )

    return pf6_atoms, li_atoms


def embed_pf5_into_pf6(
    pf6_atoms_row: np.ndarray,
    leave_F: int,
    pf5_bond_idx_local: np.ndarray,
    pf5_angle_idx_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map PF5-local topology indices onto a PF6 atom block."""
    P_atom = int(pf6_atoms_row[0])
    fluorines = [int(index) for index in pf6_atoms_row[1:]]
    leave_F = int(leave_F)

    remaining_fluorines = [
        f_idx
        for f_idx in fluorines
        if f_idx != leave_F
    ]

    if len(remaining_fluorines) != 5:
        raise ValueError(
            "leave_F is not one of the six fluorines in this PF6 block."
        )

    pf5_global = np.asarray(
        [P_atom, *remaining_fluorines],
        dtype=np.int32,
    )

    bonds_global = pf5_global[
        np.asarray(pf5_bond_idx_local, dtype=np.int32)
    ]
    angles_global = pf5_global[
        np.asarray(pf5_angle_idx_local, dtype=np.int32)
    ]

    return pf5_global, bonds_global, angles_global
