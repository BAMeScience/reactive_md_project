# reactive_md/reactions/lipf6.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .templates_pf5 import LiFTemplate, PF5Template
from ..topology_opls import remove_terms_in_molid


@dataclass(frozen=True)
class ReactionCandidate:
    """One possible PF6 -> PF5 + LiF reaction event."""

    k_pf6: int
    li_idx: int
    leave_F: int
    d_lif: float
    d_pf: float


@dataclass
class LiPF6Reaction:
    """Definition and immutable topology data for LiPF6 decomposition.

    The object owns the information needed to describe the reaction

        LiPF6 -> PF5 + LiF

    but does not own the dynamically changing system state. In particular,
    the array tracking which PF6 molecules have reacted should remain part
    of the simulation state for now.

    Parameters
    ----------
    pf6_atoms
        Integer array of shape ``(n_pf6, 7)``. Each row contains the global
        atom indices ``[P, F1, F2, F3, F4, F5, F6]``.

    li_atoms
        Integer array of shape ``(n_li,)`` containing global Li indices.

    atom_types
        Integer atom-type array for the complete system.

    pf5
        Product-side PF5 topology and nonbonded-parameter template.

    lif
        Product-side LiF nonbonded-parameter template.

    p_type, f_type, li_type
        Atom-type identifiers used for sanity checks during trial creation.
    """

    pf6_atoms: np.ndarray
    li_atoms: np.ndarray
    atom_types: np.ndarray

    pf5: PF5Template
    lif: LiFTemplate

    p_type: int
    f_type: int
    li_type: int

    def __post_init__(self) -> None:
        """Normalize arrays and validate their basic shapes."""
        self.pf6_atoms = np.asarray(
            self.pf6_atoms,
            dtype=np.int32,
        )
        self.li_atoms = np.asarray(
            self.li_atoms,
            dtype=np.int32,
        )
        self.atom_types = np.asarray(self.atom_types)

        if self.pf6_atoms.ndim != 2 or self.pf6_atoms.shape[1] != 7:
            raise ValueError(
                "pf6_atoms must have shape (n_pf6, 7), with rows "
                "[P, F1, F2, F3, F4, F5, F6]."
            )

        if self.li_atoms.ndim != 1:
            raise ValueError("li_atoms must be a one-dimensional array.")

        if self.atom_types.ndim != 1:
            raise ValueError("atom_types must be a one-dimensional array.")

        if self.atom_types.size:
            all_reaction_indices = np.concatenate(
                [
                    self.pf6_atoms.reshape(-1),
                    self.li_atoms,
                ]
            )

            if all_reaction_indices.size:
                minimum_index = int(all_reaction_indices.min())
                maximum_index = int(all_reaction_indices.max())

                if minimum_index < 0:
                    raise ValueError(
                        "Reaction atom indices must be non-negative."
                    )

                if maximum_index >= self.atom_types.shape[0]:
                    raise ValueError(
                        "Reaction atom index exceeds the atom_types array."
                    )

    @classmethod
    def from_system(
        cls,
        atom_types,
        molecule_id,
        *,
        pf5: PF5Template,
        lif: LiFTemplate,
        p_type: int,
        f_type: int,
        li_type: int,
    ) -> LiPF6Reaction:
        """Discover PF6/Li atoms and construct the reaction definition."""
        atom_types_np = np.asarray(atom_types)

        pf6_atoms, li_atoms = discover_pf6_and_li(
            atom_types_np,
            molecule_id,
            p_type=p_type,
            f_type=f_type,
            li_type=li_type,
        )

        return cls(
            pf6_atoms=pf6_atoms,
            li_atoms=li_atoms,
            atom_types=atom_types_np,
            pf5=pf5,
            lif=lif,
            p_type=int(p_type),
            f_type=int(f_type),
            li_type=int(li_type),
        )

    def find_candidates(
        self,
        R,
        disp_fn,
        *,
        pf6_reacted,
    ) -> list[ReactionCandidate]:
        """Return reaction candidates using data owned by this object."""
        return find_reaction_candidates(
            R,
            self.pf6_atoms,
            self.li_atoms,
            disp_fn,
            pf6_reacted_np=np.asarray(
                pf6_reacted,
                dtype=bool,
            ),
        )

    def phosphorus_index(
        self,
        candidate: ReactionCandidate,
    ) -> int:
        """Return the phosphorus index associated with a candidate."""
        self._validate_candidate(candidate)
        return int(self.pf6_atoms[candidate.k_pf6, 0])

    def candidate_types_are_valid(
        self,
        candidate: ReactionCandidate,
    ) -> bool:
        """Check that a candidate has the expected P, F, and Li atom types."""
        self._validate_candidate(candidate)

        phosphorus = self.phosphorus_index(candidate)

        return bool(
            self.atom_types[phosphorus] == self.p_type
            and self.atom_types[candidate.leave_F] == self.f_type
            and self.atom_types[candidate.li_idx] == self.li_type
        )

    def embed_pf5(
        self,
        candidate: ReactionCandidate,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Embed the PF5 template for a selected reaction candidate."""
        self._validate_candidate(candidate)

        return embed_pf5_into_pf6(
            self.pf6_atoms[candidate.k_pf6],
            candidate.leave_F,
            pf5_bond_idx_local=self.pf5.bond_idx_local,
            pf5_angle_idx_local=self.pf5.angle_idx_local,
        )

    def prepare_probe(
        self,
        R,
        candidate: ReactionCandidate,
        *,
        sigma_p: float,
        sigma_f: float,
        disp_fn,
        shift_fn,
        eps: float = 1.0e-8,
    ):
        """Construct the product-side probe geometry for a candidate."""
        phosphorus = self.phosphorus_index(candidate)

        return prepare_probe_geometry(
            R,
            P_atom=phosphorus,
            leave_F=candidate.leave_F,
            li_idx=candidate.li_idx,
            sigma_p=sigma_p,
            sigma_f=sigma_f,
            disp_fn=disp_fn,
            shift_fn=shift_fn,
            eps=eps,
        )

    def build_trial(
        self,
        system: Any,
        candidate: ReactionCandidate,
    ) -> tuple[dict[str, Any] | None, int | None]:
        """Build the product-side topology and nonbonded parameters.

        The returned dictionary has the same shape as the current reaction
        engine expects. A failed atom-type sanity check returns
        ``(None, None)``.
        """
        self._validate_candidate(candidate)

        if not self.candidate_types_are_valid(candidate):
            return None, None

        k_pf6 = int(candidate.k_pf6)
        li_idx = int(candidate.li_idx)
        leave_f = int(candidate.leave_F)
        phosphorus = self.phosphorus_index(candidate)

        bond_idx, k_b, r0 = (
            np.asarray(system.bonds[0], dtype=np.int32),
            np.asarray(system.bonds[1], dtype=np.float32),
            np.asarray(system.bonds[2], dtype=np.float32),
        )
        angle_idx, k_theta, theta0 = (
            np.asarray(system.angles[0], dtype=np.int32),
            np.asarray(system.angles[1], dtype=np.float32),
            np.asarray(system.angles[2], dtype=np.float32),
        )
        tors_idx, tors_k, tors_n, tors_gamma = (
            np.asarray(system.torsions[0], dtype=np.int32),
            np.asarray(system.torsions[1], dtype=np.float32),
            np.asarray(system.torsions[2], dtype=np.int32),
            np.asarray(system.torsions[3], dtype=np.float32),
        )
        impr_idx, impr_k, impr_n, impr_gamma = (
            np.asarray(system.impropers[0], dtype=np.int32),
            np.asarray(system.impropers[1], dtype=np.float32),
            np.asarray(system.impropers[2], dtype=np.int32),
            np.asarray(system.impropers[3], dtype=np.float32),
        )

        charges = np.asarray(system.charges, dtype=np.float32).copy()
        sigmas = np.asarray(system.sigmas, dtype=np.float32).copy()
        epsilons = np.asarray(system.epsilons, dtype=np.float32).copy()
        molecule_id = np.asarray(system.molecule_id, dtype=np.int32)

        pf6_molid = int(molecule_id[phosphorus])

        bond_idx, (k_b, r0) = remove_terms_in_molid(
            bond_idx, [k_b, r0], molecule_id, pf6_molid
        )
        angle_idx, (k_theta, theta0) = remove_terms_in_molid(
            angle_idx, [k_theta, theta0], molecule_id, pf6_molid
        )
        tors_idx, (tors_k, tors_n, tors_gamma) = remove_terms_in_molid(
            tors_idx,
            [tors_k, tors_n, tors_gamma],
            molecule_id,
            pf6_molid,
        )
        impr_idx, (impr_k, impr_n, impr_gamma) = remove_terms_in_molid(
            impr_idx,
            [impr_k, impr_n, impr_gamma],
            molecule_id,
            pf6_molid,
        )

        pf5_global, pf5_bonds, pf5_angles = self.embed_pf5(candidate)

        bond_idx = np.concatenate([bond_idx, pf5_bonds], axis=0)
        k_b = np.concatenate([k_b, np.asarray(self.pf5.k_b)], axis=0)
        r0 = np.concatenate([r0, np.asarray(self.pf5.r0)], axis=0)

        angle_idx = np.concatenate([angle_idx, pf5_angles], axis=0)
        k_theta = np.concatenate(
            [k_theta, np.asarray(self.pf5.k_theta)], axis=0
        )
        theta0 = np.concatenate(
            [theta0, np.asarray(self.pf5.theta0)], axis=0
        )

        product_phosphorus = int(pf5_global[0])
        product_fluorines = pf5_global[1:]

        charges[product_phosphorus] = self.pf5.q["P"]
        charges[product_fluorines] = self.pf5.q["F"]

        epsilons[product_phosphorus], sigmas[product_phosphorus] = (
            self.pf5.pair["P"]
        )
        epsilons[product_fluorines], sigmas[product_fluorines] = (
            self.pf5.pair["F"]
        )

        charges[leave_f] = self.lif.nb["F"]["q"]
        sigmas[leave_f] = self.lif.nb["F"]["sigma"]
        epsilons[leave_f] = self.lif.nb["F"]["eps"]

        charges[li_idx] = self.lif.nb["Li"]["q"]
        sigmas[li_idx] = self.lif.nb["Li"]["sigma"]
        epsilons[li_idx] = self.lif.nb["Li"]["eps"]

        molecule_id_product = molecule_id.copy()
        new_molid = int(molecule_id_product.max()) + 1
        molecule_id_product[leave_f] = new_molid

        trial = {
            "bonds": (bond_idx, k_b, r0),
            "angles": (angle_idx, k_theta, theta0),
            "torsions": (tors_idx, tors_k, tors_n, tors_gamma),
            "impropers": (impr_idx, impr_k, impr_n, impr_gamma),
            "charges": charges,
            "sigmas": sigmas,
            "epsilons": epsilons,
            "molecule_id": molecule_id_product,
        }
        return trial, pf6_molid

    def _validate_candidate(
        self,
        candidate: ReactionCandidate,
    ) -> None:
        """Validate candidate indices against this reaction definition."""
        if not isinstance(candidate, ReactionCandidate):
            raise TypeError(
                "candidate must be a lipf6.ReactionCandidate instance."
            )

        if not 0 <= candidate.k_pf6 < self.pf6_atoms.shape[0]:
            raise IndexError(
                f"PF6 index {candidate.k_pf6} is outside the available "
                f"range 0..{self.pf6_atoms.shape[0] - 1}."
            )

        pf6_row = self.pf6_atoms[candidate.k_pf6]

        if candidate.leave_F not in pf6_row[1:]:
            raise ValueError(
                "The candidate's leaving fluorine does not belong to "
                "the selected PF6 molecule."
            )

        if candidate.li_idx not in self.li_atoms:
            raise ValueError(
                "The candidate's Li index is not part of this reaction "
                "definition."
            )


def _distance(
    disp_fn,
    Rj,
    i: int,
    j: int,
) -> float:
    """Return the minimum-image distance between atoms i and j."""
    displacement = np.asarray(
        disp_fn(
            Rj[int(i)],
            Rj[int(j)],
        )
    )
    return float(np.linalg.norm(displacement))


def reaction_coordinate(
    *,
    d_pf: float,
    d_lif: float,
) -> float:
    """Return the LiPF6 decomposition reaction coordinate.

    Based on Fattebert et al., Journal of The Electrochemical Society,
    2024, 171, 080505.

        sigma = d(P-F) - d(Li-F)

    sigma < 0
        Reactant-like configuration.

    sigma approximately 0
        Transition region.

    sigma > 0
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

    Every fluorine belonging to an unreacted PF6 molecule is paired with
    every Li ion. No independent Li-F or P-F hard cutoff is applied.

    Candidates are ranked by

        sigma = d(P-F) - d(Li-F)

    in descending order.

    This free function is retained for compatibility with the current
    reaction engine. New code may use ``LiPF6Reaction.find_candidates()``.
    """
    pf6_atoms_np = np.asarray(
        pf6_atoms_np,
        dtype=np.int32,
    )
    li_atoms_np = np.asarray(
        li_atoms_np,
        dtype=np.int32,
    )
    pf6_reacted_np = np.asarray(
        pf6_reacted_np,
        dtype=bool,
    )

    if pf6_atoms_np.ndim != 2 or pf6_atoms_np.shape[1] != 7:
        raise ValueError(
            "pf6_atoms_np must have shape (n_pf6, 7)."
        )

    if li_atoms_np.ndim != 1:
        raise ValueError(
            "li_atoms_np must be one-dimensional."
        )

    if pf6_reacted_np.shape != (pf6_atoms_np.shape[0],):
        raise ValueError(
            "pf6_reacted_np must contain one boolean value per PF6 "
            "molecule."
        )

    Rj = jnp.asarray(R)
    candidates: list[ReactionCandidate] = []

    for k_pf6 in range(pf6_atoms_np.shape[0]):
        if pf6_reacted_np[k_pf6]:
            continue

        phosphorus = int(pf6_atoms_np[k_pf6, 0])
        fluorines = pf6_atoms_np[k_pf6, 1:]

        for li_atom in li_atoms_np:
            li_idx = int(li_atom)

            for fluorine in fluorines:
                fluorine_idx = int(fluorine)

                d_lif = _distance(
                    disp_fn,
                    Rj,
                    li_idx,
                    fluorine_idx,
                )
                d_pf = _distance(
                    disp_fn,
                    Rj,
                    phosphorus,
                    fluorine_idx,
                )

                candidates.append(
                    ReactionCandidate(
                        k_pf6=int(k_pf6),
                        li_idx=li_idx,
                        leave_F=fluorine_idx,
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
    """Move the departing F from P toward the reacting Li.

    The target P-F separation is the Lennard-Jones equilibrium distance
    derived from Lorentz-mixed P and F sigma values:

        r_target = 2**(1/6) * (sigma_p + sigma_f) / 2

    If the P-Li distance is not larger than that target distance, the
    original F position is retained.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    Rj = jnp.asarray(R)

    phosphorus = int(P_atom)
    fluorine = int(leave_F)
    lithium = int(li_idx)

    r_p = Rj[phosphorus]
    r_f = Rj[fluorine]
    r_li = Rj[lithium]

    r_pf_target = (
        2.0 ** (1.0 / 6.0)
        * 0.5
        * (float(sigma_p) + float(sigma_f))
    )

    if not np.isfinite(r_pf_target) or r_pf_target <= 0.0:
        raise ValueError(
            "The force-field-derived P-F target distance must be "
            "finite and positive."
        )

    p_to_li = disp_fn(r_p, r_li)
    p_to_li_distance = jnp.linalg.norm(p_to_li)

    direction_to_li = (
        p_to_li
        / jnp.maximum(
            p_to_li_distance,
            jnp.asarray(eps, dtype=p_to_li_distance.dtype),
        )
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

    return Rj.at[fluorine].set(new_f)


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

    if atom_types_np.ndim != 1:
        raise ValueError("atom_types must be one-dimensional.")

    if molecule_id_np.ndim != 1:
        raise ValueError("molecule_id must be one-dimensional.")

    if atom_types_np.shape[0] != molecule_id_np.shape[0]:
        raise ValueError(
            "atom_types and molecule_id must contain the same number "
            "of atoms."
        )

    pf6_blocks: list[np.ndarray] = []

    for molecule in np.unique(molecule_id_np):
        indices = np.where(
            molecule_id_np == molecule
        )[0]

        molecule_types = atom_types_np[indices]

        phosphorus_indices = indices[
            molecule_types == p_type
        ]
        fluorine_indices = indices[
            molecule_types == f_type
        ]

        if (
            phosphorus_indices.size == 1
            and fluorine_indices.size == 6
        ):
            phosphorus = int(phosphorus_indices[0])
            fluorines = np.sort(
                fluorine_indices
            ).astype(np.int32)

            block = np.concatenate(
                [
                    np.asarray(
                        [phosphorus],
                        dtype=np.int32,
                    ),
                    fluorines,
                ]
            )

            pf6_blocks.append(block)

    if pf6_blocks:
        pf6_atoms = np.stack(
            pf6_blocks,
            axis=0,
        ).astype(np.int32)
    else:
        pf6_atoms = np.empty(
            (0, 7),
            dtype=np.int32,
        )

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
    pf6_atoms_row = np.asarray(
        pf6_atoms_row,
        dtype=np.int32,
    )

    if pf6_atoms_row.shape != (7,):
        raise ValueError(
            "pf6_atoms_row must contain exactly seven indices: "
            "[P, F1, F2, F3, F4, F5, F6]."
        )

    phosphorus = int(pf6_atoms_row[0])
    fluorines = [
        int(index)
        for index in pf6_atoms_row[1:]
    ]
    leave_F = int(leave_F)

    remaining_fluorines = [
        fluorine
        for fluorine in fluorines
        if fluorine != leave_F
    ]

    if len(remaining_fluorines) != 5:
        raise ValueError(
            "leave_F is not exactly one of the six fluorines in this "
            "PF6 block."
        )

    pf5_global = np.asarray(
        [
            phosphorus,
            *remaining_fluorines,
        ],
        dtype=np.int32,
    )

    bond_indices_local = np.asarray(
        pf5_bond_idx_local,
        dtype=np.int32,
    )
    angle_indices_local = np.asarray(
        pf5_angle_idx_local,
        dtype=np.int32,
    )

    if bond_indices_local.size:
        if (
            int(bond_indices_local.min()) < 0
            or int(bond_indices_local.max()) >= pf5_global.size
        ):
            raise IndexError(
                "PF5 local bond indices must lie between 0 and 5."
            )

    if angle_indices_local.size:
        if (
            int(angle_indices_local.min()) < 0
            or int(angle_indices_local.max()) >= pf5_global.size
        ):
            raise IndexError(
                "PF5 local angle indices must lie between 0 and 5."
            )

    bonds_global = pf5_global[bond_indices_local]
    angles_global = pf5_global[angle_indices_local]

    return pf5_global, bonds_global, angles_global
