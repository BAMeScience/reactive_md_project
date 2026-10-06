# reactive_md/reactions/a2.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class A2AssociationCandidate:
    """Candidate for A + A -> A2."""

    i: int
    j: int
    distance: float


@dataclass(frozen=True)
class A2DissociationCandidate:
    """Candidate for A2 -> A + A."""

    i: int
    j: int
    bond_index: int
    distance: float


class A2Reaction:
    """Reaction model for reversible A-particle dimerization.

    The benchmark reaction is

        A + A <-> A2

    Unbound A-A pairs interact through the Lennard-Jones potential.
    Bonded A-A pairs are represented by a harmonic bond. The existing
    JAX-MD topology machinery excludes the directly bonded pair from
    the nonbonded interaction.

    Notes
    -----
    `k_bond_jax` must use the JAX-MD/OPLS-AA bond convention

        V_bond = k_bond_jax * (r - r0)^2

    The rs@MD benchmark paper uses

        V_bond = 0.5 * k_bond_paper * (r - r0)^2 + V0

    so reproducing that curvature requires

        k_bond_jax = 0.5 * k_bond_paper

    after unit conversion.

    The constant energy offset V0 is intentionally not included here.
    It does not affect forces and can later enter the Metropolis
    correction separately.
    """

    def __init__(
        self,
        *,
        a_indices,
        k_bond_jax: float,
        r0_bond: float,
        association_cutoff: float = 2.7,
    ):
        self.a_indices = np.asarray(
            a_indices,
            dtype=np.int32,
        )

        self.k_bond_jax = float(k_bond_jax)
        self.r0_bond = float(r0_bond)
        self.association_cutoff = float(association_cutoff)

        if self.a_indices.ndim != 1:
            raise ValueError(
                "a_indices must be a one-dimensional array."
            )

        if self.k_bond_jax <= 0.0:
            raise ValueError(
                "k_bond_jax must be positive."
            )

        if self.r0_bond <= 0.0:
            raise ValueError(
                "r0_bond must be positive."
            )

        if self.association_cutoff <= 0.0:
            raise ValueError(
                "association_cutoff must be positive."
            )

    def find_association_candidates(
        self,
        R,
        disp_fn,
        *,
        system,
        cutoff: float | None = None,
    ):
        """Find free A-A pairs within the association cutoff.

        Only A particles that are not already part of an A2 dimer
        are considered. Each pair is returned only once.
        """

        if cutoff is None:
            cutoff = self.association_cutoff

        cutoff = float(cutoff)

        if cutoff <= 0.0:
            raise ValueError(
                "cutoff must be positive."
            )

        R_np = np.asarray(R)

        bond_idx = np.asarray(
            system.bonds[0],
            dtype=np.int32,
        )

        a_set = {
            int(i)
            for i in self.a_indices
        }

        # -----------------------------------------------------
        # Determine which A atoms are already part of dimers.
        # -----------------------------------------------------

        bonded_a = set()

        for i, j in bond_idx:
            i = int(i)
            j = int(j)

            if i in a_set and j in a_set:
                bonded_a.add(i)
                bonded_a.add(j)

        free_a = [
            int(i)
            for i in self.a_indices
            if int(i) not in bonded_a
        ]

        # -----------------------------------------------------
        # Search unique free A-A pairs.
        # -----------------------------------------------------

        candidates = []

        for n, i in enumerate(free_a):
            for j in free_a[n + 1:]:

                dr = disp_fn(
                    R_np[i],
                    R_np[j],
                )

                distance = float(
                    np.linalg.norm(
                        np.asarray(dr)
                    )
                )

                if distance <= cutoff:
                    candidates.append(
                        A2AssociationCandidate(
                            i=i,
                            j=j,
                            distance=distance,
                        )
                    )

        return candidates

    def find_dissociation_candidates(
        self,
        R,
        disp_fn,
        *,
        system,
    ):
        """Return all existing A2 dimers as dissociation candidates."""

        R_np = np.asarray(R)

        bond_idx = np.asarray(
            system.bonds[0],
            dtype=np.int32,
        )

        a_set = {
            int(i)
            for i in self.a_indices
        }

        candidates = []

        for bond_index, pair in enumerate(bond_idx):
            i = int(pair[0])
            j = int(pair[1])

            # Ignore bonds that are not A-A bonds.
            if i not in a_set or j not in a_set:
                continue

            dr = disp_fn(
                R_np[i],
                R_np[j],
            )

            distance = float(
                np.linalg.norm(
                    np.asarray(dr)
                )
            )

            candidates.append(
                A2DissociationCandidate(
                    i=i,
                    j=j,
                    bond_index=bond_index,
                    distance=distance,
                )
            )

        return candidates

    def build_association_trial(
        self,
        system,
        candidate: A2AssociationCandidate,
    ) -> dict:
        """Construct trial topology for A + A -> A2."""

        i = int(candidate.i)
        j = int(candidate.j)

        if i == j:
            raise ValueError(
                "Association candidate cannot contain the same atom twice."
            )

        a_set = {
            int(index)
            for index in self.a_indices
        }

        if i not in a_set or j not in a_set:
            raise ValueError(
                "Association candidate contains an atom "
                "that is not an A particle."
            )

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

        # -----------------------------------------------------
        # Reject duplicate bonds.
        # -----------------------------------------------------

        if bond_idx.shape[0] > 0:
            same_direction = np.all(
                bond_idx
                == np.array(
                    [i, j],
                    dtype=np.int32,
                ),
                axis=1,
            )

            reverse_direction = np.all(
                bond_idx
                == np.array(
                    [j, i],
                    dtype=np.int32,
                ),
                axis=1,
            )

            if np.any(
                same_direction
                | reverse_direction
            ):
                raise ValueError(
                    f"Atoms {i} and {j} are already bonded."
                )

        # -----------------------------------------------------
        # Prevent higher A clusters.
        #
        # Each A atom may belong to at most one A2 dimer.
        # -----------------------------------------------------

        bonded_a = set()

        for atom_i, atom_j in bond_idx:
            atom_i = int(atom_i)
            atom_j = int(atom_j)

            if atom_i in a_set and atom_j in a_set:
                bonded_a.add(atom_i)
                bonded_a.add(atom_j)

        if i in bonded_a or j in bonded_a:
            raise ValueError(
                "Association candidate contains an A particle "
                "that is already part of an A2 dimer."
            )

        # -----------------------------------------------------
        # Add the new harmonic bond.
        # -----------------------------------------------------

        new_bond_idx = np.concatenate(
            [
                bond_idx,
                np.array(
                    [[i, j]],
                    dtype=np.int32,
                ),
            ],
            axis=0,
        )

        new_bond_k = np.concatenate(
            [
                bond_k,
                np.array(
                    [self.k_bond_jax],
                    dtype=bond_k.dtype,
                ),
            ],
            axis=0,
        )

        new_bond_r0 = np.concatenate(
            [
                bond_r0,
                np.array(
                    [self.r0_bond],
                    dtype=bond_r0.dtype,
                ),
            ],
            axis=0,
        )

        return self._copy_system_with_bonds(
            system,
            bonds=(
                new_bond_idx,
                new_bond_k,
                new_bond_r0,
            ),
        )

    def build_dissociation_trial(
        self,
        system,
        candidate: A2DissociationCandidate,
    ) -> dict:
        """Construct trial topology for A2 -> A + A."""

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

        bond_index = int(
            candidate.bond_index
        )

        if (
            bond_index < 0
            or bond_index >= bond_idx.shape[0]
        ):
            raise IndexError(
                f"bond_index={bond_index} is outside "
                f"the valid range "
                f"[0, {bond_idx.shape[0] - 1}]."
            )

        expected_pair = {
            int(candidate.i),
            int(candidate.j),
        }

        actual_pair = {
            int(
                bond_idx[
                    bond_index,
                    0,
                ]
            ),
            int(
                bond_idx[
                    bond_index,
                    1,
                ]
            ),
        }

        if actual_pair != expected_pair:
            raise ValueError(
                "Dissociation candidate does not match "
                "the bond at candidate.bond_index."
            )

        keep = np.ones(
            bond_idx.shape[0],
            dtype=bool,
        )

        keep[bond_index] = False

        return self._copy_system_with_bonds(
            system,
            bonds=(
                bond_idx[keep],
                bond_k[keep],
                bond_r0[keep],
            ),
        )

    @staticmethod
    def _copy_system_with_bonds(
        system,
        *,
        bonds,
    ) -> dict:
        """Copy a SystemState-like object into a trial dictionary."""

        return {
            "bonds": (
                np.asarray(
                    bonds[0],
                    dtype=np.int32,
                ),
                np.asarray(
                    bonds[1]
                ),
                np.asarray(
                    bonds[2]
                ),
            ),
            "angles": tuple(
                np.asarray(value)
                for value in system.angles
            ),
            "torsions": tuple(
                np.asarray(value)
                for value in system.torsions
            ),
            "impropers": tuple(
                np.asarray(value)
                for value in system.impropers
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
