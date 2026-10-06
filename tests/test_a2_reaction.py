# tests/test_a2_reaction.py

import numpy as np
import jax.numpy as jnp

from reactive_md.reaction import SystemState
from reactive_md.reactions.a2 import (
    A2AssociationCandidate,
    A2DissociationCandidate,
    A2Reaction,
)


def make_empty_system(n_atoms=4):
    empty_bonds = (
        jnp.empty((0, 2), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
    )

    empty_angles = (
        jnp.empty((0, 3), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
    )

    empty_four_body = (
        jnp.empty((0, 4), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
    )

    return SystemState(
        bonds=empty_bonds,
        angles=empty_angles,
        torsions=empty_four_body,
        impropers=empty_four_body,
        charges=jnp.zeros((n_atoms,), dtype=jnp.float32),
        sigmas=jnp.ones((n_atoms,), dtype=jnp.float32),
        epsilons=jnp.ones((n_atoms,), dtype=jnp.float32),
        molecule_id=jnp.arange(n_atoms, dtype=jnp.int32),
        pf6_reacted=jnp.zeros((0,), dtype=jnp.bool_),
    )


def test_a2_association_adds_one_bond():
    system = make_empty_system()

    reaction = A2Reaction(
        a_indices=np.array([0, 1, 2], dtype=np.int32),
        k_bond_jax=29.8757,
        r0_bond=1.5,
    )

    candidate = A2AssociationCandidate(
        i=0,
        j=1,
        distance=2.0,
    )

    trial = reaction.build_association_trial(
        system,
        candidate,
    )

    bond_idx, bond_k, bond_r0 = trial["bonds"]

    assert bond_idx.shape == (1, 2)
    assert bond_k.shape == (1,)
    assert bond_r0.shape == (1,)

    np.testing.assert_array_equal(
        bond_idx,
        np.array([[0, 1]], dtype=np.int32),
    )

    np.testing.assert_allclose(
        bond_k,
        np.array([29.8757], dtype=np.float32),
    )

    np.testing.assert_allclose(
        bond_r0,
        np.array([1.5], dtype=np.float32),
    )


def test_a2_dissociation_removes_bond_again():
    system = make_empty_system()

    reaction = A2Reaction(
        a_indices=np.array([0, 1, 2], dtype=np.int32),
        k_bond_jax=29.8757,
        r0_bond=1.5,
    )

    association = A2AssociationCandidate(
        i=0,
        j=1,
        distance=2.0,
    )

    trial_assoc = reaction.build_association_trial(
        system,
        association,
    )

    system_with_bond = SystemState(
        bonds=tuple(
            jnp.asarray(x)
            for x in trial_assoc["bonds"]
        ),
        angles=tuple(
            jnp.asarray(x)
            for x in trial_assoc["angles"]
        ),
        torsions=tuple(
            jnp.asarray(x)
            for x in trial_assoc["torsions"]
        ),
        impropers=tuple(
            jnp.asarray(x)
            for x in trial_assoc["impropers"]
        ),
        charges=jnp.asarray(trial_assoc["charges"]),
        sigmas=jnp.asarray(trial_assoc["sigmas"]),
        epsilons=jnp.asarray(trial_assoc["epsilons"]),
        molecule_id=jnp.asarray(
            trial_assoc["molecule_id"],
            dtype=jnp.int32,
        ),
        pf6_reacted=jnp.zeros((0,), dtype=jnp.bool_),
    )

    dissociation = A2DissociationCandidate(
        i=0,
        j=1,
        bond_index=0,
        distance=1.5,
    )

    trial_dissoc = reaction.build_dissociation_trial(
        system_with_bond,
        dissociation,
    )

    bond_idx, bond_k, bond_r0 = trial_dissoc["bonds"]

    assert bond_idx.shape == (0, 2)
    assert bond_k.shape == (0,)
    assert bond_r0.shape == (0,)


def test_a2_trial_preserves_unrelated_system_data():
    system = make_empty_system()

    reaction = A2Reaction(
        a_indices=np.array([0, 1, 2], dtype=np.int32),
        k_bond_jax=29.8757,
        r0_bond=1.5,
    )

    candidate = A2AssociationCandidate(
        i=0,
        j=1,
        distance=2.0,
    )

    trial = reaction.build_association_trial(
        system,
        candidate,
    )

    np.testing.assert_allclose(
        trial["charges"],
        np.asarray(system.charges),
    )

    np.testing.assert_allclose(
        trial["sigmas"],
        np.asarray(system.sigmas),
    )

    np.testing.assert_allclose(
        trial["epsilons"],
        np.asarray(system.epsilons),
    )

    np.testing.assert_array_equal(
        trial["molecule_id"],
        np.asarray(system.molecule_id),
    )


def test_a2_association_rejects_existing_bond():
    system = make_empty_system()

    reaction = A2Reaction(
        a_indices=np.array([0, 1, 2], dtype=np.int32),
        k_bond_jax=29.8757,
        r0_bond=1.5,
    )

    candidate = A2AssociationCandidate(
        i=0,
        j=1,
        distance=2.0,
    )

    trial = reaction.build_association_trial(
        system,
        candidate,
    )

    system_with_bond = SystemState(
        bonds=tuple(
            jnp.asarray(x)
            for x in trial["bonds"]
        ),
        angles=tuple(
            jnp.asarray(x)
            for x in trial["angles"]
        ),
        torsions=tuple(
            jnp.asarray(x)
            for x in trial["torsions"]
        ),
        impropers=tuple(
            jnp.asarray(x)
            for x in trial["impropers"]
        ),
        charges=jnp.asarray(trial["charges"]),
        sigmas=jnp.asarray(trial["sigmas"]),
        epsilons=jnp.asarray(trial["epsilons"]),
        molecule_id=jnp.asarray(
            trial["molecule_id"],
            dtype=jnp.int32,
        ),
        pf6_reacted=jnp.zeros((0,), dtype=jnp.bool_),
    )

    try:
        reaction.build_association_trial(
            system_with_bond,
            candidate,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected duplicate A-A association to raise ValueError."
        )

def test_find_association_candidates_finds_close_free_a_pair():
    system = make_empty_system(n_atoms=4)

    reaction = A2Reaction(
        a_indices=np.array([0, 1, 2], dtype=np.int32),
        k_bond_jax=29.8757,
        r0_bond=1.5,
    )

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],  # A0
            [2.5, 0.0, 0.0],  # A1 -> within 2.7 A
            [6.0, 0.0, 0.0],  # A2 -> too far
            [0.0, 6.0, 0.0],  # B
        ],
        dtype=jnp.float32,
    )

    def disp_fn(a, b):
        return b - a

    candidates = reaction.find_association_candidates(
        R,
        disp_fn,
        system=system,
        cutoff=2.7,
    )

    assert len(candidates) == 1

    cand = candidates[0]

    assert cand.i == 0
    assert cand.j == 1

    np.testing.assert_allclose(
        cand.distance,
        2.5,
    )

def test_find_association_candidates_finds_close_free_a_pair():
    system = make_empty_system(n_atoms=4)

    reaction = A2Reaction(
        a_indices=np.array([0, 1, 2], dtype=np.int32),
        k_bond_jax=29.8757,
        r0_bond=1.5,
    )

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],  # A0
            [2.5, 0.0, 0.0],  # A1 -> within 2.7 A
            [6.0, 0.0, 0.0],  # A2 -> too far
            [0.0, 6.0, 0.0],  # B
        ],
        dtype=jnp.float32,
    )

    def disp_fn(a, b):
        return b - a

    candidates = reaction.find_association_candidates(
        R,
        disp_fn,
        system=system,
        cutoff=2.7,
    )

    assert len(candidates) == 1

    cand = candidates[0]

    assert cand.i == 0
    assert cand.j == 1

    np.testing.assert_allclose(
        cand.distance,
        2.5,
    )

def test_find_association_candidates_excludes_bonded_a_atoms():
    system = make_empty_system(n_atoms=4)

    reaction = A2Reaction(
        a_indices=np.array([0, 1, 2], dtype=np.int32),
        k_bond_jax=29.8757,
        r0_bond=1.5,
    )

    assoc = A2AssociationCandidate(
        i=0,
        j=1,
        distance=1.5,
    )

    trial = reaction.build_association_trial(
        system,
        assoc,
    )

    system_with_bond = SystemState(
        bonds=tuple(
            jnp.asarray(x)
            for x in trial["bonds"]
        ),
        angles=tuple(
            jnp.asarray(x)
            for x in trial["angles"]
        ),
        torsions=tuple(
            jnp.asarray(x)
            for x in trial["torsions"]
        ),
        impropers=tuple(
            jnp.asarray(x)
            for x in trial["impropers"]
        ),
        charges=jnp.asarray(trial["charges"]),
        sigmas=jnp.asarray(trial["sigmas"]),
        epsilons=jnp.asarray(trial["epsilons"]),
        molecule_id=jnp.asarray(
            trial["molecule_id"],
            dtype=jnp.int32,
        ),
        pf6_reacted=jnp.zeros(
            (0,),
            dtype=jnp.bool_,
        ),
    )

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],  # bonded A
            [1.5, 0.0, 0.0],  # bonded A
            [2.0, 0.0, 0.0],  # free A
            [8.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    def disp_fn(a, b):
        return b - a

    candidates = reaction.find_association_candidates(
        R,
        disp_fn,
        system=system_with_bond,
        cutoff=2.7,
    )

    # Atom 2 is free, but atoms 0 and 1 are already bonded,
    # so there is no possible free A-A pair.
    assert candidates == []

def test_find_dissociation_candidates_returns_existing_a2_bond():
    system = make_empty_system(n_atoms=4)

    reaction = A2Reaction(
        a_indices=np.array([0, 1, 2], dtype=np.int32),
        k_bond_jax=29.8757,
        r0_bond=1.5,
    )

    assoc = A2AssociationCandidate(
        i=0,
        j=1,
        distance=1.5,
    )

    trial = reaction.build_association_trial(
        system,
        assoc,
    )

    system_with_bond = SystemState(
        bonds=tuple(
            jnp.asarray(x)
            for x in trial["bonds"]
        ),
        angles=tuple(
            jnp.asarray(x)
            for x in trial["angles"]
        ),
        torsions=tuple(
            jnp.asarray(x)
            for x in trial["torsions"]
        ),
        impropers=tuple(
            jnp.asarray(x)
            for x in trial["impropers"]
        ),
        charges=jnp.asarray(trial["charges"]),
        sigmas=jnp.asarray(trial["sigmas"]),
        epsilons=jnp.asarray(trial["epsilons"]),
        molecule_id=jnp.asarray(
            trial["molecule_id"],
            dtype=jnp.int32,
        ),
        pf6_reacted=jnp.zeros(
            (0,),
            dtype=jnp.bool_,
        ),
    )

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    def disp_fn(a, b):
        return b - a

    candidates = reaction.find_dissociation_candidates(
        R,
        disp_fn,
        system=system_with_bond,
    )

    assert len(candidates) == 1

    cand = candidates[0]

    assert cand.i == 0
    assert cand.j == 1
    assert cand.bond_index == 0

    np.testing.assert_allclose(
        cand.distance,
        1.6,
    )

def test_find_association_candidates_returns_unique_pairs():
    system = make_empty_system(n_atoms=3)

    reaction = A2Reaction(
        a_indices=np.array([0, 1, 2], dtype=np.int32),
        k_bond_jax=29.8757,
        r0_bond=1.5,
    )

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    def disp_fn(a, b):
        return b - a

    candidates = reaction.find_association_candidates(
        R,
        disp_fn,
        system=system,
        cutoff=2.7,
    )

    pairs = {
        tuple(sorted((cand.i, cand.j)))
        for cand in candidates
    }

    assert len(pairs) == len(candidates)
