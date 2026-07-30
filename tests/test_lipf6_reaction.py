import numpy as np
import pytest
import jax.numpy as jnp

from reactive_md.reaction import SystemState
from reactive_md.reactions.lipf6 import (
    LiPF6Reaction,
    ReactionCandidate,
    reaction_coordinate,
)
from reactive_md.reactions.templates_pf5 import (
    make_lif_template,
    make_pf5_template,
)


P_TYPE = 6
F_TYPE = 7
LI_TYPE = 8


def make_lipf6_case():
    """Create the same minimal PF6/Li system used by the reaction tests."""
    atom_types = np.array(
        [P_TYPE, F_TYPE, F_TYPE, F_TYPE, F_TYPE, F_TYPE, F_TYPE, LI_TYPE],
        dtype=np.int32,
    )
    molecule_id = np.array([1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32)

    # One PF6-internal bond, which must be removed, and one cross-molecule
    # bond, which must remain.
    bond_idx = np.array([[0, 1], [0, 7]], dtype=np.int32)
    k_b = np.array([100.0, 200.0], dtype=np.float32)
    r0 = np.array([1.6, 2.0], dtype=np.float32)

    angle_idx = np.zeros((0, 3), dtype=np.int32)
    k_theta = np.zeros((0,), dtype=np.float32)
    theta0 = np.zeros((0,), dtype=np.float32)

    tors_idx = np.zeros((0, 4), dtype=np.int32)
    tors_k = np.zeros((0,), dtype=np.float32)
    tors_n = np.zeros((0,), dtype=np.int32)
    tors_gamma = np.zeros((0,), dtype=np.float32)

    impr_idx = np.zeros((0, 4), dtype=np.int32)
    impr_k = np.zeros((0,), dtype=np.float32)
    impr_n = np.zeros((0,), dtype=np.int32)
    impr_gamma = np.zeros((0,), dtype=np.float32)

    system = SystemState(
        bonds=(bond_idx, k_b, r0),
        angles=(angle_idx, k_theta, theta0),
        torsions=(tors_idx, tors_k, tors_n, tors_gamma),
        impropers=(impr_idx, impr_k, impr_n, impr_gamma),
        charges=np.zeros((8,), dtype=np.float32),
        sigmas=np.full((8,), 3.0, dtype=np.float32),
        epsilons=np.full((8,), 0.1, dtype=np.float32),
        molecule_id=molecule_id,
        pf6_reacted=jnp.array([False], dtype=jnp.bool_),
    )

    reaction = LiPF6Reaction.from_system(
        atom_types,
        molecule_id,
        pf5=make_pf5_template(),
        lif=make_lif_template(),
        p_type=P_TYPE,
        f_type=F_TYPE,
        li_type=LI_TYPE,
    )

    candidate = ReactionCandidate(
        k_pf6=0,
        li_idx=7,
        leave_F=6,
        d_lif=1.5,
        d_pf=2.0,
    )

    return system, reaction, candidate


def test_from_system_discovers_pf6_and_li():
    _, reaction, _ = make_lipf6_case()

    np.testing.assert_array_equal(
        reaction.pf6_atoms,
        np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        reaction.li_atoms,
        np.array([7], dtype=np.int32),
    )


def test_reaction_coordinate_is_pf_minus_lif_distance():
    assert reaction_coordinate(d_pf=2.0, d_lif=1.5) == pytest.approx(0.5)


def test_phosphorus_index_and_candidate_type_validation():
    _, reaction, candidate = make_lipf6_case()

    assert reaction.phosphorus_index(candidate) == 0
    assert reaction.candidate_types_are_valid(candidate)


def test_candidate_type_validation_detects_wrong_atom_type():
    system, reaction, candidate = make_lipf6_case()
    del system

    reaction.atom_types = reaction.atom_types.copy()
    reaction.atom_types[candidate.leave_F] = P_TYPE

    assert not reaction.candidate_types_are_valid(candidate)


def test_invalid_candidate_indices_are_rejected():
    _, reaction, candidate = make_lipf6_case()

    with pytest.raises(IndexError):
        reaction.phosphorus_index(
            ReactionCandidate(
                k_pf6=1,
                li_idx=candidate.li_idx,
                leave_F=candidate.leave_F,
                d_lif=candidate.d_lif,
                d_pf=candidate.d_pf,
            )
        )

    with pytest.raises(ValueError):
        reaction.phosphorus_index(
            ReactionCandidate(
                k_pf6=0,
                li_idx=candidate.li_idx,
                leave_F=candidate.li_idx,
                d_lif=candidate.d_lif,
                d_pf=candidate.d_pf,
            )
        )

    with pytest.raises(ValueError):
        reaction.phosphorus_index(
            ReactionCandidate(
                k_pf6=0,
                li_idx=0,
                leave_F=candidate.leave_F,
                d_lif=candidate.d_lif,
                d_pf=candidate.d_pf,
            )
        )


def test_embed_pf5_excludes_leaving_fluorine():
    _, reaction, candidate = make_lipf6_case()

    pf5_global, pf5_bonds, pf5_angles = reaction.embed_pf5(candidate)

    np.testing.assert_array_equal(
        pf5_global,
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
    )
    assert candidate.leave_F not in pf5_global
    assert pf5_bonds.shape == reaction.pf5.bond_idx_local.shape
    assert pf5_angles.shape == reaction.pf5.angle_idx_local.shape
    np.testing.assert_array_equal(pf5_bonds, pf5_global[reaction.pf5.bond_idx_local])
    np.testing.assert_array_equal(pf5_angles, pf5_global[reaction.pf5.angle_idx_local])


def test_find_candidates_uses_owned_pf6_and_li_data():
    _, reaction, _ = make_lipf6_case()

    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=jnp.float32,
    )

    def displacement(a, b):
        return b - a

    candidates = reaction.find_candidates(
        positions,
        displacement,
        pf6_reacted=np.array([False]),
    )

    assert len(candidates) == 6
    assert all(candidate.k_pf6 == 0 for candidate in candidates)
    assert all(candidate.li_idx == 7 for candidate in candidates)
    assert {candidate.leave_F for candidate in candidates} == {1, 2, 3, 4, 5, 6}

    sigmas = [
        reaction_coordinate(d_pf=candidate.d_pf, d_lif=candidate.d_lif)
        for candidate in candidates
    ]
    assert sigmas == sorted(sigmas, reverse=True)
    assert candidates[0].leave_F == 6

    assert reaction.find_candidates(
        positions,
        displacement,
        pf6_reacted=np.array([True]),
    ) == []


def test_prepare_probe_moves_only_leaving_fluorine():
    _, reaction, candidate = make_lipf6_case()

    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    def displacement(a, b):
        return b - a

    def shift(position, delta):
        return position + delta

    sigma_p = 2.0
    sigma_f = 2.0
    expected_distance = 2.0 ** (1.0 / 6.0) * 0.5 * (sigma_p + sigma_f)

    probe = reaction.prepare_probe(
        positions,
        candidate,
        sigma_p=sigma_p,
        sigma_f=sigma_f,
        disp_fn=displacement,
        shift_fn=shift,
    )

    unchanged = [index for index in range(positions.shape[0]) if index != candidate.leave_F]
    np.testing.assert_allclose(np.asarray(probe)[unchanged], np.asarray(positions)[unchanged])
    np.testing.assert_allclose(
        np.asarray(probe[candidate.leave_F]),
        np.array([expected_distance, 0.0, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_trial_updates_counts_parameters_and_molecule_ids():
    system, reaction, candidate = make_lipf6_case()

    trial, pf6_molid = reaction.build_trial(system, candidate)

    assert trial is not None
    assert pf6_molid == 1

    # The PF6-internal bond is removed, the cross-molecule bond remains,
    # and the five PF5 bonds are inserted.
    assert trial["bonds"][0].shape[0] == 6
    assert any(np.array_equal(bond, np.array([0, 7])) for bond in trial["bonds"][0])

    pf5_global, _, _ = reaction.embed_pf5(candidate)
    product_phosphorus = int(pf5_global[0])
    product_fluorines = pf5_global[1:]

    assert trial["charges"][product_phosphorus] == pytest.approx(reaction.pf5.q["P"])
    np.testing.assert_allclose(
        trial["charges"][product_fluorines],
        reaction.pf5.q["F"],
    )
    assert trial["epsilons"][product_phosphorus] == pytest.approx(
        reaction.pf5.pair["P"][0]
    )
    assert trial["sigmas"][product_phosphorus] == pytest.approx(
        reaction.pf5.pair["P"][1]
    )

    leave_f = candidate.leave_F
    lithium = candidate.li_idx

    assert trial["charges"][leave_f] == pytest.approx(reaction.lif.nb["F"]["q"])
    assert trial["sigmas"][leave_f] == pytest.approx(reaction.lif.nb["F"]["sigma"])
    assert trial["epsilons"][leave_f] == pytest.approx(reaction.lif.nb["F"]["eps"])

    assert trial["charges"][lithium] == pytest.approx(reaction.lif.nb["Li"]["q"])
    assert trial["sigmas"][lithium] == pytest.approx(reaction.lif.nb["Li"]["sigma"])
    assert trial["epsilons"][lithium] == pytest.approx(reaction.lif.nb["Li"]["eps"])

    assert trial["molecule_id"][leave_f] == np.max(system.molecule_id) + 1
    assert trial["molecule_id"][lithium] == system.molecule_id[lithium]

    # Trial construction must not mutate the input system arrays.
    np.testing.assert_array_equal(
        np.asarray(system.molecule_id),
        np.array([1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32),
    )
    np.testing.assert_allclose(np.asarray(system.charges), 0.0)


def test_build_trial_returns_none_for_wrong_candidate_types():
    system, reaction, candidate = make_lipf6_case()

    reaction.atom_types = reaction.atom_types.copy()
    reaction.atom_types[candidate.li_idx] = F_TYPE

    trial, pf6_molid = reaction.build_trial(system, candidate)

    assert trial is None
    assert pf6_molid is None

