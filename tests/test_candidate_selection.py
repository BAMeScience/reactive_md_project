import numpy as np
import jax.numpy as jnp

from reactive_md.reaction import (
    ReactionCandidate,
    candidate_records_from_reaction_candidates,
    find_reaction_candidates,
    reaction_coordinate,
)


def _disp(a, b):
    return b - a


def test_reaction_candidate_dataclass_fields():
    cand = ReactionCandidate(
        k_pf6=0,
        li_idx=7,
        leave_F=1,
        d_lif=1.2,
        d_pf=2.0,
    )

    assert cand.k_pf6 == 0
    assert cand.li_idx == 7
    assert cand.leave_F == 1
    assert cand.d_lif == 1.2
    assert cand.d_pf == 2.0
    assert reaction_coordinate(cand.d_pf, cand.d_lif) == 0.8


def test_find_reaction_candidates_ranks_by_descending_sigma():
    # Atom layout:
    # 0 P
    # 1 F close to Li and farther from P -> high sigma
    # 2 F close to P and farther from Li -> low sigma
    # 3..6 additional PF6 fluorines
    # 7 Li
    R = jnp.array(
        [
            [0.0, 0.0, 0.0],   # P
            [2.0, 0.0, 0.0],   # F1: d_pf=2.0, d_lif=0.5 if Li at 2.5
            [1.0, 0.0, 0.0],   # F2: d_pf=1.0, d_lif=1.5
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [2.5, 0.0, 0.0],   # Li
        ],
        dtype=jnp.float32,
    )

    pf6_atoms = np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.int32)
    li_atoms = np.array([7], dtype=np.int32)
    pf6_reacted = np.array([False], dtype=bool)

    candidates = find_reaction_candidates(
        R,
        pf6_atoms,
        li_atoms,
        _disp,
        pf6_reacted_np=pf6_reacted,
    )

    assert len(candidates) == 6

    sigmas = [
        reaction_coordinate(d_pf=c.d_pf, d_lif=c.d_lif)
        for c in candidates
    ]

    assert sigmas == sorted(sigmas, reverse=True)
    assert candidates[0].leave_F == 1


def test_find_reaction_candidates_skips_reacted_pf6():
    R = jnp.zeros((8, 3), dtype=jnp.float32)
    pf6_atoms = np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.int32)
    li_atoms = np.array([7], dtype=np.int32)
    pf6_reacted = np.array([True], dtype=bool)

    candidates = find_reaction_candidates(
        R,
        pf6_atoms,
        li_atoms,
        _disp,
        pf6_reacted_np=pf6_reacted,
    )

    assert candidates == []


def test_candidate_records_contain_sigma_not_old_gate_columns():
    candidates = [
        ReactionCandidate(k_pf6=0, li_idx=7, leave_F=1, d_lif=1.0, d_pf=2.0),
        ReactionCandidate(k_pf6=0, li_idx=7, leave_F=2, d_lif=2.0, d_pf=1.0),
    ]

    records = candidate_records_from_reaction_candidates(candidates, top_n=10)

    assert len(records) == 2
    assert records[0]["rank"] == 0
    assert records[0]["sigma"] == 1.0
    assert records[1]["sigma"] == -1.0

    for rec in records:
        assert "passes_lif" not in rec
        assert "passes_pf" not in rec
        assert "passes_all" not in rec
        assert "q_pf_minus_lif" not in rec

