"""Unit tests for the sigma-only LiPF6 reaction logic.

These tests intentionally focus on small, deterministic pieces of reaction.py.
They do not require a full MD simulation or an HPC environment.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.reaction import (
    accept_reject,
    candidate_records_from_sigma_candidates,
    find_reaction_candidates,
    make_probe_geometry,
    reaction_coordinate,
    resolve_rate_ps,
)

# The current develop branch may still expose these names.  The tests support
# either the older sigma_* names or the cleaner reaction_* names.
import reactive_md.reaction as reaction_module


if hasattr(reaction_module, "reaction_probability"):
    reaction_probability = reaction_module.reaction_probability
else:
    reaction_probability = reaction_module.sigma_gate_factor


if hasattr(reaction_module, "rate_probability_from_reaction_coordinate"):
    rate_probability_from_reaction_coordinate = (
        reaction_module.rate_probability_from_reaction_coordinate
    )
else:
    rate_probability_from_reaction_coordinate = (
        reaction_module.reaction_probability_from_sigma
    )


def _disp(a, b):
    """Non-periodic displacement compatible with the reaction helper functions."""
    return b - a


def _shift(r, dr):
    """Non-periodic shift compatible with JAX-MD-style shift functions."""
    return r + dr


def _sigma_of_candidate(cand):
    return reaction_coordinate(d_pf=cand.d_pf, d_lif=cand.d_lif)


def test_reaction_coordinate_sign_convention():
    assert reaction_coordinate(d_pf=1.6, d_lif=3.0) == pytest.approx(-1.4)
    assert reaction_coordinate(d_pf=2.0, d_lif=2.0) == pytest.approx(0.0)
    assert reaction_coordinate(d_pf=3.0, d_lif=1.6) == pytest.approx(1.4)


def test_reaction_probability_midpoint_and_monotonicity():
    p_left = reaction_probability(sigma=-1.0, midpoint=0.0, width=0.2)
    p_mid = reaction_probability(sigma=0.0, midpoint=0.0, width=0.2)
    p_right = reaction_probability(sigma=1.0, midpoint=0.0, width=0.2)

    assert p_mid == pytest.approx(0.5)
    assert 0.0 < p_left < p_mid < p_right < 1.0


def test_reaction_probability_rejects_nonpositive_width():
    with pytest.raises(ValueError):
        reaction_probability(sigma=0.0, midpoint=0.0, width=0.0)

    with pytest.raises(ValueError):
        reaction_probability(sigma=0.0, midpoint=0.0, width=-0.1)


def test_rate_probability_uses_sigma_factor_only():
    p_react, k_eff, sigma_factor = rate_probability_from_reaction_coordinate(
        sigma=0.0,
        base_rate_ps=2.0,
        reactive_interval_ps=0.5,
        midpoint=0.0,
        width=0.2,
    )

    assert sigma_factor == pytest.approx(0.5)
    assert k_eff == pytest.approx(1.0)
    assert p_react == pytest.approx(1.0 - math.exp(-0.5))


def test_resolve_rate_exclusive_inputs():
    with pytest.raises(ValueError):
        resolve_rate_ps(
            reaction_rate_ps=1.0,
            activation_energy_eV=0.2,
            temperature_k=400.0,
        )


def test_resolve_rate_direct_rate_and_zero_default():
    assert resolve_rate_ps(
        reaction_rate_ps=0.123,
        activation_energy_eV=None,
        temperature_k=400.0,
    ) == pytest.approx(0.123)

    assert resolve_rate_ps(
        reaction_rate_ps=None,
        activation_energy_eV=None,
        temperature_k=400.0,
    ) == pytest.approx(0.0)


def test_find_reaction_candidates_ranks_by_sigma_only():
    # Atom layout:
    # P  = atom 0 at x=0
    # F1 = atom 1 at x=1, Li at x=2 -> d_pf=1, d_lif=1, sigma=0
    # F2 = atom 2 at x=3, Li at x=2 -> d_pf=3, d_lif=1, sigma=2
    # A distance-gated algorithm might reject F2 for being far from P, but the
    # sigma-only algorithm should rank it first because sigma is larger.
    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    pf6_atoms = np.array([[0, 1, 2]], dtype=np.int32)
    li_atoms = np.array([3], dtype=np.int32)
    pf6_reacted = np.array([False], dtype=bool)

    candidates = find_reaction_candidates(
        R,
        pf6_atoms,
        li_atoms,
        _disp,
        pf6_reacted_np=pf6_reacted,
    )

    assert len(candidates) == 2
    assert candidates[0].leave_F == 2
    assert _sigma_of_candidate(candidates[0]) == pytest.approx(2.0)
    assert candidates[1].leave_F == 1
    assert _sigma_of_candidate(candidates[1]) == pytest.approx(0.0)


def test_find_reaction_candidates_skips_reacted_pf6():
    R = jnp.zeros((4, 3))
    pf6_atoms = np.array([[0, 1, 2]], dtype=np.int32)
    li_atoms = np.array([3], dtype=np.int32)
    pf6_reacted = np.array([True], dtype=bool)

    candidates = find_reaction_candidates(
        R,
        pf6_atoms,
        li_atoms,
        _disp,
        pf6_reacted_np=pf6_reacted,
    )

    assert candidates == []


def test_candidate_records_contain_sigma_not_legacy_gate_flags():
    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    pf6_atoms = np.array([[0, 1, 2]], dtype=np.int32)
    li_atoms = np.array([3], dtype=np.int32)
    candidates = find_reaction_candidates(
        R,
        pf6_atoms,
        li_atoms,
        _disp,
        pf6_reacted_np=np.array([False], dtype=bool),
    )

    records = candidate_records_from_sigma_candidates(candidates, top_n=1)

    assert len(records) == 1
    assert records[0]["sigma"] == pytest.approx(2.0)
    assert "passes_lif" not in records[0]
    assert "passes_pf" not in records[0]
    assert "passes_all" not in records[0]


def test_make_probe_geometry_moves_leaving_f_away_from_p():
    R = jnp.array(
        [
            [0.0, 0.0, 0.0],  # P
            [1.0, 0.0, 0.0],  # leaving F
        ]
    )

    R_probe = make_probe_geometry(
        R,
        P_atom=0,
        leave_F=1,
        disp_fn=_disp,
        shift_fn=_shift,
        r_pf_probe=4.0,
    )

    assert np.asarray(R_probe[0]).tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert np.asarray(R_probe[1]).tolist() == pytest.approx([4.0, 0.0, 0.0])


def test_accept_reject_probability_bounds_and_downhill_acceptance():
    key = jax.random.PRNGKey(0)

    _, accepted_downhill, p_downhill = accept_reject(key, dE=-10.0, beta=1.0)
    assert accepted_downhill is True
    assert p_downhill == pytest.approx(1.0)

    _, _, p_uphill = accept_reject(key, dE=1.0, beta=1.0)
    assert 0.0 < p_uphill < 1.0

