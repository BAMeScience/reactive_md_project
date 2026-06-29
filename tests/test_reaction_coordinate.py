import math

import pytest

from reactive_md.reaction import (
    reaction_coordinate,
    reaction_probability,
    rate_probability_from_reaction_coordinate,
)


def test_reaction_coordinate_sign_convention():
    assert reaction_coordinate(d_pf=1.6, d_lif=2.4) == pytest.approx(-0.8)
    assert reaction_coordinate(d_pf=2.4, d_lif=1.6) == pytest.approx(0.8)


def test_reaction_coordinate_transition_region():
    assert reaction_coordinate(d_pf=2.0, d_lif=2.0) == pytest.approx(0.0)


def test_reaction_probability_midpoint_is_half():
    assert reaction_probability(0.0, midpoint=0.0, width=0.2) == pytest.approx(0.5)


def test_reaction_probability_monotonic_in_sigma():
    p_low = reaction_probability(-1.0, midpoint=0.0, width=0.2)
    p_mid = reaction_probability(0.0, midpoint=0.0, width=0.2)
    p_high = reaction_probability(1.0, midpoint=0.0, width=0.2)

    assert 0.0 < p_low < p_mid < p_high < 1.0


def test_reaction_probability_rejects_nonpositive_width():
    with pytest.raises(ValueError):
        reaction_probability(0.0, width=0.0)
    with pytest.raises(ValueError):
        reaction_probability(0.0, width=-1.0)


def test_rate_probability_from_reaction_coordinate():
    p_rate, k_eff, sigma_factor = rate_probability_from_reaction_coordinate(
        sigma=0.0,
        base_rate_ps=1.0,
        reactive_interval_ps=1.0,
        midpoint=0.0,
        width=0.2,
    )

    assert p_rate == pytest.approx(1.0 - math.exp(-0.5))
    assert k_eff == pytest.approx(0.5)
    assert sigma_factor == pytest.approx(0.5)


def test_rate_probability_is_zero_for_zero_base_rate():
    p_rate, k_eff, sigma_factor = rate_probability_from_reaction_coordinate(
        sigma=10.0,
        base_rate_ps=0.0,
        reactive_interval_ps=1.0,
        midpoint=0.0,
        width=0.2,
    )

    assert p_rate == pytest.approx(0.0)
    assert k_eff == pytest.approx(0.0)
    assert sigma_factor == pytest.approx(1.0)

