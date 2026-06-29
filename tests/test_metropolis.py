import math

import jax
import pytest

from reactive_md.reaction import accept_reject


def test_accept_reject_combines_sigma_and_metropolis_factors():
    key = jax.random.PRNGKey(0)

    _, _accepted, p_total, p_sigma, p_metropolis = accept_reject(
        key,
        sigma=0.0,
        dE=-1.0,
        beta=1.0,
        sigma_mid=0.0,
        sigma_width=0.2,
    )

    assert p_sigma == pytest.approx(0.5)
    assert p_metropolis == pytest.approx(1.0)
    assert p_total == pytest.approx(0.5)


def test_accept_reject_positive_energy_uses_metropolis_factor():
    key = jax.random.PRNGKey(0)

    _, _accepted, p_total, p_sigma, p_metropolis = accept_reject(
        key,
        sigma=10.0,
        dE=2.0,
        beta=0.5,
        sigma_mid=0.0,
        sigma_width=0.2,
    )

    assert p_sigma == pytest.approx(1.0)
    assert p_metropolis == pytest.approx(math.exp(-1.0))
    assert p_total == pytest.approx(math.exp(-1.0))


def test_accept_reject_rejects_nonfinite_energy():
    key = jax.random.PRNGKey(0)

    _, accepted, p_total, p_sigma, p_metropolis = accept_reject(
        key,
        sigma=10.0,
        dE=float("nan"),
        beta=1.0,
        sigma_mid=0.0,
        sigma_width=0.2,
    )

    assert accepted is False
    assert p_sigma == pytest.approx(1.0)
    assert p_metropolis == pytest.approx(0.0)
    assert p_total == pytest.approx(0.0)


def test_accept_reject_sigma_can_suppress_downhill_reaction():
    key = jax.random.PRNGKey(0)

    _, _accepted, p_total, p_sigma, p_metropolis = accept_reject(
        key,
        sigma=-10.0,
        dE=-100.0,
        beta=1.0,
        sigma_mid=0.0,
        sigma_width=0.2,
    )

    assert p_sigma == pytest.approx(0.0)
    assert p_metropolis == pytest.approx(1.0)
    assert p_total == pytest.approx(0.0)

