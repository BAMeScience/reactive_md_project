import math

import jax
import pytest

from reactive_md.reaction import accept_reject


def test_accept_reject_always_accepts_negative_energy_change():
    key = jax.random.PRNGKey(0)

    _, accepted, p_metropolis = accept_reject(
        key,
        dE=-1.0,
        beta=1.0,
    )

    assert accepted is True
    assert p_metropolis == pytest.approx(1.0)


def test_accept_reject_positive_energy_uses_metropolis_factor():
    key = jax.random.PRNGKey(0)

    _, _accepted, p_metropolis = accept_reject(
        key,
        dE=2.0,
        beta=0.5,
    )

    assert p_metropolis == pytest.approx(math.exp(-1.0))


def test_accept_reject_rejects_nonfinite_energy():
    key = jax.random.PRNGKey(0)

    _, accepted, p_metropolis = accept_reject(
        key,
        dE=float("nan"),
        beta=1.0,
    )

    assert accepted is False
    assert p_metropolis == pytest.approx(0.0)


def test_accept_reject_does_not_depend_on_sigma():
    key = jax.random.PRNGKey(123)

    _, accepted_a, p_a = accept_reject(
        key,
        dE=1.0,
        beta=2.0,
    )

    _, accepted_b, p_b = accept_reject(
        key,
        dE=1.0,
        beta=2.0,
    )

    assert p_a == pytest.approx(math.exp(-2.0))
    assert p_b == pytest.approx(math.exp(-2.0))
    assert accepted_a == accepted_b
