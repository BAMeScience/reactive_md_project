import math

import jax
import pytest

from reactive_md.reaction import accept_reject


def test_accept_reject_always_accepts_negative_energy_change():
    key = jax.random.PRNGKey(0)

    _, accepted, p_acc = accept_reject(key, dE=-1.0, beta=1.0)

    assert accepted is True
    assert p_acc == pytest.approx(1.0)


def test_accept_reject_probability_for_positive_energy_change():
    key = jax.random.PRNGKey(0)

    _, _accepted, p_acc = accept_reject(key, dE=2.0, beta=0.5)

    assert p_acc == pytest.approx(math.exp(-1.0))


def test_accept_reject_strongly_unfavorable_has_tiny_probability():
    key = jax.random.PRNGKey(0)

    _, accepted, p_acc = accept_reject(key, dE=1000.0, beta=1.0)

    assert accepted is False
    assert p_acc == pytest.approx(0.0)


def test_metropolis_probability_does_not_depend_on_sigma():
    # sigma is deliberately not an argument to accept_reject.
    # This documents the intended split:
    #   sigma ranks/selects candidates
    #   dE and beta decide Metropolis acceptance
    key = jax.random.PRNGKey(123)

    _, accepted_a, p_a = accept_reject(key, dE=1.0, beta=2.0)
    _, accepted_b, p_b = accept_reject(key, dE=1.0, beta=2.0)

    assert p_a == p_b
    assert accepted_a == accepted_b

