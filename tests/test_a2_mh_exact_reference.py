"""Dependency-free exact 10-state equilibrium reference for the A2 MH kernel."""

import math

import pytest

from validation.a2_mh_exact_reference import (
    STATES, boltzmann_weights, transition_matrix,
)


@pytest.mark.parametrize('p_association', [0.2, 0.5, 0.8])
@pytest.mark.parametrize('kT', [0.6, 1.0, 2.5])
def test_ten_state_kernel_satisfies_detailed_balance_and_stationarity(
        p_association, kT):
    assert len(STATES) == 10
    P = transition_matrix(p_association=p_association, kT=kT)
    pi = boltzmann_weights(kT=kT)
    assert sum(pi) == pytest.approx(1.0, abs=1e-14)
    for i in range(10):
        assert sum(P[i]) == pytest.approx(1.0, abs=1e-14)
        assert all(-1e-14 <= value <= 1.0 + 1e-14 for value in P[i])
        for j in range(10):
            assert pi[i] * P[i][j] == pytest.approx(pi[j] * P[j][i], abs=1e-14)
    for j in range(10):
        assert sum(pi[i] * P[i][j] for i in range(10)) == pytest.approx(
            pi[j], abs=1e-14)


def test_all_twelve_edges_and_null_direction_mass():
    P = transition_matrix(p_association=0.37, kT=1.0)
    undirected = sum(P[i][j] > 0 or P[j][i] > 0
                     for i in range(10) for j in range(i + 1, 10))
    assert undirected == 12
    # The all-monomer state has no dissociation candidates, so at least the
    # entire dissociation-direction probability remains as a self transition.
    assert P[0][0] >= 0.63 - 1e-14
    # Three fully dimerized states have no association candidates.
    for index in range(7, 10):
        assert P[index][index] >= 0.37 - 1e-14
