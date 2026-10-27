"""Fast, dependency-light checks for the independent empirical-sampling reference."""
import numpy as np
import pytest

from validation.a2_mh_exact_reference import STATES, energy
from validation.a2_mh_sampling_reference import (
    boltzmann, transition_matrix, occupation_standard_errors, sampling_diagnostics,
)


@pytest.mark.parametrize('p,kt', [(0.2, 0.5), (0.37, 1.0), (0.8, 2.0)])
def test_reference_kernel_and_stationary_measure(p, kt):
    e = np.array([energy(s) for s in STATES])
    pi = boltzmann(e, kt)
    kernel = transition_matrix(e, p_association=p, kt=kt)
    np.testing.assert_allclose(kernel.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(pi @ kernel, pi, atol=1e-12)
    np.testing.assert_allclose(pi[:, None] * kernel, pi[None, :] * kernel.T, atol=1e-12)
    assert np.all(kernel >= 0)
    assert np.all(pi > 0)


def test_autocorrelation_aware_errors_exceed_independent_draw_errors():
    e = np.array([energy(s) for s in STATES])
    pi = boltzmann(e, 1.0)
    kernel = transition_matrix(e, p_association=0.37, kt=1.0)
    se = occupation_standard_errors(kernel, pi, samples_per_chain=1000, chains=4)
    iid_se = np.sqrt(pi * (1 - pi) / 4000)
    assert np.all(np.isfinite(se))
    assert np.max(se / iid_se) > 1.2
    assert np.all(se > 0)


def test_exact_kernel_empirical_chain_passes_with_seed():
    e = np.array([energy(s) for s in STATES])
    pi = boltzmann(e, 1.0)
    kernel = transition_matrix(e, p_association=0.37, kt=1.0)
    rng = np.random.default_rng(37)
    counts = np.zeros(len(STATES), dtype=np.int64)
    samples, chains = 3000, 4
    for chain in range(chains):
        state = 0 if chain % 2 == 0 else len(STATES) - 1
        for step in range(500 + samples):
            state = rng.choice(len(STATES), p=kernel[state])
            if step >= 500:
                counts[state] += 1
    report = sampling_diagnostics(counts, e, p_association=0.37, kt=1.0,
                                  samples_per_chain=samples, chains=chains)
    assert report['within_bounds']
    assert report['total_variation'] < 0.10
    np.testing.assert_allclose(report['expected'], pi)
