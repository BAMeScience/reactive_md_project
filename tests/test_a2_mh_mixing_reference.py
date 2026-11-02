"""NumPy-only tests of independent radial reference and mixing diagnostics."""
import numpy as np
import pytest

from validation.a2_mh_moving_reference import TwoAtomModel, integrate_reference
from validation.a2_mh_mixing_reference import (
    radial_reference, integrated_autocorrelation, batch_means_se, observed_radial,
)


def test_radial_reference_normalizes_and_matches_partition():
    model = TwoAtomModel()
    ref = radial_reference(model, order=64, bins=10)
    exact = integrate_reference(model, order=64)
    assert abs(ref['p_bonded'] - exact['p_bonded']) < 1e-10
    assert np.isclose(sum(ref['prob_r_by_shell']), 1.0)
    assert np.isclose(sum(ref['prob_bonded_by_shell']), exact['p_bonded'])
    assert np.isclose(sum(ref['prob_r_given_unbound']), 1.0)
    assert np.isclose(sum(ref['prob_r_given_bonded']), 1.0)


def test_autocorrelation_constant_is_undefined():
    result = integrated_autocorrelation(np.ones(100))
    assert result['ess'] is None
    assert result['tau_checks'] is None


def test_autocorrelation_iid_and_persistent():
    rng = np.random.default_rng(8)
    iid = rng.normal(size=3000)
    persistent = np.cumsum(rng.normal(size=3000))
    assert integrated_autocorrelation(iid)['ess'] > 400
    assert integrated_autocorrelation(persistent)['ess'] < 200


def test_batch_se_requires_four_batches():
    assert batch_means_se([np.arange(8)], block_size=4)['se'] is None
    result = batch_means_se([np.arange(40)], block_size=5)
    assert result['n_batches'] == 8
    assert result['se'] > 0
    with pytest.raises(ValueError):
        batch_means_se([np.arange(40)], block_size=0)


def test_observed_histogram_and_validation():
    ref = radial_reference(TwoAtomModel(), order=32, bins=8)
    samples = np.asarray([[0, 2.0], [1, 2.0], [1, 3.0], [0, 3.5]])
    obs = observed_radial(samples, ref)
    assert np.isclose(sum(obs['observed_r_by_shell']), 1.0)
    assert 0 <= obs['radial_total_variation'] <= 1
    with pytest.raises(ValueError):
        observed_radial(np.empty((0, 2)), ref)
