"""Pure NumPy tests; no JAX-MD needed."""
import numpy as np
import pytest
from validation.a2_thermostat_calibration_math import (
    mass_from_amu, momentum_ratios, summarize_momenta, assess_langevin,
)


def test_mass_conversion():
    assert mass_from_amu(4.0) == pytest.approx(0.0095602294455, rel=1e-9)
    with pytest.raises(ValueError):
        mass_from_amu(0)


def test_pure_relative_and_com_kinetic_decomposition():
    p = np.array([[[1., 0, 0], [-1., 0, 0]],
                  [[1., 0, 0], [1., 0, 0]]])
    r = momentum_ratios(p, mass=1., kT=1.)
    np.testing.assert_allclose(r['total'], [1 / 3, 1 / 3])
    np.testing.assert_allclose(r['relative'], [2 / 3, 0])
    np.testing.assert_allclose(r['com'], [0, 2 / 3])
    np.testing.assert_allclose(r['total'], (r['relative'] + r['com']) / 2)


def test_canonical_gaussian_momenta():
    rng = np.random.default_rng(123)
    mass, kT = 4., 0.6
    p = rng.normal(scale=np.sqrt(mass * kT), size=(10, 2000, 2, 3))
    stats = summarize_momenta(p, mass, kT)
    for k in ('total', 'relative', 'com'):
        assert abs(stats[k]['mean'] - 1) < 0.04
    assert assess_langevin(stats)['status'] == 'CONSISTENT_WITH_CANONICAL_MOMENTA'


def test_detect_underthermalized_momenta():
    rng = np.random.default_rng(4)
    p = rng.normal(scale=np.sqrt(0.25 * 4 * 0.6), size=(10, 2000, 2, 3))
    stats = summarize_momenta(p, 4., 0.6)
    assert assess_langevin(stats)['status'] == 'INVESTIGATE_KINETIC_NORMALIZATION'


def test_invalid_shapes_and_parameters():
    with pytest.raises(ValueError):
        momentum_ratios(np.ones((3, 3)), 4., 1.)
    with pytest.raises(ValueError):
        momentum_ratios(np.ones((2, 3)), -1., 1.)
    with pytest.raises(ValueError):
        summarize_momenta(np.ones((1, 1, 2, 3)), 4., 1.)
