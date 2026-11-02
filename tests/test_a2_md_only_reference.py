"""NumPy-only tests for fixed-topology MD control analysis."""
import numpy as np
import pytest

from validation.a2_md_only_reference import (
    initial_displacement, kinetic_diagnostics, conditional_radial_summary,
)
from validation.a2_mh_moving_reference import TwoAtomModel
from validation.a2_mh_mixing_reference import radial_reference


def test_initial_geometries():
    assert np.linalg.norm(initial_displacement('near')) == 2.0
    assert np.linalg.norm(initial_displacement('far')) > 3.0
    with pytest.raises(ValueError):
        initial_displacement('invalid')


def test_kinetic_ratios_for_equal_and_opposite_momenta():
    mass = 4.0
    kt = 0.6
    # p = (+a, -a), with a^2 = 3/2 m kT => K_relative = 3/2 kT.
    a = np.sqrt(1.5 * mass * kt)
    p = np.array([[a, 0, 0], [-a, 0, 0]])
    stats = kinetic_diagnostics(p, mass=mass, kt=kt)
    np.testing.assert_allclose(stats['relative_kinetic_ratio_3dof'], 1.0)
    np.testing.assert_allclose(stats['kinetic_ratio_6dof'], 0.5)
    np.testing.assert_allclose(stats['com_momentum_norm'], 0.0)


def test_kinetic_ratios_pure_com_motion():
    p = np.array([[2.0, 0, 0], [2.0, 0, 0]])
    stats = kinetic_diagnostics(p, mass=2.0, kt=1.0)
    assert stats['relative_kinetic_ratio_3dof'] == 0.0
    assert stats['kinetic_ratio_6dof'] > 0
    assert stats['com_momentum_norm'] == 4.0


def test_conditional_histograms_normalize_and_distinguish_topologies():
    model = TwoAtomModel()
    reference = radial_reference(model, order=24, bins=8)
    for bonded in (False, True):
        x = np.linspace(1.6, 3.5, 40)
        summary = conditional_radial_summary(x, reference=reference, bonded=bonded)
        assert 0 <= summary['conditional_radial_tv'] <= 1
        np.testing.assert_allclose(sum(summary['observed_conditional_shells']), 1.0)
        np.testing.assert_allclose(sum(summary['expected_conditional_shells']), 1.0)
        assert summary['min_r_A'] == x[0]
    a = conditional_radial_summary(x, reference=reference, bonded=False)
    b = conditional_radial_summary(x, reference=reference, bonded=True)
    assert a['expected_conditional_shells'] != b['expected_conditional_shells']


def test_invalid_diagnostic_inputs():
    with pytest.raises(ValueError):
        kinetic_diagnostics(np.zeros((3, 3)), mass=4, kt=1)
    with pytest.raises(ValueError):
        kinetic_diagnostics(np.zeros((2, 3)), mass=0, kt=1)
    reference = radial_reference(TwoAtomModel(), order=16, bins=6)
    with pytest.raises(ValueError):
        conditional_radial_summary([], reference=reference, bonded=False)
    with pytest.raises(ValueError):
        conditional_radial_summary([100], reference=reference, bonded=True)
