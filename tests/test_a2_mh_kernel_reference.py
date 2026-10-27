"""Independent NumPy-only tests for the moving-coordinate two-atom MH kernel."""

import math

import pytest

from validation.a2_mh_kernel_reference import kernel_at_distance
from validation.a2_mh_moving_reference import TwoAtomModel, integrate_reference


def test_conditional_probability_at_initial_geometry():
    reference = kernel_at_distance(2.0)
    assert reference['conditional_bonded_probability'] == pytest.approx(0.68547, abs=1e-5)
    assert reference['association_eligible']
    assert reference['association_acceptance'] == pytest.approx(1.0)
    assert reference['dissociation_acceptance'] == pytest.approx(0.45886, abs=1e-5)


@pytest.mark.parametrize('r', [1.5, 2.0, 2.5, 2.95, 3.0, 3.05, 3.5])
def test_local_detailed_balance_and_cutoff(r):
    result = kernel_at_distance(r)
    assert result['forward_flux'] == pytest.approx(result['reverse_flux'], rel=1e-12)
    if r > TwoAtomModel().association_cutoff:
        assert result['association_acceptance'] == 0
        assert result['dissociation_acceptance'] == 0
    else:
        assert 0 < result['association_acceptance'] <= 1
        assert 0 < result['dissociation_acceptance'] <= 1
    assert 0 < result['conditional_bonded_probability'] < 1


def test_full_box_reference_converges_and_differs_from_initial_conditional():
    m = TwoAtomModel()
    p64 = integrate_reference(m, order=64)['p_bonded']
    p96 = integrate_reference(m, order=96)['p_bonded']
    assert p64 == pytest.approx(p96, abs=2e-5)
    assert p96 == pytest.approx(0.284330, abs=1e-5)
    assert abs(p96 - kernel_at_distance(2.0, m)['conditional_bonded_probability']) > 0.3


def test_invalid_distance():
    for r in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            kernel_at_distance(r)
