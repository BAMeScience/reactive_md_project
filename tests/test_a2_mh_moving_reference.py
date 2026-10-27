"""Pure NumPy checks; no JAX-MD dependency."""
import numpy as np
import pytest

from validation.a2_mh_moving_reference import (
    TwoAtomModel, energy_unbound, energy_bonded, integrate_reference,
)


def test_energy_functions_and_cutoff():
    model = TwoAtomModel()
    np.testing.assert_allclose(energy_unbound([model.sigma], model), [0.0], atol=1e-12)
    assert energy_unbound(4.0, model) == 0.0
    assert energy_bonded(model.bond_r0, model) == model.bond_offset
    assert energy_bonded(3.0, model) > energy_bonded(2.0, model)


def test_cube_quadrature_converges():
    model = TwoAtomModel()
    low = integrate_reference(model, order=64)
    high = integrate_reference(model, order=96)
    assert 0.0 < high['p_bonded'] < 1.0
    assert abs(low['p_bonded'] - high['p_bonded']) < 0.003
    assert abs(low['mean_r_unbound'] - high['mean_r_unbound']) < 0.02
    assert abs(low['mean_r_bonded'] - high['mean_r_bonded']) < 0.02


def test_reference_rejects_inconsistent_periodic_cutoff():
    with pytest.raises(ValueError, match='r_cut'):
        integrate_reference(TwoAtomModel(r_cut=4.0), order=16)
