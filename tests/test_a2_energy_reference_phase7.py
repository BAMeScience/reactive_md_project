"""Pure NumPy checks of the independent spectator LJ reference (no JAX-MD)."""
import numpy as np
import pytest

from reactive_md.reactions.a2_energy_reference import spectator_lj_reference


def test_reference_omits_reacting_pair_and_uses_shifted_lj():
    R = np.asarray([[0., 0., 0.], [1., 0., 0.],
                    [4., 0., 0.], [7., 0., 0.]])
    sigma, epsilon, cutoff = 3.4, 0.5, 9.0
    sr6 = (sigma / 3.) ** 6
    sc6 = (sigma / cutoff) ** 6
    expected = 4 * epsilon * (sr6 ** 2 - sr6 - sc6 ** 2 + sc6)
    assert spectator_lj_reference(R, [20., 20., 20.], (0, 1),
                                  sigma=sigma, epsilon=epsilon, cutoff=cutoff) == pytest.approx(expected)


def test_reference_minimum_image_and_cutoff():
    R = np.asarray([[5., 5., 5.], [6., 5., 5.],
                    [0.5, 0., 0.], [19.5, 0., 0.]])
    sigma, epsilon, cutoff = 3.4, 0.5, 9.0
    sr6 = (sigma / 1.) ** 6
    sc6 = (sigma / cutoff) ** 6
    expected = 4 * epsilon * (sr6 ** 2 - sr6 - sc6 ** 2 + sc6)
    assert spectator_lj_reference(R, [20., 20., 20.], (0, 1),
                                  sigma=sigma, epsilon=epsilon, cutoff=cutoff) == pytest.approx(expected)
    R[3] = [10., 10., 10.]
    assert spectator_lj_reference(R, [20., 20., 20.], (0, 1),
                                  sigma=sigma, epsilon=epsilon, cutoff=cutoff) == 0.0
