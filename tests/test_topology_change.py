import numpy as np
import jax.numpy as jnp
import pytest

from reactive_md.reaction import make_probe_geometry
from reactive_md.reaction import prepare_probe_geometry


def _disp(a, b):
    return b - a


def _shift(r, dr):
    return r + dr


def test_make_probe_geometry_moves_leaving_f_along_pf_direction():
    R = jnp.array(
        [
            [0.0, 0.0, 0.0],  # P
            [1.0, 0.0, 0.0],  # leaving F
        ],
        dtype=jnp.float32,
    )

    R_new = make_probe_geometry(
        R,
        P_atom=0,
        leave_F=1,
        disp_fn=_disp,
        shift_fn=_shift,
        r_pf_probe=4.0,
    )

    np.testing.assert_allclose(np.asarray(R_new[0]), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(R_new[1]), [4.0, 0.0, 0.0])


def test_make_probe_geometry_does_not_change_unrelated_atoms():
    R = jnp.array(
        [
            [0.0, 0.0, 0.0],  # P
            [1.0, 0.0, 0.0],  # leaving F
            [0.0, 2.0, 0.0],  # unrelated atom
        ],
        dtype=jnp.float32,
    )

    R_new = make_probe_geometry(
        R,
        P_atom=0,
        leave_F=1,
        disp_fn=_disp,
        shift_fn=_shift,
        r_pf_probe=4.0,
    )

    np.testing.assert_allclose(np.asarray(R_new[2]), [0.0, 2.0, 0.0])


def _free_space_functions():
    """Non-periodic displacement and shift functions for unit tests."""

    def disp_fn(r_a, r_b):
        return r_b - r_a

    def shift_fn(r, dr):
        return r + dr

    return disp_fn, shift_fn


def test_prepare_r_probe_moves_f_to_lj_target():
    disp_fn, shift_fn = _free_space_functions()

    p_atom = 0
    f_atom = 1

    # Initial P-F distance is 1.0, deliberately below the LJ target.
    R = jnp.array(
        [
            [0.0, 0.0, 0.0],  # P
            [1.0, 0.0, 0.0],  # F
        ]
    )

    trial_sigmas = jnp.array([3.0, 3.4])

    R_probe = prepare_probe_geometry(
        R,
        P_atom=p_atom,
        F_atom=f_atom,
        trial_sigmas=trial_sigmas,
        disp_fn=disp_fn,
        shift_fn=shift_fn,
    )

    sigma_pf = 0.5 * (
        float(trial_sigmas[p_atom])
        + float(trial_sigmas[f_atom])
    )
    expected_distance = 2.0 ** (1.0 / 6.0) * sigma_pf

    pf_vector = disp_fn(
        R_probe[p_atom],
        R_probe[f_atom],
    )
    actual_distance = float(jnp.linalg.norm(pf_vector))

    assert actual_distance == pytest.approx(
        expected_distance,
        rel=1.0e-6,
    )

    # P should remain unchanged.
    assert jnp.allclose(R_probe[p_atom], R[p_atom])

    # The departure direction should remain +x.
    original_direction = disp_fn(R[p_atom], R[f_atom])
    probe_direction = disp_fn(
        R_probe[p_atom],
        R_probe[f_atom],
    )

    original_direction /= jnp.linalg.norm(original_direction)
    probe_direction /= jnp.linalg.norm(probe_direction)

    assert jnp.allclose(
        probe_direction,
        original_direction,
        atol=1.0e-6,
    )


def test_prepare_r_probe_does_not_move_f_when_already_separated():
    disp_fn, shift_fn = _free_space_functions()

    p_atom = 0
    f_atom = 1

    trial_sigmas = jnp.array([3.0, 3.4])

    sigma_pf = 0.5 * (
        float(trial_sigmas[p_atom])
        + float(trial_sigmas[f_atom])
    )
    target_distance = 2.0 ** (1.0 / 6.0) * sigma_pf

    # Place F beyond the required target distance.
    initial_distance = target_distance + 1.0
    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [initial_distance, 0.0, 0.0],
        ]
    )

    R_probe = prepare_probe_geometry(
        R,
        P_atom=p_atom,
        F_atom=f_atom,
        trial_sigmas=trial_sigmas,
        disp_fn=disp_fn,
        shift_fn=shift_fn,
    )

    assert jnp.allclose(R_probe, R)


def test_prepare_r_probe_returns_finite_coordinates():
    disp_fn, shift_fn = _free_space_functions()

    p_atom = 0
    f_atom = 1

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )

    trial_sigmas = jnp.array([3.0, 3.4])

    R_probe = prepare_probe_geometry(
        R,
        P_atom=p_atom,
        F_atom=f_atom,
        trial_sigmas=trial_sigmas,
        disp_fn=disp_fn,
        shift_fn=shift_fn,
    )

    assert bool(jnp.all(jnp.isfinite(R_probe)))
