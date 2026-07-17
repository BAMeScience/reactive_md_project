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

import jax.numpy as jnp
import pytest

from reactive_md.reaction import prepare_probe_geometry


def _free_space_functions():
    """Return displacement and shift functions without periodic boundaries."""

    def disp_fn(position_a, position_b):
        return position_b - position_a

    def shift_fn(position, displacement):
        return position + displacement

    return disp_fn, shift_fn


def _distance(R, atom_a, atom_b, disp_fn):
    displacement = disp_fn(R[atom_a], R[atom_b])
    return float(jnp.linalg.norm(displacement))


def _pf_lj_target(sigma_p, sigma_f):
    sigma_pf = 0.5 * (sigma_p + sigma_f)
    return 2.0 ** (1.0 / 6.0) * sigma_pf


def test_prepare_r_probe_moves_f_to_lj_target_and_preserves_lif():
    """The probe reaches the P-F target while preserving Li-F."""

    disp_fn, shift_fn = _free_space_functions()

    p_atom = 0
    f_atom = 1
    li_atom = 2

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],  # P
            [1.0, 0.0, 0.0],  # departing F
            [2.0, 1.0, 0.0],  # Li
        ],
        dtype=jnp.float32,
    )

    trial_sigmas = jnp.array(
        [3.0, 3.4, 2.1],
        dtype=jnp.float32,
    )

    initial_lif_distance = _distance(
        R,
        li_atom,
        f_atom,
        disp_fn,
    )

    expected_pf_distance = _pf_lj_target(
        float(trial_sigmas[p_atom]),
        float(trial_sigmas[f_atom]),
    )

    R_probe = prepare_probe_geometry(
        R,
        P_atom=p_atom,
        leave_F=f_atom,
        li_idx=li_atom,
        sigma_p=float(trial_sigmas[p_atom]),
        sigma_f=float(trial_sigmas[f_atom]),
        disp_fn=disp_fn,
        shift_fn=shift_fn,
    )

    actual_pf_distance = _distance(
        R_probe,
        p_atom,
        f_atom,
        disp_fn,
    )

    actual_lif_distance = _distance(
        R_probe,
        li_atom,
        f_atom,
        disp_fn,
    )

    assert actual_pf_distance == pytest.approx(
        expected_pf_distance,
        rel=1.0e-6,
    )

    assert actual_lif_distance == pytest.approx(
        initial_lif_distance,
        rel=1.0e-6,
    )

    # P and Li must remain unchanged.
    assert jnp.allclose(R_probe[p_atom], R[p_atom])
    assert jnp.allclose(R_probe[li_atom], R[li_atom])


def test_prepare_r_probe_returns_original_geometry_when_no_intersection():
    """An impossible pair of target distances leaves R unchanged."""

    disp_fn, shift_fn = _free_space_functions()

    p_atom = 0
    f_atom = 1
    li_atom = 2

    # P-Li = 2.0 and Li-F = 1.0.
    #
    # The P-F LJ target is approximately 3.59, so the two target
    # spheres cannot intersect:
    #
    # 3.59 > 2.0 + 1.0
    R = jnp.array(
        [
            [0.0, 0.0, 0.0],  # P
            [1.0, 0.0, 0.0],  # departing F
            [2.0, 0.0, 0.0],  # Li
        ],
        dtype=jnp.float32,
    )

    trial_sigmas = jnp.array(
        [3.0, 3.4, 2.1],
        dtype=jnp.float32,
    )

    R_probe = prepare_probe_geometry(
        R,
        P_atom=p_atom,
        leave_F=f_atom,
        li_idx=li_atom,
        sigma_p=float(trial_sigmas[p_atom]),
        sigma_f=float(trial_sigmas[f_atom]),
        disp_fn=disp_fn,
        shift_fn=shift_fn,
    )

    assert jnp.allclose(R_probe, R)


def test_prepare_r_probe_handles_collinear_geometry():
    """The deterministic fallback handles collinear P-F-Li atoms."""

    disp_fn, shift_fn = _free_space_functions()

    p_atom = 0
    f_atom = 1
    li_atom = 2

    # These atoms are collinear, but the target spheres intersect.
    # This exercises the fallback perpendicular direction.
    R = jnp.array(
        [
            [0.0, 0.0, 0.0],  # P
            [1.0, 0.0, 0.0],  # departing F
            [4.0, 0.0, 0.0],  # Li
        ],
        dtype=jnp.float32,
    )

    trial_sigmas = jnp.array(
        [3.0, 3.4, 2.1],
        dtype=jnp.float32,
    )

    initial_lif_distance = _distance(
        R,
        li_atom,
        f_atom,
        disp_fn,
    )

    expected_pf_distance = _pf_lj_target(
        float(trial_sigmas[p_atom]),
        float(trial_sigmas[f_atom]),
    )

    R_probe = prepare_probe_geometry(
        R,
        P_atom=p_atom,
        leave_F=f_atom,
        li_idx=li_atom,
        sigma_p=float(trial_sigmas[p_atom]),
        sigma_f=float(trial_sigmas[f_atom]),
        disp_fn=disp_fn,
        shift_fn=shift_fn,
    )

    actual_pf_distance = _distance(
        R_probe,
        p_atom,
        f_atom,
        disp_fn,
    )

    actual_lif_distance = _distance(
        R_probe,
        li_atom,
        f_atom,
        disp_fn,
    )

    assert jnp.all(jnp.isfinite(R_probe))

    assert actual_pf_distance == pytest.approx(
        expected_pf_distance,
        rel=1.0e-6,
    )

    assert actual_lif_distance == pytest.approx(
        initial_lif_distance,
        rel=1.0e-6,
    )

    assert jnp.allclose(R_probe[p_atom], R[p_atom])
    assert jnp.allclose(R_probe[li_atom], R[li_atom])
