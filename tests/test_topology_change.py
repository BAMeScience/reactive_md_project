import numpy as np
import jax.numpy as jnp
import pytest

from reactive_md.reaction import make_probe_geometry


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

