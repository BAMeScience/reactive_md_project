
"""Shared A2 model parameters and initial system construction."""

import jax.numpy as jnp
import numpy as np

from ..reaction import SystemState


KB_REAL = 0.0019872041
EPSILON = 1.9572 / 4.184
SIGMA = 3.4
K_BOND_JAX = 0.5 * 25000.0 / 100.0 / 4.184


def make_system(n_atoms):
    empty_four = (
        np.empty((0, 4), dtype=np.int32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.float32),
    )

    return SystemState(
        bonds=(
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ),
        angles=(
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ),
        torsions=empty_four,
        impropers=empty_four,
        charges=np.zeros(n_atoms, dtype=np.float32),
        sigmas=np.full(n_atoms, SIGMA, dtype=np.float32),
        epsilons=np.full(n_atoms, EPSILON, dtype=np.float32),
        molecule_id=np.arange(n_atoms, dtype=np.int32),
        pf6_reacted=jnp.zeros((0,), dtype=jnp.bool_),
    )

