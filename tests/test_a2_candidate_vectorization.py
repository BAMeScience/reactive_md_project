"""Regression tests for the vectorized A2 association finder.

Run from the repository root with:
    python -m pytest -q tests/test_a2_candidate_vectorization.py

The tests import the production module directly from the repository.
This test uses only JAX, NumPy, and pytest; JAX-MD is not required.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.reactions.a2 import A2Reaction


@dataclass
class FakeSystem:
    bonds: tuple


def periodic_displacement(box_length):
    def displacement(a, b):
        d = a - b
        return d - box_length * jnp.round(d / box_length)
    return displacement


def scalar_reference(reaction, positions, displacement, system, cutoff=None):
    """Independent reproduction of the original nested-loop finder."""
    cutoff = reaction.association_cutoff if cutoff is None else float(cutoff)
    coords = np.asarray(positions)
    bonds = np.asarray(system.bonds[0], dtype=np.int32)
    a_set = set(map(int, reaction.a_indices))
    bonded = set()
    for i, j in bonds:
        if int(i) in a_set and int(j) in a_set:
            bonded.update((int(i), int(j)))
    free = [int(i) for i in reaction.a_indices if int(i) not in bonded]
    result = []
    for n, i in enumerate(free):
        for j in free[n + 1:]:
            dr = displacement(coords[i], coords[j])
            distance = float(np.linalg.norm(np.asarray(dr)))
            if distance <= cutoff:
                result.append((i, j, distance))
    return result


def compare(reaction, positions, disp_fn, system, cutoff=None):
    expected = scalar_reference(reaction, positions, disp_fn, system, cutoff)
    got = reaction.find_association_candidates(
        positions, disp_fn, system=system, cutoff=cutoff
    )
    assert [(x.i, x.j) for x in got] == [(x[0], x[1]) for x in expected]
    np.testing.assert_allclose(
        [x.distance for x in got],
        [x[2] for x in expected],
        atol=2e-6, rtol=2e-6,
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_random_periodic_positions(seed):
    rng = np.random.default_rng(seed)
    coords = jnp.asarray(rng.uniform(0, 30.4781, size=(128, 3)), dtype=jnp.float32)
    reaction = A2Reaction(
        a_indices=np.arange(64), k_bond_jax=29.8757,
        r0_bond=1.5, association_cutoff=2.7,
    )
    disp = periodic_displacement(30.4781)
    system = FakeSystem(bonds=(np.empty((0, 2), dtype=np.int32),))
    compare(reaction, coords, disp, system)
    compare(reaction, coords, disp, system, cutoff=4.0)
    # Check that the cached compiled function can be reused.
    compare(reaction, coords, disp, system)


def test_boundary_and_order():
    coords = jnp.asarray([
        [0.1, 0.0, 0.0], [9.9, 0.0, 0.0],
        [0.6, 0.0, 0.0], [5.0, 5.0, 5.0],
        [9.6, 0.0, 0.0], [4.0, 4.0, 4.0],
    ], dtype=jnp.float32)
    reaction = A2Reaction(
        a_indices=[4, 2, 0, 1], k_bond_jax=1.0,
        r0_bond=1.5, association_cutoff=1.0,
    )
    disp = periodic_displacement(10.0)
    system = FakeSystem(bonds=(np.empty((0, 2), dtype=np.int32),))
    compare(reaction, coords, disp, system)
    assert [(x.i, x.j) for x in reaction.find_association_candidates(
        coords, disp, system=system
    )] == [(4, 2), (4, 0), (4, 1), (2, 0), (2, 1), (0, 1)]


def test_bonded_a_excluded_and_non_a_bond_ignored():
    coords = jnp.asarray([
        [0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
        [0.4, 0.0, 0.0], [0.6, 0.0, 0.0],
        [0.8, 0.0, 0.0], [1.0, 0.0, 0.0],
    ], dtype=jnp.float32)
    reaction = A2Reaction(
        a_indices=[0, 1, 2, 3, 4], k_bond_jax=1.0,
        r0_bond=1.5, association_cutoff=2.7,
    )
    disp = periodic_displacement(10.0)
    system = FakeSystem(bonds=(np.asarray([[0, 1], [4, 5]], dtype=np.int32),))
    compare(reaction, coords, disp, system)
    assert [(x.i, x.j) for x in reaction.find_association_candidates(
        coords, disp, system=system
    )] == [(2, 3), (2, 4), (3, 4)]


def test_empty_and_single_free_a():
    reaction = A2Reaction(
        a_indices=[0, 1], k_bond_jax=1.0, r0_bond=1.5,
    )
    coords = jnp.asarray([[0, 0, 0], [1, 0, 0]], dtype=jnp.float32)
    disp = periodic_displacement(10.0)
    system = FakeSystem(bonds=(np.asarray([[0, 1]], dtype=np.int32),))
    compare(reaction, coords, disp, system)
    assert reaction.find_association_candidates(coords, disp, system=system) == []


def test_displacement_function_change_invalidates_cache():
    coords = jnp.asarray([[0.1, 0, 0], [9.9, 0, 0]], dtype=jnp.float32)
    reaction = A2Reaction(
        a_indices=[0, 1], k_bond_jax=1.0, r0_bond=1.5,
        association_cutoff=1.0,
    )
    system = FakeSystem(bonds=(np.empty((0, 2), dtype=np.int32),))
    compare(reaction, coords, periodic_displacement(10.0), system)
    compare(reaction, coords, periodic_displacement(20.0), system)
