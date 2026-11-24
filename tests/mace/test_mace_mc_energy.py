from typing import NamedTuple

import jax.numpy as jnp
import pytest

from reactive_md.mace_mc_energy import MaceJaxEnergyEvaluator


class FakeNeighbor(NamedTuple):
    did_buffer_overflow: object

    def update(self, positions):
        return self


class FakeNeighborFunction:
    def __init__(self, always_overflow=False):
        self.capacities = []
        self.always_overflow = always_overflow

    def allocate(self, positions, *, extra_capacity=0):
        self.capacities.append(extra_capacity)
        overflow = self.always_overflow or extra_capacity < 8
        return FakeNeighbor(jnp.asarray(overflow))


def make_energy_fn(neighbors):
    def energy_fn(positions, *, box):
        return jnp.sum(positions ** 2)
    return energy_fn


def test_mace_mc_recovers_neighbor_overflow():
    neighbors = FakeNeighborFunction()
    evaluator = MaceJaxEnergyEvaluator(
        neighbor_fn=neighbors,
        make_energy_fn=make_energy_fn,
        box0=jnp.array([10.0, 10.0, 10.0]),
    )

    energy = evaluator.energy(jnp.array([[1.0, 2.0, 3.0]]))

    assert energy == pytest.approx(14.0)
    assert neighbors.capacities == [2, 8]


def test_mace_mc_rejects_persistent_neighbor_overflow():
    neighbors = FakeNeighborFunction(always_overflow=True)
    evaluator = MaceJaxEnergyEvaluator(
        neighbor_fn=neighbors,
        make_energy_fn=make_energy_fn,
        box0=jnp.array([10.0, 10.0, 10.0]),
    )

    with pytest.raises(RuntimeError, match="overflowed"):
        evaluator.energy(jnp.array([[1.0, 2.0, 3.0]]))

    assert len(neighbors.capacities) == 13


def test_mace_mc_uses_updated_neighbors_and_current_positions():
    class Neighbor(NamedTuple):
        did_buffer_overflow: object
        capacity: object
        reference: object

        def update(self, positions):
            return Neighbor(self.capacity < 8, self.capacity, positions)

    class NeighborFunction:
        def __init__(self):
            self.capacities = []

        def allocate(self, positions, *, extra_capacity):
            self.capacities.append(extra_capacity)
            return Neighbor(jnp.asarray(False), jnp.asarray(extra_capacity),
                            jnp.zeros_like(positions))

    def make_energy(neighbors):
        def energy(positions, *, box):
            return jnp.sum(positions) + jnp.sum(neighbors.reference) + box[0]
        return energy

    neighbor_fn = NeighborFunction()
    evaluator = MaceJaxEnergyEvaluator(
        neighbor_fn=neighbor_fn, make_energy_fn=make_energy,
        box0=jnp.array([10., 10., 10.]),
    )
    first = evaluator.energy(jnp.array([[1., 2., 3.]]))
    second = evaluator.energy(jnp.array([[4., 5., 6.]]))
    assert isinstance(first, float) and isinstance(second, float)
    assert first == pytest.approx(22.)
    assert second == pytest.approx(40.)
    assert neighbor_fn.capacities == [2, 8, 2, 8]
