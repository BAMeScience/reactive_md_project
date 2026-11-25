"""MACE-MD contracts using a differentiable surrogate, not model weights."""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.config import SimConfig
from reactive_md.engine import NVTStepper, ReactionOutcome
from reactive_md.md_driver import run_md_nvt


class MDNeighbors(NamedTuple):
    idx: object
    did_buffer_overflow: object


class MDNeighborFunction:
    def allocate(self, positions, *, extra_capacity=0):
        return MDNeighbors(jnp.zeros((len(positions), 1), dtype=jnp.int32),
                           jnp.asarray(False))

    def update(self, positions, neighbors):
        assert isinstance(neighbors, MDNeighbors)
        return neighbors


def md_energy(positions, neighbor):
    assert isinstance(neighbor, MDNeighbors)
    return jnp.sum(positions ** 2) / 2


class ClassicalNeighbors:
    def __init__(self, positions, overflow=False):
        self.positions = np.asarray(positions).copy()
        self.did_buffer_overflow = overflow


class ClassicalNeighborFunction:
    def __init__(self, *, always_overflow=False):
        self.capacities = []
        self.always_overflow = always_overflow

    def allocate(self, positions, *, extra_capacity=0):
        self.capacities.append(extra_capacity)
        return ClassicalNeighbors(
            positions, self.always_overflow or extra_capacity < 8
        )


def classical_ff(positions, *, always_overflow=False):
    def forbidden_energy(*args, **kwargs):
        raise AssertionError('Classical energy used for MACE MD')

    return SimpleNamespace(
        nlist=ClassicalNeighbors(positions),
        neighbor_fn=ClassicalNeighborFunction(always_overflow=always_overflow),
        energy_fn=forbidden_energy,
    )


def shift(positions, displacement, **kwargs):
    return positions + displacement


@dataclass(frozen=True)
class State:
    position: object
    momentum: object
    force: object
    thermostat_chain: object


@pytest.mark.parametrize('preserve_momenta', [False, True])
def test_mace_synchronize_keeps_integrator_and_classical_list(preserve_momenta):
    positions = jnp.ones((2, 3))
    masses = jnp.asarray([2., 3.])
    old_ff = classical_ff(positions)
    new_ff = classical_ff(positions + 1)
    old_list, new_list = old_ff.nlist, new_ff.nlist
    neighbors = MDNeighborFunction()
    stepper = NVTStepper(cfg=SimConfig(), ff=old_ff, shift_fn=shift,
                         masses=masses, md_energy_fn=md_energy,
                         md_neighbor_fn=neighbors)
    init_nvt, md_chunk = stepper.init_nvt, stepper.md_chunk
    chain = object()
    state = State(positions, positions * 3, -positions, chain)
    outcome = ReactionOutcome(
        key=None, positions=positions + 1, velocities=positions * 4,
        system=None, ff=new_ff, events=('reaction',),
        preserve_momenta=preserve_momenta,
    )
    updated, nlist = stepper.synchronize(state, outcome)
    assert stepper.init_nvt is init_nvt
    assert stepper.md_chunk is md_chunk
    assert stepper.neighbor_fn is neighbors
    assert stepper.ff is new_ff
    assert stepper.nlist is nlist
    assert isinstance(nlist, MDNeighbors)
    assert old_ff.nlist is old_list
    assert new_ff.nlist is new_list
    assert updated.thermostat_chain is chain
    np.testing.assert_array_equal(updated.position, outcome.positions)
    np.testing.assert_allclose(updated.force, -outcome.positions)
    expected = state.momentum if preserve_momenta else outcome.velocities * masses[:, None]
    np.testing.assert_allclose(updated.momentum, expected)


def test_mace_driver_continues_after_reaction_with_separate_lists():
    positions = jnp.asarray([[1., 0., 0.], [-1., 0., 0.]])
    old_ff = classical_ff(positions)
    initial_list = old_ff.nlist
    new_ff = classical_ff(positions)
    old_system, new_system = object(), object()
    seen = []
    observations = []

    class Handler:
        def attempt(self, context):
            seen.append(context)
            expected_ff = old_ff if len(seen) == 1 else new_ff
            assert context.ff is expected_ff
            assert isinstance(context.ff.nlist, ClassicalNeighbors)
            assert not context.ff.nlist.did_buffer_overflow
            np.testing.assert_array_equal(context.ff.nlist.positions, context.positions)
            if len(seen) == 1:
                assert context.system is old_system
                relaxed = context.positions + 0.25
                new_ff.nlist = ClassicalNeighbors(relaxed)
                return ReactionOutcome(
                    key=context.key, positions=relaxed,
                    velocities=context.velocities, system=new_system,
                    ff=new_ff, events=('reaction',), relaxation_time_ps=0.01,
                )
            assert context.system is new_system
            return ReactionOutcome(
                key=context.key, positions=context.positions,
                velocities=context.velocities, system=context.system, ff=context.ff,
            )

    class Observer:
        def on_observation(self, observation):
            assert isinstance(observation.ff.nlist, ClassicalNeighbors)
            if observation.step == 0:
                assert observation.ff.nlist is initial_list
            np.testing.assert_allclose(observation.state.force,
                                       -observation.state.position, rtol=1e-6, atol=1e-6)
            observations.append(observation)

    result = run_md_nvt(
        jax.random.PRNGKey(0), cfg=SimConfig(steps=4, check_every=2, dt=0.001),
        init_positions=positions, masses=jnp.ones(2), shift_fn=shift,
        ff=old_ff, sys=old_system, reaction_handler=Handler(),
        observers=(Observer(),), md_energy_fn=md_energy,
        md_neighbor_fn=MDNeighborFunction(),
    )
    assert result.steps_done == 4
    assert result.accepted_events == 1
    assert result.ff is new_ff and result.sys is new_system
    assert result.cumulative_relaxation_time_ps == pytest.approx(0.01)
    assert [o.step for o in observations] == [0, 2, 4]
    np.testing.assert_allclose(observations[1].state.position, seen[0].positions + 0.25)
    assert not np.array_equal(observations[1].state.position, result.final_md_state.position)
    assert old_ff.neighbor_fn.capacities == [0, 8]
    assert new_ff.neighbor_fn.capacities == [0, 8]


def test_mace_driver_rejects_persistent_classical_overflow():
    positions = jnp.asarray([[1., 0., 0.], [-1., 0., 0.]])
    ff = classical_ff(positions, always_overflow=True)

    class Handler:
        def attempt(self, context):
            pytest.fail('Reaction check received an overflowing neighbor list')

    with pytest.raises(RuntimeError, match='Classical reaction neighbor-list allocation overflowed'):
        run_md_nvt(
            jax.random.PRNGKey(0), cfg=SimConfig(steps=2, check_every=2, dt=0.001),
            init_positions=positions, masses=jnp.ones(2), shift_fn=shift,
            ff=ff, sys=object(), reaction_handler=Handler(),
            md_energy_fn=md_energy, md_neighbor_fn=MDNeighborFunction(),
        )
    assert len(ff.neighbor_fn.capacities) == 13


def test_mace_force_refresh_reuses_compiled_kernel():
    positions = jnp.ones((2, 3))
    traces = []

    def energy(r, neighbor):
        traces.append(None)
        return md_energy(r, neighbor)

    neighbors = MDNeighborFunction()
    stepper = NVTStepper(cfg=SimConfig(), ff=classical_ff(positions),
                         shift_fn=shift, masses=jnp.ones(2),
                         md_energy_fn=energy, md_neighbor_fn=neighbors)
    nl = neighbors.allocate(positions)
    state = State(positions, positions, positions, None)
    kernel = stepper.force_fn
    traces.clear()
    first = stepper.refresh_force(state, nl)
    outcome = ReactionOutcome(
        key=None, positions=positions + 2, velocities=positions,
        system=None, ff=classical_ff(positions), events=('reaction',),
    )
    second, _ = stepper.synchronize(first, outcome)
    np.testing.assert_allclose(first.force, -positions)
    np.testing.assert_allclose(second.force, -(positions + 2))
    assert stepper.force_fn is kernel
    assert len(traces) == 1
