"""JAX-MD-free safety tests for the chemistry-neutral NVT stepper."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.engine import NVTStepper, ReactionOutcome


@dataclass(frozen=True)
class FakeState:
    position: object
    momentum: object
    velocity: object
    force: object
    thermostat_chain: str = 'original'


def _state():
    pos = jnp.zeros((2, 3))
    return FakeState(pos, pos, pos, pos)


def _neighbor(*, overflow=False, capacity=4):
    return SimpleNamespace(
        did_buffer_overflow=overflow,
        idx=np.zeros((2, capacity), dtype=int),
    )


def _stepper():
    # Bypass JAX-MD construction: exercise transactional recovery logic alone.
    stepper = object.__new__(NVTStepper)
    stepper.max_retries = 2
    return stepper


def test_overflow_retries_from_original_state_and_valid_list():
    stepper = _stepper()
    original = _state()
    initial_nl = _neighbor()
    seen = []
    grown = _neighbor(capacity=8)

    def chunk(state, nlist, *, chunk):
        seen.append((state, nlist, chunk))
        if len(seen) == 1:
            return replace(state, position=state.position + 99), nlist, True
        return replace(state, position=state.position + 1), nlist, False

    stepper.md_chunk = chunk
    stepper.allocate_valid_nlist = lambda position, *, extra_capacity: grown
    stepper.refresh_force = lambda state, nlist: replace(state, force=state.force + 1)
    result, nlist = stepper.advance(original, initial_nl, chunk=5)
    assert len(seen) == 2
    assert seen[0][0] is original
    assert seen[1][1] is grown
    np.testing.assert_array_equal(np.asarray(seen[1][0].position), np.zeros((2, 3)))
    np.testing.assert_array_equal(np.asarray(result.position), np.ones((2, 3)))
    assert nlist is grown


def test_persistent_neighbor_overflow_fails_closed():
    stepper = _stepper()
    stepper.md_chunk = lambda state, nl, *, chunk: (state, nl, True)
    stepper.allocate_valid_nlist = lambda pos, *, extra_capacity: _neighbor()
    stepper.refresh_force = lambda state, nl: state
    with pytest.raises(RuntimeError, match='overflow persisted'):
        stepper.advance(_state(), _neighbor(), chunk=1)


@pytest.mark.parametrize('bad_field', ['position', 'momentum'])
def test_nonfinite_accepted_md_chunk_is_rejected(bad_field):
    stepper = _stepper()
    def chunk(state, nl, *, chunk):
        return replace(state, **{bad_field: jnp.full((2, 3), jnp.nan)}), nl, False
    stepper.md_chunk = chunk
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        stepper.advance(_state(), _neighbor(), chunk=1)


def test_synchronize_preserves_thermostat_chain_without_reset():
    stepper = _stepper()
    stepper.mass = jnp.asarray([2., 3.])
    stepper.set_forcefield = lambda ff: setattr(stepper, 'ff', ff)
    stepper.allocate_valid_nlist = lambda positions: _neighbor()
    stepper.refresh_force = lambda state, nl: replace(state, force=jnp.ones((2, 3)))
    ff = SimpleNamespace(nlist=None)
    outcome = ReactionOutcome(
        key=None, positions=jnp.ones((2, 3)),
        velocities=jnp.full((2, 3), 4.), system=None, ff=ff,
        events=('association',),
    )
    state, nlist = stepper.synchronize(_state(), outcome)
    assert state.thermostat_chain == 'original'
    np.testing.assert_allclose(np.asarray(state.momentum), [[8.] * 3, [12.] * 3])
    np.testing.assert_allclose(np.asarray(state.force), np.ones((2, 3)))
    assert ff.nlist is nlist


def test_synchronize_fixed_coordinate_mh_preserves_momenta_exactly():
    stepper = _stepper()
    stepper.mass = jnp.asarray([2., 39.948])
    stepper.set_forcefield = lambda ff: setattr(stepper, 'ff', ff)
    stepper.allocate_valid_nlist = lambda positions: _neighbor()
    stepper.refresh_force = lambda state, nl: replace(state, force=jnp.ones((2, 3)))
    ff = SimpleNamespace(nlist=None)
    initial_momenta = jnp.asarray([[0.25, -0.5, 0.75], [-0.636, 0.335, 1.29]])
    state = replace(_state(), momentum=initial_momenta, velocity=initial_momenta)
    outcome = ReactionOutcome(
        key=None, positions=jnp.ones((2, 3)),
        velocities=state.velocity, system=None, ff=ff,
        events=('association',), preserve_momenta=True,
    )
    updated, _ = stepper.synchronize(state, outcome)
    np.testing.assert_array_equal(np.asarray(updated.momentum),
                                  np.asarray(initial_momenta))
    assert updated.thermostat_chain == 'original'
    np.testing.assert_array_equal(np.asarray(updated.position), np.ones((2, 3)))
