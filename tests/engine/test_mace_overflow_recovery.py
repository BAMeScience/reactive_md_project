"""Host-side tests for bounded, transactional MACE NVT recovery."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.engine import NVTStepper


@dataclass(frozen=True)
class State:
    position: object
    momentum: object
    force: object
    thermostat: int = 7


def neighbor(width, overflow=False):
    return SimpleNamespace(idx=np.zeros((2, width), dtype=np.int32),
                           did_buffer_overflow=overflow)


def setup(base=97, *, mace=True, retries=3, fixed=False):
    stepper = object.__new__(NVTStepper)
    stepper.md_energy_fn = (lambda r, n: jnp.sum(r ** 2)) if mace else None
    stepper.max_retries = retries
    requests = []
    refreshed = []
    pos = jnp.ones((2, 3))
    state = State(pos, pos * 2, -2 * pos)

    def allocate(positions, *, extra_capacity):
        np.testing.assert_array_equal(positions, state.position)
        requests.append(extra_capacity)
        return neighbor(base if fixed else base + extra_capacity)

    def refresh(s, nl):
        refreshed.append(nl.idx.shape[-1])
        return replace(s, force=-2 * s.position)

    stepper.neighbor_fn = SimpleNamespace(allocate=allocate)
    stepper.refresh_force = refresh
    return stepper, state, requests, refreshed


@pytest.mark.parametrize('base,expected_requests', [(97, [10, 13]), (110, [10])])
def test_mace_recovery_matches_sufficient_list(base, expected_requests):
    stepper, original, requests, refreshed = setup(base)
    seen = []

    def chunk(s, nl, *, chunk):
        seen.append(nl.idx.shape[-1])
        np.testing.assert_array_equal(s.position, original.position)
        np.testing.assert_array_equal(s.momentum, original.momentum)
        assert s.thermostat == original.thermostat
        if nl.idx.shape[-1] < 110:
            return replace(s, position=s.position + 99, momentum=s.momentum + 99,
                           thermostat=99), nl, True
        np.testing.assert_array_equal(s.force, -2 * s.position)
        return replace(s, position=s.position + 1, momentum=s.momentum + 1), nl, False

    stepper.md_chunk = chunk
    result, nl = stepper.advance(original, neighbor(100), chunk=1000)
    reference, _ = stepper.advance(original, neighbor(nl.idx.shape[-1]), chunk=1000)
    assert requests == expected_requests
    assert refreshed == [nl.idx.shape[-1]]
    assert seen[0] == 100 and seen[1] >= 110
    np.testing.assert_array_equal(result.position, reference.position)
    np.testing.assert_array_equal(result.momentum, reference.momentum)
    assert result.thermostat == reference.thermostat == 7


def test_mace_repeated_overflows_grow_progressively():
    stepper, state, requests, refreshed = setup()
    widths = []

    def chunk(s, nl, *, chunk):
        width = nl.idx.shape[-1]
        widths.append(width)
        return s, nl, width < 121

    stepper.md_chunk = chunk
    _, nl = stepper.advance(state, neighbor(100), chunk=1000)
    assert widths == [100, 110, 121]
    assert refreshed == [110, 121]
    assert requests == [10, 13, 11, 24]
    assert nl.idx.shape[-1] == 121


def test_mace_recovery_fails_closed_when_allocator_cannot_grow():
    stepper, state, requests, refreshed = setup(base=97, fixed=True, retries=2)
    stepper.md_chunk = lambda s, nl, *, chunk: (s, nl, True)
    with pytest.raises(RuntimeError, match='minimum capacity'):
        stepper.advance(state, neighbor(100), chunk=1000)
    assert len(requests) == 3
    assert refreshed == []


def test_classical_recovery_retains_original_capacity_policy():
    stepper, state, requests, refreshed = setup(mace=False)
    stepper.md_chunk = lambda s, nl, *, chunk: (s, nl, nl.idx.shape[-1] < 197)
    _, nl = stepper.advance(state, neighbor(100), chunk=1000)
    assert requests == [100]
    assert refreshed == [197]
    assert nl.idx.shape[-1] == 197


def test_mace_recovery_rejects_overflowing_allocation_even_at_target_width():
    stepper, state, requests, refreshed = setup()
    original_allocate = stepper.neighbor_fn.allocate

    def allocate(positions, *, extra_capacity):
        nl = original_allocate(positions, extra_capacity=extra_capacity)
        nl.did_buffer_overflow = extra_capacity <= 13
        return nl

    stepper.neighbor_fn.allocate = allocate
    stepper.md_chunk = lambda s, nl, *, chunk: (s, nl, nl.idx.shape[-1] < 110)
    _, nl = stepper.advance(state, neighbor(100), chunk=1000)
    assert requests == [10, 18]
    assert refreshed == [115]
    assert not nl.did_buffer_overflow


def test_mace_persistent_chunk_overflow_fails_closed():
    stepper, state, requests, refreshed = setup(retries=2)
    stepper.md_chunk = lambda s, nl, *, chunk: (s, nl, True)
    with pytest.raises(RuntimeError, match='overflow persisted'):
        stepper.advance(state, neighbor(100), chunk=1000)
    assert refreshed == [110, 121]
