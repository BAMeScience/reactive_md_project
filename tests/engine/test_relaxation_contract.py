"""CPU regression tests for relaxation precision and transactional retries.

Use the real JAX-MD Langevin integrator with a harmonic surrogate and a
JAX-compatible neighbor-list double; no MACE model or weights are required.
"""
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from reactive_md import relaxation


class Neighbor(NamedTuple):
    capacity: object
    updates: object
    did_buffer_overflow: object


class NeighborFunction:
    def __init__(self, *, initial_overflow=False, overflow_after=None,
                 persistent=False):
        self.initial_overflow = initial_overflow
        self.overflow_after = overflow_after
        self.persistent = persistent
        self.allocations = []

    def allocate(self, positions, *, extra_capacity=0):
        self.allocations.append(extra_capacity)
        return Neighbor(jnp.asarray(extra_capacity, dtype=jnp.int32),
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.asarray(self.initial_overflow and
                                    (self.persistent or extra_capacity < 8)))

    def update(self, positions, neighbor):
        updates = neighbor.updates + 1
        overflow = neighbor.did_buffer_overflow
        if self.overflow_after is not None:
            overflow = jnp.logical_or(
                overflow, (updates >= self.overflow_after) &
                (self.persistent | (neighbor.capacity < 8)))
        return neighbor._replace(updates=updates, did_buffer_overflow=overflow)


def harmonic_energy(positions, neighbor):
    assert isinstance(neighbor, Neighbor)
    return jnp.sum(positions ** 2) * 0.5


def forbidden(*args, **kwargs):
    raise AssertionError('Classical backend must not be used with overrides')


@pytest.fixture
def x64_enabled():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def run_relaxation(neighbor_fn, *, dtype=jnp.float32,
                   input_dtype=jnp.float32, override=True, **kwargs):
    _, shift = space.free()
    ff = SimpleNamespace(
        neighbor_fn=SimpleNamespace(allocate=forbidden, update=forbidden)
        if override else neighbor_fn,
        energy_fn=forbidden if override else
        lambda r, n: {'total': harmonic_energy(r, n)},
    )
    options = dict(ff=ff, shift_fn=shift,
                   masses=jnp.array([2., 3.], dtype=input_dtype),
                   temperature_k=300., kb_real=0.0019872041,
                   dt_ps=0.00025, relaxation_time_ps=0.0005,
                   thermostat_tau_ps=0.01)
    if override:
        options.update(md_energy_fn=harmonic_energy,
                       md_neighbor_fn=neighbor_fn)
    options.update(kwargs)
    return relaxation.langevin_relax_with_nlist(
        jax.random.PRNGKey(17),
        jnp.array([[1., 0., 0.], [0., 1., 0.]], dtype=dtype),
        jnp.full((2, 3), 0.1, dtype=input_dtype), **options)


@pytest.mark.parametrize('override', [False, True], ids=['classical', 'override'])
@pytest.mark.parametrize('dtype,input_dtype', [
    (jnp.float32, jnp.float64), (jnp.float64, jnp.float32),
], ids=['float32-coordinates', 'float64-coordinates'])
def test_mixed_precision_preserves_coordinate_dtype(x64_enabled, override,
                                                    dtype, input_dtype):
    neighbors = NeighborFunction()
    key, positions, velocities, nlist = run_relaxation(
        neighbors, dtype=dtype, input_dtype=input_dtype, override=override)
    assert positions.dtype == velocities.dtype == dtype
    assert np.isfinite(positions).all() and np.isfinite(velocities).all()
    assert not bool(nlist.did_buffer_overflow)
    assert neighbors.allocations == [0]
    assert not np.array_equal(key, jax.random.PRNGKey(17))


@pytest.mark.parametrize('initial_overflow,overflow_after', [
    (True, None), (False, 2), (False, 3),
], ids=['allocation', 'during-chunk', 'final-update'])
def test_overflow_retry_matches_uninterrupted_relaxation(initial_overflow,
                                                         overflow_after):
    baseline = run_relaxation(NeighborFunction())
    neighbors = NeighborFunction(initial_overflow=initial_overflow,
                                 overflow_after=overflow_after)
    recovered = run_relaxation(neighbors)
    assert neighbors.allocations == [0, 8]
    assert not bool(recovered[3].did_buffer_overflow)
    # Compare physical state as well as the returned key: replaying with an
    # advanced internal Langevin RNG would change the stochastic trajectory.
    np.testing.assert_array_equal(recovered[0], baseline[0])
    for actual, expected in zip(recovered[1:3], baseline[1:3]):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('initial_overflow,overflow_after,message', [
    (True, None, 'allocation overflowed'),
    (False, 3, 'overflow persisted'),
])
def test_persistent_overflow_fails_closed(monkeypatch, initial_overflow,
                                          overflow_after, message):
    monkeypatch.setattr(relaxation, '_MAX_NLIST_RETRIES', 2)
    neighbors = NeighborFunction(initial_overflow=initial_overflow,
                                 overflow_after=overflow_after, persistent=True)
    with pytest.raises(RuntimeError, match=message):
        run_relaxation(neighbors)
    assert neighbors.allocations == [0, 8, 16]


@pytest.mark.parametrize('options,message', [
    ({'dt_ps': 0.}, 'dt_ps must be positive'),
    ({'relaxation_time_ps': float('nan')}, 'relaxation_time_ps must be positive'),
    ({'thermostat_tau_ps': -1.}, 'thermostat_tau_ps must be positive'),
    ({'relaxation_time_ps': 0.0006}, 'integer multiple'),
    ({'md_energy_fn': None}, 'must be provided together'),
    ({'md_neighbor_fn': None}, 'must be provided together'),
])
def test_invalid_relaxation_parameters_fail_before_allocation(options, message):
    neighbors = NeighborFunction()
    with pytest.raises(ValueError, match=message):
        run_relaxation(neighbors, **options)
    assert neighbors.allocations == []


def test_relaxation_performance_preserves_retry_and_state(monkeypatch, capsys):
    import json
    from reactive_md.performance import PerformanceTimer

    monkeypatch.setenv('REACTIVE_MD_PERFORMANCE', '0')
    baseline = run_relaxation(NeighborFunction(overflow_after=2))
    assert '[REACTIVE_MD PERF]' not in capsys.readouterr().out
    monkeypatch.setenv('REACTIVE_MD_PERFORMANCE', '1')
    with PerformanceTimer().measure('trial_relaxation', 10):
        measured = run_relaxation(NeighborFunction(overflow_after=2))
    prefix = '[REACTIVE_MD PERF] '
    rows = [json.loads(line[len(prefix):]) for line in capsys.readouterr().out.splitlines()
            if line.startswith(prefix)]
    np.testing.assert_array_equal(measured[0], baseline[0])
    for actual, expected in zip(measured[1:3], baseline[1:3]):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert all(r['step'] == 10 and r['status'] == 'ok' for r in rows)
    assert [r['attempt'] for r in rows if r['stage'] == 'langevin_chunk_attempt'] == [0, 1]
    assert sum(r['stage'] == 'langevin_overflow_recovery' for r in rows) == 1
    assert [r['chunk_steps'] for r in rows if r['stage'] == 'langevin_chunk'] == [2]
