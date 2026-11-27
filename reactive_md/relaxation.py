"""Post-reaction Langevin relaxation with recoverable neighbor-list overflow.

Neighbor-list storage is allocated outside JIT. An MD chunk is committed only
if no neighbor-list overflow occurred at any step or at its final coordinates.
If overflow occurs, the entire attempted chunk (including its RNG state) is
discarded, the neighbor list is enlarged, and the chunk is retried.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate
from .performance import PerformanceTimer


# Short chunks limit how much work must be replayed after an overflow.
_RELAXATION_CHUNK_STEPS = 50
_MAX_NLIST_RETRIES = 12


def langevin_relax_with_nlist(
    key,
    R,
    velocities,
    *,
    ff,
    shift_fn,
    masses,
    temperature_k: float,
    kb_real: float,
    dt_ps: float = 0.00025,
    relaxation_time_ps: float = 1.0,
    thermostat_tau_ps: float = 0.01,
    md_energy_fn=None,
    md_neighbor_fn=None,
):
    """Relax post-reaction positions using JAX-MD Langevin dynamics.

    Returns (updated_key, positions, velocities, neighbor_list).

    No reaction parameters, force-field parameters, or time step are modified.
    Neighbor-list capacity grows adaptively on overflow. An overflowing chunk
    is never accepted, and its random-number state is rolled back as well.

    When both optional MACE overrides are supplied, the returned neighbor list
    belongs to the override, not to ``ff``.
    """
    if (md_energy_fn is None) != (md_neighbor_fn is None):
        raise ValueError(
            "md_energy_fn and md_neighbor_fn must be provided together"
        )
    perf = PerformanceTimer()
    neighbor_fn = md_neighbor_fn if md_neighbor_fn is not None else ff.neighbor_fn

    dt_ps = float(dt_ps)
    relaxation_time_ps = float(relaxation_time_ps)
    thermostat_tau_ps = float(thermostat_tau_ps)

    if not np.isfinite(dt_ps) or dt_ps <= 0.0:
        raise ValueError("dt_ps must be positive and finite.")
    if not np.isfinite(relaxation_time_ps) or relaxation_time_ps <= 0.0:
        raise ValueError("relaxation_time_ps must be positive and finite.")
    if not np.isfinite(thermostat_tau_ps) or thermostat_tau_ps <= 0.0:
        raise ValueError("thermostat_tau_ps must be positive and finite.")

    n_steps = int(round(relaxation_time_ps / dt_ps))
    if n_steps < 1 or not np.isclose(
        n_steps * dt_ps, relaxation_time_ps, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "relaxation_time_ps must be an integer multiple of dt_ps."
        )

    # Keep all Langevin state and integrator parameters in the
    # same precision as the incoming coordinates.
    R = jnp.asarray(R)
    dtype = R.dtype
    mass = jnp.asarray(masses, dtype=dtype)
    velocities = jnp.asarray(velocities, dtype=dtype)

    dt = jnp.asarray(dt_ps, dtype=dtype)
    kT = jnp.asarray(
        float(kb_real) * float(temperature_k), dtype=dtype
    )
    gamma = jnp.asarray(
        1.0 / thermostat_tau_ps, dtype=dtype
    )

    def energy_scalar(positions, neighbor):
        if md_energy_fn is not None:
            return md_energy_fn(positions, neighbor)
        return ff.energy_fn(positions, neighbor)["total"]

    init_fn, apply_fn = simulate.nvt_langevin(
        energy_scalar,
        shift_fn,
        dt=dt,
        kT=kT,
        gamma=gamma,
    )

    def force_fn(positions, neighbor):
        return -jax.grad(energy_scalar)(positions, neighbor=neighbor)

    # The initializer evaluates a force before the compiled Langevin chunk.
    # Compile it and the overflow-recovery force for the MACE backend too.
    if md_energy_fn is not None:
        init_fn = jax.jit(init_fn)
        force_fn = jax.jit(force_fn)

    def allocate_valid_nlist(positions, *, extra_capacity=0):
        """Allocate outside JIT; grow storage until no overflow is reported."""
        for attempt in range(_MAX_NLIST_RETRIES + 1):
            with perf.measure(
                'langevin_neighbor_allocate', attempt=attempt, extra_capacity=extra_capacity,
            ) as wait:
                nl = neighbor_fn.allocate(
                    positions, extra_capacity=extra_capacity
                )
                wait(nl)
            if not bool(np.asarray(nl.did_buffer_overflow)):
                return nl
            if attempt == _MAX_NLIST_RETRIES:
                break
            extra_capacity = max(8, 2 * extra_capacity)
        raise RuntimeError(
            "Langevin neighbor-list allocation overflowed despite "
            "adaptive capacity growth."
        )

    # A valid list is required before the Langevin initializer computes its
    # initial force. Do not initialize with an overflowing list.
    with perf.measure('langevin_initialize') as wait:
        nlist = allocate_valid_nlist(R)
        key, sub = jax.random.split(key)
        state = init_fn(sub, R, mass=mass, neighbor=nlist)
        state = replace(
            state,
            momentum=velocities * mass[:, None],
        )
        wait(state, nlist, key)

    @partial(jax.jit, static_argnames=("chunk",))
    def run_chunk(state, nlist, *, chunk):
        def body(carry, _):
            current_state, current_nlist, overflow_seen = carry
            current_nlist = neighbor_fn.update(
                current_state.position, current_nlist
            )
            overflow_seen = jnp.logical_or(
                overflow_seen, current_nlist.did_buffer_overflow
            )
            current_state = apply_fn(current_state, neighbor=current_nlist)
            return (current_state, current_nlist, overflow_seen), None

        (trial_state, trial_nlist, overflow_seen), _ = jax.lax.scan(
            body,
            (state, nlist, jnp.asarray(False)),
            xs=None,
            length=chunk,
        )
        # The final integrator step moves the particles after the preceding
        # list update. Validate the list at the final coordinates too.
        trial_nlist = neighbor_fn.update(
            trial_state.position, trial_nlist
        )
        overflow_seen = jnp.logical_or(
            overflow_seen, trial_nlist.did_buffer_overflow
        )
        return trial_state, trial_nlist, overflow_seen

    def advance_chunk(state, nlist, *, chunk):
        if bool(np.asarray(nlist.did_buffer_overflow)):
            raise RuntimeError(
                "Langevin chunk started with an overflowing neighbor list."
            )

        extra_capacity = 8
        for attempt in range(_MAX_NLIST_RETRIES + 1):
            with perf.measure(
                'langevin_chunk_attempt', attempt=attempt, chunk_steps=chunk,
            ) as wait:
                trial_state, trial_nlist, overflow_seen = run_chunk(
                    state, nlist, chunk=chunk
                )
                wait(trial_state, trial_nlist, overflow_seen)
            if not bool(np.asarray(overflow_seen)):
                # Only a fully validated chunk is accepted.
                if not bool(np.asarray(jnp.all(jnp.isfinite(trial_state.position)))):
                    raise FloatingPointError(
                        "Nonfinite positions in accepted Langevin chunk."
                    )
                if not bool(np.asarray(jnp.all(jnp.isfinite(trial_state.momentum)))):
                    raise FloatingPointError(
                        "Nonfinite momenta in accepted Langevin chunk."
                    )
                return trial_state, trial_nlist

            if attempt == _MAX_NLIST_RETRIES:
                raise RuntimeError(
                    f"Langevin neighbor-list overflow persisted after "
                    f"{_MAX_NLIST_RETRIES} retries in a {chunk}-step chunk."
                )

            # Discard trial_state AND its advanced Langevin RNG. Rebuild at
            # the last accepted coordinates, then retry the identical chunk.
            with perf.measure(
                'langevin_overflow_recovery', attempt=attempt, chunk_steps=chunk,
            ) as wait:
                nlist = allocate_valid_nlist(
                    state.position, extra_capacity=extra_capacity
                )
                extra_capacity *= 2

                # JAX-MD Langevin carries a cached force. Recompute it against
                # the replacement list before replaying the chunk.
                if hasattr(state, "force"):
                    state = replace(
                        state,
                        force=force_fn(state.position, nlist),
                    )
                wait(state, nlist)

        raise AssertionError("Unreachable Langevin recovery state")

    steps_done = 0
    while steps_done < n_steps:
        chunk = min(_RELAXATION_CHUNK_STEPS, n_steps - steps_done)
        with perf.measure(
            'langevin_chunk', chunk_steps=chunk, relaxation_step=steps_done + chunk,
        ) as wait:
            state, nlist = advance_chunk(state, nlist, chunk=chunk)
            wait(state, nlist)
        steps_done += chunk

    return (
        key,
        state.position,
        state.momentum / mass[:, None],
        nlist,
    )

