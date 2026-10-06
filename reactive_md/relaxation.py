# reactive_md/relaxation.py

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax_md import simulate


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
):
    """Relax a post-reaction structure using short Langevin dynamics.

    Parameters
    ----------
    key
        JAX PRNG key.

    R
        Initial post-reaction positions.

    velocities
        Velocities from the MD state immediately before the
        topology-changing reaction.

    ff
        Force field corresponding to the post-reaction topology.

    shift_fn
        JAX-MD position shift function.

    masses
        Particle masses.

    temperature_k
        Relaxation temperature in K.

    kb_real
        Boltzmann constant in the force-field unit system.

    dt_ps
        Relaxation timestep in ps. The rs@MD benchmark uses
        0.25 fs = 0.00025 ps.

    relaxation_time_ps
        Total relaxation duration. The benchmark uses 1 ps.

    thermostat_tau_ps
        Langevin thermostat time scale. The benchmark uses
        0.01 ps.

    Returns
    -------
    key
        Updated PRNG key.

    R_relaxed
        Relaxed positions.

    velocities_relaxed
        Velocities after relaxation.

    nlist_relaxed
        Updated neighbor list.
    """

    dt_ps = float(dt_ps)
    relaxation_time_ps = float(relaxation_time_ps)
    thermostat_tau_ps = float(thermostat_tau_ps)

    if dt_ps <= 0.0:
        raise ValueError(
            "dt_ps must be positive."
        )

    if relaxation_time_ps <= 0.0:
        raise ValueError(
            "relaxation_time_ps must be positive."
        )

    if thermostat_tau_ps <= 0.0:
        raise ValueError(
            "thermostat_tau_ps must be positive."
        )

    # ---------------------------------------------------------
    # Number of integration steps.
    # ---------------------------------------------------------

    n_steps_float = (
        relaxation_time_ps
        / dt_ps
    )

    n_steps = int(
        round(n_steps_float)
    )

    if not bool(
        jnp.isclose(
            n_steps * dt_ps,
            relaxation_time_ps,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError(
            "relaxation_time_ps must be an integer multiple "
            "of dt_ps."
        )

    # ---------------------------------------------------------
    # Thermodynamic parameters.
    # ---------------------------------------------------------

    kT = (
        float(kb_real)
        * float(temperature_k)
    )

    mass = jnp.asarray(
        masses
    )

    # JAX-MD Langevin uses gamma rather than tau.
    #
    # gamma = 1 / tau
    gamma = (
        1.0
        / thermostat_tau_ps
    )

    # ---------------------------------------------------------
    # Scalar energy wrapper.
    # ---------------------------------------------------------

    def energy_scalar(
        positions,
        neighbor,
    ):
        return ff.energy_fn(
            positions,
            neighbor,
        )["total"]

    # ---------------------------------------------------------
    # Langevin integrator.
    # ---------------------------------------------------------

    init_fn, apply_fn = (
        simulate.nvt_langevin(
            energy_scalar,
            shift_fn,
            dt=dt_ps,
            kT=kT,
            gamma=gamma,
            mass=mass,
        )
    )

    # ---------------------------------------------------------
    # Allocate neighbor list for the post-reaction force field.
    # ---------------------------------------------------------

    nlist = ff.neighbor_fn.allocate(
        R
    )

    # ---------------------------------------------------------
    # Initialize Langevin state.
    #
    # The PRNG key is supplied at initialization. In the installed
    # JAX-MD version, the random state is then carried internally
    # by NVTLangevinState, so apply_fn does not take a new key.
    # ---------------------------------------------------------

    key, sub = jax.random.split(
        key
    )

    state = init_fn(
        sub,
        R,
        neighbor=nlist,
    )

    # ---------------------------------------------------------
    # Preserve incoming velocities.
    #
    # NVTLangevinState stores momentum rather than velocity:
    #
    #     p = m * v
    # ---------------------------------------------------------

    velocities = jnp.asarray(
        velocities
    )

    momentum = (
        velocities
        * mass[:, None]
    )

    state = replace(
        state,
        momentum=momentum,
    )

    # ---------------------------------------------------------
    # JIT-compiled relaxation chunk.
    # ---------------------------------------------------------

    @partial(
        jax.jit,
        static_argnames=("n_steps",),
    )
    def run_relaxation(
        state,
        nlist,
        *,
        n_steps,
    ):
        def body_fn(
            carry,
            _,
        ):
            state, nlist = carry

            # Update neighbor list at the current positions.
            nlist = ff.neighbor_fn.update(
                state.position,
                nlist,
            )

            # For this JAX-MD version, apply_fn expects the
            # state as its only positional argument.
            state = apply_fn(
                state,
                neighbor=nlist,
            )

            return (
                state,
                nlist,
            ), None

        (
            state_out,
            nlist_out,
        ), _ = jax.lax.scan(
            body_fn,
            (
                state,
                nlist,
            ),
            xs=None,
            length=n_steps,
        )

        return (
            state_out,
            nlist_out,
        )

    # ---------------------------------------------------------
    # Run relaxation.
    # ---------------------------------------------------------

    state, nlist = (
        run_relaxation(
            state,
            nlist,
            n_steps=n_steps,
        )
    )

    # ---------------------------------------------------------
    # Convert momentum back to velocity.
    # ---------------------------------------------------------

    velocities_out = (
        state.momentum
        / mass[:, None]
    )

    return (
        key,
        state.position,
        velocities_out,
        nlist,
    )
