# tests/test_relaxation.py

import numpy as np
import jax
import jax.numpy as jnp

from jax_md import space

from reactive_md.forcefield import build_forcefield
from reactive_md.relaxation import langevin_relax_with_nlist


def test_langevin_relax_with_nlist_runs_and_returns_finite_state():
    # ---------------------------------------------------------
    # Simple bonded A2 pair.
    #
    # Start away from the equilibrium bond length so that
    # non-zero forces are present.
    # ---------------------------------------------------------

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    box = jnp.array(
        [20.0, 20.0, 20.0],
        dtype=jnp.float32,
    )

    _, shift_fn = space.periodic(box)

    # ---------------------------------------------------------
    # rs@MD benchmark bond parameters.
    #
    # Paper:
    #
    #   V = 0.5 * k_paper * (r-r0)^2
    #
    # JAX-MD OPLS-AA:
    #
    #   V = k_jax * (r-r0)^2
    #
    # Therefore k_jax = 0.5 * k_paper after unit conversion.
    # ---------------------------------------------------------

    k_bond_paper = (
        25000.0
        / 100.0
        / 4.184
    )

    k_bond_jax = (
        0.5
        * k_bond_paper
    )

    r0_bond = 1.5

    bonds = (
        np.array(
            [[0, 1]],
            dtype=np.int32,
        ),
        np.array(
            [k_bond_jax],
            dtype=np.float32,
        ),
        np.array(
            [r0_bond],
            dtype=np.float32,
        ),
    )

    angles = (
        np.zeros(
            (0, 3),
            dtype=np.int32,
        ),
        np.zeros(
            (0,),
            dtype=np.float32,
        ),
        np.zeros(
            (0,),
            dtype=np.float32,
        ),
    )

    torsions = (
        np.zeros(
            (0, 4),
            dtype=np.int32,
        ),
        np.zeros(
            (0,),
            dtype=np.float32,
        ),
        np.zeros(
            (0,),
            dtype=np.int32,
        ),
        np.zeros(
            (0,),
            dtype=np.float32,
        ),
    )

    impropers = (
        np.zeros(
            (0, 4),
            dtype=np.int32,
        ),
        np.zeros(
            (0,),
            dtype=np.float32,
        ),
        np.zeros(
            (0,),
            dtype=np.int32,
        ),
        np.zeros(
            (0,),
            dtype=np.float32,
        ),
    )

    charges = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    sigma = 3.4

    sigmas = np.array(
        [sigma, sigma],
        dtype=np.float32,
    )

    epsilon = (
        1.9572
        / 4.184
    )

    epsilons = np.array(
        [epsilon, epsilon],
        dtype=np.float32,
    )

    molecule_id = np.array(
        [0, 0],
        dtype=np.int32,
    )

    ff = build_forcefield(
        R=R,
        box=box,
        bond_idx=bonds[0],
        k_b=bonds[1],
        r0=bonds[2],
        angle_idx=angles[0],
        k_theta=angles[1],
        theta0=angles[2],
        torsions=torsions,
        impropers=impropers,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        molecule_id=molecule_id,
        r_cut=9.0,
        dr_threshold=0.5,
    )

    # ---------------------------------------------------------
    # Give the system a small non-zero initial velocity.
    # ---------------------------------------------------------

    velocities = jnp.array(
        [
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    masses = jnp.array(
        [1.0, 1.0],
        dtype=jnp.float32,
    )

    key = jax.random.PRNGKey(0)

    # ---------------------------------------------------------
    # Use the benchmark timestep and thermostat coupling,
    # but only a very short total relaxation here so the unit
    # test stays fast.
    #
    # 0.002 ps / 0.00025 ps = 8 integration steps.
    # ---------------------------------------------------------

    (
        key_new,
        R_relaxed,
        velocities_relaxed,
        nlist_relaxed,
    ) = langevin_relax_with_nlist(
        key,
        R,
        velocities,
        ff=ff,
        shift_fn=shift_fn,
        masses=masses,
        temperature_k=188.0,
        kb_real=0.0019872041,
        dt_ps=0.00025,
        relaxation_time_ps=0.002,
        thermostat_tau_ps=0.01,
    )

    # ---------------------------------------------------------
    # Basic validity checks.
    # ---------------------------------------------------------

    assert R_relaxed.shape == R.shape

    assert (
        velocities_relaxed.shape
        == velocities.shape
    )

    assert jnp.all(
        jnp.isfinite(R_relaxed)
    )

    assert jnp.all(
        jnp.isfinite(
            velocities_relaxed
        )
    )

    assert nlist_relaxed is not None

    # The PRNG state should have advanced.
    assert not np.array_equal(
        np.asarray(key_new),
        np.asarray(key),
    )

    # The dynamics should not be a complete no-op.
    assert not jnp.allclose(
        R_relaxed,
        R,
    )
