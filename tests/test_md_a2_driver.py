# tests/test_md_a2_driver.py

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from reactive_md.config import SimConfig
from reactive_md.forcefield import build_forcefield
from reactive_md.md_a2_driver import run_md_nvt_a2
from reactive_md.reaction import SystemState
from reactive_md.reactions.a2 import A2Reaction


BOX = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
KB_REAL = 0.0019872041
BOND_K_JAX = 0.5 * (25000.0 / 100.0 / 4.184)


def make_system():
    empty_four_body = (
        jnp.empty((0, 4), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
    )
    return SystemState(
        bonds=(
            jnp.empty((0, 2), dtype=jnp.int32),
            jnp.empty((0,), dtype=jnp.float32),
            jnp.empty((0,), dtype=jnp.float32),
        ),
        angles=(
            jnp.empty((0, 3), dtype=jnp.int32),
            jnp.empty((0,), dtype=jnp.float32),
            jnp.empty((0,), dtype=jnp.float32),
        ),
        torsions=empty_four_body,
        impropers=empty_four_body,
        charges=jnp.zeros((2,), dtype=jnp.float32),
        sigmas=jnp.full((2,), 3.4, dtype=jnp.float32),
        epsilons=jnp.full((2,), 1.9572 / 4.184, dtype=jnp.float32),
        molecule_id=jnp.array([0, 1], dtype=jnp.int32),
        pf6_reacted=jnp.zeros((0,), dtype=jnp.bool_),
    )


def make_forcefield(R, system):
    return build_forcefield(
        R=R,
        box=BOX,
        bond_idx=np.asarray(system.bonds[0]),
        k_b=np.asarray(system.bonds[1]),
        r0=np.asarray(system.bonds[2]),
        angle_idx=np.asarray(system.angles[0]),
        k_theta=np.asarray(system.angles[1]),
        theta0=np.asarray(system.angles[2]),
        torsions=tuple(np.asarray(x) for x in system.torsions),
        impropers=tuple(np.asarray(x) for x in system.impropers),
        charges=np.asarray(system.charges),
        sigmas=np.asarray(system.sigmas),
        epsilons=np.asarray(system.epsilons),
        molecule_id=np.asarray(system.molecule_id),
        r_cut=9.0,
        dr_threshold=0.5,
    )


def make_reaction():
    return A2Reaction(
        a_indices=np.array([0, 1], dtype=np.int32),
        k_bond_jax=BOND_K_JAX,
        r0_bond=1.5,
        association_cutoff=2.7,
    )


def run_two_atom_md(*, distance, association_rate_ps):
    R = jnp.array(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    _, shift_fn = space.periodic(BOX)
    system = make_system()
    ff = make_forcefield(R, system)
    cfg = SimConfig(
        steps=2,
        check_every=1,
        max_events=2,
        dt=0.00025,
        temperature_k=188.0,
        kb_real=KB_REAL,
    )
    result = run_md_nvt_a2(
        jax.random.PRNGKey(0),
        cfg=cfg,
        init_positions=R,
        masses=jnp.array([39.948, 39.948], dtype=jnp.float32),
        box=BOX,
        shift_fn=shift_fn,
        ff=ff,
        sys=system,
        reaction=make_reaction(),
        association_rate_ps=association_rate_ps,
        dissociation_rate_ps=0.0,
        rng=np.random.default_rng(0),
        relaxation_time_ps=0.002,  # 8 steps, not the production 1 ps
    )
    return result, ff


def test_a2_driver_no_reaction_runs_two_md_steps():
    result, ff_initial = run_two_atom_md(
        distance=5.0,
        association_rate_ps=0.0,
    )
    assert result.steps_done == 2
    assert result.accepted_events == 0
    assert len(result.event_history) == 0
    assert result.population_history == ((0, 0), (1, 0), (2, 0))
    assert result.ff is ff_initial
    assert np.all(np.isfinite(np.asarray(result.final_md_state.position)))
    assert np.all(np.isfinite(np.asarray(result.final_md_state.velocity)))


def test_a2_driver_association_relaxes_and_resumes_md():
    # tau = 0.00025 ps, so k = 4000 ps^-1 gives deterministic p = 1.
    # This artificial rate is used only for the integration test.
    result, ff_initial = run_two_atom_md(
        distance=2.65,
        association_rate_ps=4000.0,
    )
    assert result.steps_done == 2  # MD resumed after the event at step 1
    assert result.accepted_events == 1
    assert len(result.event_history) == 1
    assert result.event_history[0].step == 1
    assert result.event_history[0].event.reaction_type == "association"
    np.testing.assert_array_equal(
        np.asarray(result.sys.bonds[0]),
        np.array([[0, 1]], dtype=np.int32),
    )
    assert result.population_history == ((0, 0), (1, 1), (2, 1))
    assert result.ff is not ff_initial
    assert result.ff.nlist is not None
    assert np.all(np.isfinite(np.asarray(result.final_md_state.position)))
    assert np.all(np.isfinite(np.asarray(result.final_md_state.velocity)))
    assert np.all(np.isfinite(np.asarray(result.final_md_state.force)))

