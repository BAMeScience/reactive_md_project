# tests/test_a2_rate.py

import numpy as np
import jax
import jax.numpy as jnp

from jax_md import space

from reactive_md.forcefield import build_forcefield
from reactive_md.reaction import SystemState
from reactive_md.reactions.a2 import (
    A2AssociationCandidate,
    A2Reaction,
)
from reactive_md.reactions.a2_rate import (
    a2_rate_reactive_cycle,
)


def make_empty_system(n_atoms=4):
    empty_bonds = (
        jnp.empty((0, 2), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
    )

    empty_angles = (
        jnp.empty((0, 3), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.float32),
    )

    empty_four_body = (
        jnp.empty((0, 4), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
        jnp.empty((0,), dtype=jnp.int32),
        jnp.empty((0,), dtype=jnp.float32),
    )

    return SystemState(
        bonds=empty_bonds,
        angles=empty_angles,
        torsions=empty_four_body,
        impropers=empty_four_body,
        charges=jnp.zeros(
            (n_atoms,),
            dtype=jnp.float32,
        ),
        sigmas=jnp.full(
            (n_atoms,),
            3.4,
            dtype=jnp.float32,
        ),
        epsilons=jnp.full(
            (n_atoms,),
            1.9572 / 4.184,
            dtype=jnp.float32,
        ),
        molecule_id=jnp.arange(
            n_atoms,
            dtype=jnp.int32,
        ),
        pf6_reacted=jnp.zeros(
            (0,),
            dtype=jnp.bool_,
        ),
    )


def make_a2_forcefield(
    R,
    box,
    system,
):
    return build_forcefield(
        R=R,
        box=box,
        bond_idx=np.asarray(
            system.bonds[0]
        ),
        k_b=np.asarray(
            system.bonds[1]
        ),
        r0=np.asarray(
            system.bonds[2]
        ),
        angle_idx=np.asarray(
            system.angles[0]
        ),
        k_theta=np.asarray(
            system.angles[1]
        ),
        theta0=np.asarray(
            system.angles[2]
        ),
        torsions=tuple(
            np.asarray(x)
            for x in system.torsions
        ),
        impropers=tuple(
            np.asarray(x)
            for x in system.impropers
        ),
        charges=np.asarray(
            system.charges
        ),
        sigmas=np.asarray(
            system.sigmas
        ),
        epsilons=np.asarray(
            system.epsilons
        ),
        molecule_id=np.asarray(
            system.molecule_id,
            dtype=np.int32,
        ),
        r_cut=9.0,
        dr_threshold=0.5,
    )


def make_common_dynamics(R, box):
    disp_fn, shift_fn = space.periodic(
        box
    )

    velocities = jnp.zeros_like(
        R
    )

    masses = jnp.ones(
        (R.shape[0],),
        dtype=jnp.float32,
    )

    return (
        disp_fn,
        shift_fn,
        velocities,
        masses,
    )


def test_a2_rate_association_probability_one_adds_bond_and_relaxes():
    system = make_empty_system(
        n_atoms=4
    )

    reaction = A2Reaction(
        a_indices=np.array(
            [0, 1, 2],
            dtype=np.int32,
        ),
        k_bond_jax=29.8757,
        r0_bond=1.5,
        association_cutoff=2.7,
    )

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    box = jnp.array(
        [20.0, 20.0, 20.0],
        dtype=jnp.float32,
    )

    (
        disp_fn,
        shift_fn,
        velocities,
        masses,
    ) = make_common_dynamics(
        R,
        box,
    )

    ff = make_a2_forcefield(
        R,
        box,
        system,
    )

    rng = np.random.default_rng(
        0
    )

    result = a2_rate_reactive_cycle(
        key=jax.random.PRNGKey(0),
        R=R,
        velocities=velocities,
        box=box,
        system=system,
        ff=ff,
        reaction=reaction,
        disp_fn=disp_fn,
        shift_fn=shift_fn,
        masses=masses,
        association_rate_ps=1.0,
        dissociation_rate_ps=0.0,
        reactive_interval_ps=1.0,
        temperature_k=188.0,
        kb_real=0.0019872041,
        rng=rng,
        relaxation_time_ps=0.002,
    )

    bond_idx = np.asarray(
        result.system.bonds[0]
    )

    assert len(
        result.events
    ) == 1

    assert (
        result.events[0].reaction_type
        == "association"
    )

    np.testing.assert_array_equal(
        bond_idx,
        np.array(
            [[0, 1]],
            dtype=np.int32,
        ),
    )

    assert jnp.all(
        jnp.isfinite(
            result.positions
        )
    )

    assert jnp.all(
        jnp.isfinite(
            result.velocities
        )
    )

    assert result.ff is not ff
    assert result.ff.nlist is not None


def test_a2_rate_association_probability_zero_adds_no_bond():
    system = make_empty_system(
        n_atoms=4
    )

    reaction = A2Reaction(
        a_indices=np.array(
            [0, 1, 2],
            dtype=np.int32,
        ),
        k_bond_jax=29.8757,
        r0_bond=1.5,
        association_cutoff=2.7,
    )

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    box = jnp.array(
        [20.0, 20.0, 20.0],
        dtype=jnp.float32,
    )

    (
        disp_fn,
        shift_fn,
        velocities,
        masses,
    ) = make_common_dynamics(
        R,
        box,
    )

    ff = make_a2_forcefield(
        R,
        box,
        system,
    )

    rng = np.random.default_rng(
        0
    )

    result = a2_rate_reactive_cycle(
        key=jax.random.PRNGKey(0),
        R=R,
        velocities=velocities,
        box=box,
        system=system,
        ff=ff,
        reaction=reaction,
        disp_fn=disp_fn,
        shift_fn=shift_fn,
        masses=masses,
        association_rate_ps=0.0,
        dissociation_rate_ps=0.0,
        reactive_interval_ps=1.0,
        temperature_k=188.0,
        kb_real=0.0019872041,
        rng=rng,
        relaxation_time_ps=0.002,
    )

    assert len(
        result.events
    ) == 0

    assert (
        result.system.bonds[0].shape
        == (0, 2)
    )

    assert result.ff is ff

    np.testing.assert_allclose(
        np.asarray(
            result.positions
        ),
        np.asarray(R),
    )

    np.testing.assert_allclose(
        np.asarray(
            result.velocities
        ),
        np.asarray(
            velocities
        ),
    )


def test_a2_rate_dissociation_probability_one_removes_bond_and_relaxes():
    system = make_empty_system(
        n_atoms=4
    )

    reaction = A2Reaction(
        a_indices=np.array(
            [0, 1, 2],
            dtype=np.int32,
        ),
        k_bond_jax=29.8757,
        r0_bond=1.5,
        association_cutoff=2.7,
    )

    association_candidate = (
        A2AssociationCandidate(
            i=0,
            j=1,
            distance=1.5,
        )
    )

    trial = (
        reaction.build_association_trial(
            system,
            association_candidate,
        )
    )

    system_with_bond = SystemState(
        bonds=tuple(
            jnp.asarray(x)
            for x in trial["bonds"]
        ),
        angles=tuple(
            jnp.asarray(x)
            for x in trial["angles"]
        ),
        torsions=tuple(
            jnp.asarray(x)
            for x in trial["torsions"]
        ),
        impropers=tuple(
            jnp.asarray(x)
            for x in trial["impropers"]
        ),
        charges=jnp.asarray(
            trial["charges"]
        ),
        sigmas=jnp.asarray(
            trial["sigmas"]
        ),
        epsilons=jnp.asarray(
            trial["epsilons"]
        ),
        molecule_id=jnp.asarray(
            trial["molecule_id"],
            dtype=jnp.int32,
        ),
        pf6_reacted=jnp.zeros(
            (0,),
            dtype=jnp.bool_,
        ),
    )

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    box = jnp.array(
        [20.0, 20.0, 20.0],
        dtype=jnp.float32,
    )

    (
        disp_fn,
        shift_fn,
        velocities,
        masses,
    ) = make_common_dynamics(
        R,
        box,
    )

    ff = make_a2_forcefield(
        R,
        box,
        system_with_bond,
    )

    rng = np.random.default_rng(
        0
    )

    result = a2_rate_reactive_cycle(
        key=jax.random.PRNGKey(0),
        R=R,
        velocities=velocities,
        box=box,
        system=system_with_bond,
        ff=ff,
        reaction=reaction,
        disp_fn=disp_fn,
        shift_fn=shift_fn,
        masses=masses,
        association_rate_ps=0.0,
        dissociation_rate_ps=1.0,
        reactive_interval_ps=1.0,
        temperature_k=188.0,
        kb_real=0.0019872041,
        rng=rng,
        relaxation_time_ps=0.002,
    )

    assert len(
        result.events
    ) == 1

    assert (
        result.events[0].reaction_type
        == "dissociation"
    )

    assert (
        result.system.bonds[0].shape
        == (0, 2)
    )

    assert jnp.all(
        jnp.isfinite(
            result.positions
        )
    )

    assert jnp.all(
        jnp.isfinite(
            result.velocities
        )
    )

    assert result.ff is not ff
    assert result.ff.nlist is not None


def test_a2_rate_rejects_association_probability_greater_than_one():
    system = make_empty_system(
        n_atoms=2
    )

    reaction = A2Reaction(
        a_indices=np.array(
            [0, 1],
            dtype=np.int32,
        ),
        k_bond_jax=29.8757,
        r0_bond=1.5,
        association_cutoff=2.7,
    )

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

    (
        disp_fn,
        shift_fn,
        velocities,
        masses,
    ) = make_common_dynamics(
        R,
        box,
    )

    ff = make_a2_forcefield(
        R,
        box,
        system,
    )

    rng = np.random.default_rng(
        0
    )

    try:
        a2_rate_reactive_cycle(
            key=jax.random.PRNGKey(0),
            R=R,
            velocities=velocities,
            box=box,
            system=system,
            ff=ff,
            reaction=reaction,
            disp_fn=disp_fn,
            shift_fn=shift_fn,
            masses=masses,
            association_rate_ps=2.0,
            dissociation_rate_ps=0.0,
            reactive_interval_ps=1.0,
            temperature_k=188.0,
            kb_real=0.0019872041,
            rng=rng,
            relaxation_time_ps=0.002,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected p_assoc > 1 to raise ValueError."
        )


def test_a2_rate_rejects_dissociation_probability_greater_than_one():
    system = make_empty_system(
        n_atoms=2
    )

    reaction = A2Reaction(
        a_indices=np.array(
            [0, 1],
            dtype=np.int32,
        ),
        k_bond_jax=29.8757,
        r0_bond=1.5,
        association_cutoff=2.7,
    )

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

    (
        disp_fn,
        shift_fn,
        velocities,
        masses,
    ) = make_common_dynamics(
        R,
        box,
    )

    ff = make_a2_forcefield(
        R,
        box,
        system,
    )

    rng = np.random.default_rng(
        0
    )

    try:
        a2_rate_reactive_cycle(
            key=jax.random.PRNGKey(0),
            R=R,
            velocities=velocities,
            box=box,
            system=system,
            ff=ff,
            reaction=reaction,
            disp_fn=disp_fn,
            shift_fn=shift_fn,
            masses=masses,
            association_rate_ps=0.0,
            dissociation_rate_ps=2.0,
            reactive_interval_ps=1.0,
            temperature_k=188.0,
            kb_real=0.0019872041,
            rng=rng,
            relaxation_time_ps=0.002,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected p_diss > 1 to raise ValueError."
        )
