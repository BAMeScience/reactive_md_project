"""Legacy LiPF6 public entry-point test with a JAX-compatible dummy neighbor list.

The NVT engine needs a real neighbor-list *interface* (allocate with optional
capacity, update, idx and overflow flag), even if no pair forces are present.
The earlier stub returned None and was not a valid NVT engine input.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from reactive_md.config import SimConfig
from reactive_md.forcefield import FFBundle
from reactive_md.md_driver import run_md_nvt_with_reactions
from reactive_md.reaction import SystemState


class DummyNeighbor(NamedTuple):
    idx: object
    did_buffer_overflow: object


class DummyNeighborFn:
    def allocate(self, R, *, extra_capacity=0):
        return DummyNeighbor(
            idx=jnp.zeros((len(R), 1 + int(extra_capacity)), dtype=jnp.int32),
            did_buffer_overflow=jnp.asarray(False),
        )

    def update(self, R, nlist):
        return nlist


def test_md_driver_runs_with_dummy_forcefield():
    # Differentiable constant-zero potential, so the real Nose-Hoover
    # integrator is exercised without chemistry or pair interactions.
    def energy_fn(R, neighbor):
        return {"total": jnp.sum(0.0 * R)}

    ff = FFBundle(
        topo=None, params=None, energy_fn=energy_fn,
        neighbor_fn=DummyNeighborFn(), disp_fn=lambda a, b: a - b,
        nlist=None, coulomb_handler=None,
        nb_options=type("NB", (), {"r_cut": 8.0, "dr_threshold": 0.5})(),
    )
    empty_bonds = (np.zeros((0, 2), np.int32),
                   np.zeros(0, np.float32), np.zeros(0, np.float32))
    empty_angles = (np.zeros((0, 3), np.int32),
                    np.zeros(0, np.float32), np.zeros(0, np.float32))
    empty_four = (np.zeros((0, 4), np.int32),
                  np.zeros(0, np.float32), np.zeros(0, np.int32),
                  np.zeros(0, np.float32))
    system = SystemState(
        bonds=empty_bonds, angles=empty_angles,
        torsions=empty_four, impropers=empty_four,
        charges=np.zeros(2, np.float32),
        sigmas=np.ones(2, np.float32),
        epsilons=np.ones(2, np.float32),
        molecule_id=np.array([0, 1], np.int32),
        pf6_reacted=jnp.zeros(0, dtype=jnp.bool_),
    )
    initial = jnp.array([[0., 0., 0.], [1., 0., 0.]], dtype=jnp.float32)
    calls = []

    def no_reaction(key, positions, ff_in, system_in):
        calls.append(np.asarray(positions).copy())
        return key, False, ff_in, system_in, {}, positions

    result = run_md_nvt_with_reactions(
        jax.random.PRNGKey(0),
        cfg=SimConfig(steps=10, check_every=5, max_events=1,
                      dt=0.001, tau_T=0.5, temperature_k=188.0),
        init_positions=initial, masses=np.ones(2, dtype=np.float32),
        shift_fn=lambda R, dR, **kwargs: R + dR,
        ff=ff, sys=system, reaction_step_fn=no_reaction,
    )
    assert len(calls) == 2
    assert result.accepted_events == 0
    assert result.sys is system
    assert bool(jnp.all(jnp.isfinite(result.final_md_state.position)))
    assert not bool(np.asarray(result.ff.nlist.did_buffer_overflow))
