import numpy as np
import jax
import jax.numpy as jnp

from reactive_md.config import SimConfig
from reactive_md.md_driver import run_md_nvt_with_reactions
from reactive_md.forcefield import FFBundle
from reactive_md.reaction import SystemState

class DummyNeighborFn:
    def allocate(self, R):
        return None
    def update(self, R, nlist):
        return None

def test_md_driver_runs_with_dummy_forcefield():
    # constant-energy forcefield
    def energy_fn(R, neighbor):
        return {"total": jnp.array(0.0)}

    def disp_fn(a, b):
        return a - b

    neighbor_fn = DummyNeighborFn()

    ff = FFBundle(
        topo=None,
        params=None,
        energy_fn=energy_fn,
        neighbor_fn=neighbor_fn,
        disp_fn=disp_fn,
        nlist=None,
        coulomb_handler=None,
        nb_options=type("NB", (), {"r_cut": 8.0, "dr_threshold": 0.5})(),
    )

    # minimal sys (no reaction actually happens)
    sys = SystemState(
        bonds=(np.zeros((0,2), np.int32), np.zeros((0,), np.float32), np.zeros((0,), np.float32)),
        angles=(np.zeros((0,3), np.int32), np.zeros((0,), np.float32), np.zeros((0,), np.float32)),
        torsions=(np.zeros((0,4), np.int32), np.zeros((0,), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)),
        impropers=(np.zeros((0,4), np.int32), np.zeros((0,), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)),
        charges=np.zeros((2,), np.float32),
        sigmas=np.ones((2,), np.float32),
        epsilons=np.ones((2,), np.float32),
        molecule_id=np.array([1,1], np.int32),
        pf6_reacted=jnp.zeros((0,), dtype=jnp.bool_),
    )

    cfg = SimConfig(steps=10, check_every=5, max_events=1)

    # 2 atoms, masses
    init_positions = jnp.array([[0.0,0.0,0.0],[1.0,0.0,0.0]], dtype=jnp.float32)
    masses = np.array([1.0, 1.0], dtype=np.float32)

    #shift_fn = lambda R, dR: R + dR
    shift_fn = lambda R, dR, **kwargs: R + dR
    def reaction_step_fn(key, positions, ff_in, sys_in):
        return key, False, ff_in, sys_in, {}

    res = run_md_nvt_with_reactions(
        jax.random.PRNGKey(0),
        cfg=cfg,
        init_positions=init_positions,
        masses=masses,
        shift_fn=shift_fn,
        ff=ff,
        sys=sys,
        reaction_step_fn=reaction_step_fn
    )
    assert res.accepted_events == 0

