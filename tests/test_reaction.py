import numpy as np
import jax.numpy as jnp

from reactive_md.templates_pf5 import make_pf5_template, make_lif_template
from reactive_md.reaction import (
    SystemState,
    Candidate,
    propose_reaction_trial,
    accept_reject,
)

def make_sys_for_pf6_case():
    # 8 atoms: PF6 atoms 0..6, Li=7
    # types: P=6, F=7, Li=8
    atom_types = np.array([6,7,7,7,7,7,7,8], dtype=np.int32)
    molecule_id = np.array([1,1,1,1,1,1,1,2], dtype=np.int32)

    # bonds: one PF6 internal (0-1) should be removed; one cross (0-7) should remain
    bond_idx = np.array([[0,1],[0,7]], dtype=np.int32)
    k_b = np.array([100.0, 200.0], dtype=np.float32)
    r0 = np.array([1.6, 2.0], dtype=np.float32)

    # no angles/torsions/impropers for test (simple)
    angle_idx = np.zeros((0,3), dtype=np.int32)
    k_theta = np.zeros((0,), dtype=np.float32)
    theta0 = np.zeros((0,), dtype=np.float32)

    tors_idx = np.zeros((0,4), dtype=np.int32)
    tors_k = np.zeros((0,), dtype=np.float32)
    tors_n = np.zeros((0,), dtype=np.int32)
    tors_g = np.zeros((0,), dtype=np.float32)

    impr_idx = np.zeros((0,4), dtype=np.int32)
    impr_k = np.zeros((0,), dtype=np.float32)
    impr_n = np.zeros((0,), dtype=np.int32)
    impr_g = np.zeros((0,), dtype=np.float32)

    charges = np.zeros((8,), dtype=np.float32)
    sigmas = np.full((8,), 3.0, dtype=np.float32)
    eps = np.full((8,), 0.1, dtype=np.float32)

    sys = SystemState(
        bonds=(bond_idx, k_b, r0),
        angles=(angle_idx, k_theta, theta0),
        torsions=(tors_idx, tors_k, tors_n, tors_g),
        impropers=(impr_idx, impr_k, impr_n, impr_g),
        charges=charges,
        sigmas=sigmas,
        epsilons=eps,
        molecule_id=molecule_id,
        pf6_reacted=jnp.array([False], dtype=jnp.bool_),
    )
    pf6_atoms = np.array([[0,1,2,3,4,5,6]], dtype=np.int32)
    return sys, pf6_atoms, atom_types

def test_propose_reaction_trial_updates_counts_and_params():
    pf5 = make_pf5_template()
    lif = make_lif_template()

    sys, pf6_atoms, atom_types = make_sys_for_pf6_case()

    cand = Candidate(k_pf6=0, li_idx=7, leave_F=6, dmin=2.0)

    trial, pf6_molid = propose_reaction_trial(
        sys, cand,
        pf6_atoms_np=pf6_atoms,
        atom_types_np=atom_types,
        pf5=pf5,
        lif=lif,
        p_type=6, f_type=7, li_type=8
    )
    assert trial is not None
    assert pf6_molid == 1

    # bond (0,1) removed; bond (0,7) kept; + 5 PF5 bonds added => total 1 + 5 = 6
    bond_idx2 = trial["bonds"][0]
    assert bond_idx2.shape[0] == 6

    # leaving F got new molid
    mol2 = trial["molecule_id"]
    assert mol2[6] != 1
    assert mol2[6] == mol2.max()

    # charges updated:
    # - leaving F charge becomes LiF F charge (-1)
    # - Li becomes +1
    assert abs(trial["charges"][6] - lif.nb["F"]["q"]) < 1e-6
    assert abs(trial["charges"][7] - lif.nb["Li"]["q"]) < 1e-6

def test_accept_reject_probability_bounds():
    import jax
    key = jax.random.PRNGKey(0)

    key2, accepted, p = accept_reject(key, dE=0.0, beta=1.0)
    assert 0.0 <= p <= 1.0

    key3, accepted2, p2 = accept_reject(key2, dE=1e6, beta=1.0)
    assert p2 == 0.0  # underflow -> exp(-beta*dE) ~ 0

