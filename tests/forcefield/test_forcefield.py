import numpy as np
import jax.numpy as jnp
import pytest

from reactive_md.forcefield import build_forcefield

def test_build_forcefield_minimal_two_atoms():
    # Tiny system: 2 atoms, no bonded terms
    R = jnp.array([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0]], dtype=jnp.float32)
    box = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32)

    bonds = (np.zeros((0,2), dtype=np.int32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))
    angles = (np.zeros((0,3), dtype=np.int32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))
    torsions = (np.zeros((0,4), dtype=np.int32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32))
    impropers = (np.zeros((0,4), dtype=np.int32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32))

    charges = np.array([0.0, 0.0], dtype=np.float32)
    sigmas = np.array([3.0, 3.0], dtype=np.float32)
    epsilons = np.array([0.1, 0.1], dtype=np.float32)
    molecule_id = np.array([1, 1], dtype=np.int32)

    ff = build_forcefield(
        R=R, box=box,
        bond_idx=bonds[0], k_b=bonds[1], r0=bonds[2],
        angle_idx=angles[0], k_theta=angles[1], theta0=angles[2],
        torsions=torsions, impropers=impropers,
        charges=charges, sigmas=sigmas, epsilons=epsilons,
        molecule_id=molecule_id,
        r_cut=8.0, dr_threshold=0.5
    )

    out = ff.energy_fn(R, ff.nlist)
    assert "total" in out
    assert ff.topo.n_atoms == 2

@pytest.mark.parametrize("r_cut", [5.0, 8.0])
def test_build_forcefield_r_cut_effect_is_constructible(r_cut):
    R = jnp.array([[0.0,0.0,0.0],[1.0,0.0,0.0]], dtype=jnp.float32)
    box = jnp.array([10.0,10.0,10.0], dtype=jnp.float32)
    empty2 = np.zeros((0,2), dtype=np.int32)
    empty3 = np.zeros((0,3), dtype=np.int32)
    empty4 = np.zeros((0,4), dtype=np.int32)

    ff = build_forcefield(
        R=R, box=box,
        bond_idx=empty2, k_b=np.zeros((0,), np.float32), r0=np.zeros((0,), np.float32),
        angle_idx=empty3, k_theta=np.zeros((0,), np.float32), theta0=np.zeros((0,), np.float32),
        torsions=(empty4, np.zeros((0,), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)),
        impropers=(empty4, np.zeros((0,), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)),
        charges=np.array([0.0,0.0], np.float32),
        sigmas=np.array([3.0,3.0], np.float32),
        epsilons=np.array([0.1,0.1], np.float32),
        molecule_id=np.array([1,1], np.int32),
        r_cut=r_cut, dr_threshold=0.5
    )
    assert float(ff.nb_options.r_cut) == r_cut

