import numpy as np
import jax.numpy as jnp
import pytest
from reactive_md.forcefield import build_forcefield

def test_aa_bond_replaces_lj_with_harmonic():
    """
    For the rs@MD A/A2 benchmark:

    unbound A-A  -> Lennard-Jones interaction
    bonded A-A   -> harmonic bond, with the LJ pair excluded
    """

    # Use a distance away from both the LJ minimum and bond minimum
    # so that both expected energies are non-zero.
    r = 2.0

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [r,   0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    box = jnp.array(
        [20.0, 20.0, 20.0],
        dtype=jnp.float32,
    )

    # rs@MD benchmark parameters converted to
    # kcal/mol and Angstrom.
    epsilon = 1.9572 / 4.184
    sigma = 3.4

    r0_bond = 1.5
    k_bond_paper = 25000.0 / 100.0 / 4.184
    k_bond_jax = 0.5 * k_bond_paper

    charges = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    sigmas = np.array(
        [sigma, sigma],
        dtype=np.float32,
    )

    epsilons = np.array(
        [epsilon, epsilon],
        dtype=np.float32,
    )

    # Keep both particles in the same molecule here so the only
    # structural difference between the two force fields is the bond.
    molecule_id = np.array(
        [1, 1],
        dtype=np.int32,
    )

    empty_angles = (
        np.zeros((0, 3), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )

    empty_torsions = (
        np.zeros((0, 4), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
    )

    empty_impropers = (
        np.zeros((0, 4), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
    )

    # ---------------------------------------------------------
    # 1. Unbound A-A pair
    # ---------------------------------------------------------

    empty_bonds = (
        np.zeros((0, 2), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )

    ff_unbound = build_forcefield(
        R=R,
        box=box,
        bond_idx=empty_bonds[0],
        k_b=empty_bonds[1],
        r0=empty_bonds[2],
        angle_idx=empty_angles[0],
        k_theta=empty_angles[1],
        theta0=empty_angles[2],
        torsions=empty_torsions,
        impropers=empty_impropers,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        molecule_id=molecule_id,
        r_cut=9.0,
        dr_threshold=0.5,
    )

    energy_unbound = float(
        ff_unbound.energy_fn(
            R,
            ff_unbound.nlist,
        )["total"]
    )

    # ---------------------------------------------------------
    # 2. Bonded A2 pair
    # ---------------------------------------------------------

    bonded = (
        np.array([[0, 1]], dtype=np.int32),
        np.array([k_bond_jax], dtype=np.float32),
        np.array([r0_bond], dtype=np.float32),
    )

    ff_bonded = build_forcefield(
        R=R,
        box=box,
        bond_idx=bonded[0],
        k_b=bonded[1],
        r0=bonded[2],
        angle_idx=empty_angles[0],
        k_theta=empty_angles[1],
        theta0=empty_angles[2],
        torsions=empty_torsions,
        impropers=empty_impropers,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        molecule_id=molecule_id,
        r_cut=9.0,
        dr_threshold=0.5,
    )

    energy_bonded = float(
        ff_bonded.energy_fn(
            R,
            ff_bonded.nlist,
        )["total"]
    )

    # ---------------------------------------------------------
    # Expected harmonic energy
    # ---------------------------------------------------------

    expected_bond = (
         k_bond_jax
        * (r - r0_bond) ** 2
    )

    np.testing.assert_allclose(
        energy_bonded,
        expected_bond,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    # And the bonded and unbound states must clearly differ.
    assert not np.isclose(
        energy_unbound,
        energy_bonded,
    )
