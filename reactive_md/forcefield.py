# reactive_md/forcefield.py
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import jax.numpy as jnp

from jax_md.mm_forcefields import oplsaa, neighbor
from jax_md.mm_forcefields.base import NonbondedOptions, Topology
from jax_md.mm_forcefields.oplsaa.params import Parameters
from jax_md.mm_forcefields.nonbonded.electrostatics import CutoffCoulomb

@dataclass
class FFBundle:
    topo: Topology
    params: Parameters
    energy_fn: callable
    neighbor_fn: callable
    disp_fn: callable
    nlist: object
    coulomb_handler: object
    nb_options: NonbondedOptions

def build_forcefield(
    R,
    box,
    *,
    bond_idx, k_b, r0,
    angle_idx, k_theta, theta0,
    torsions, impropers,
    charges, sigmas, epsilons,
    molecule_id,
    r_cut: float,
    dr_threshold: float
) -> FFBundle:
    """
    Construct Topology, Parameters, energy_fn, neighbor_fn, disp_fn, and allocate nlist.
    Mirrors your original setup.
    """
    n_atoms = R.shape[0]
    tors_idx, tors_k, tors_n, tors_gamma = torsions
    impr_idx, impr_k, impr_n, impr_gamma = impropers

    exclusion_mask = neighbor.make_exclusion_mask(n_atoms, bond_idx, angle_idx, molecule_id)
    pair_14_mask = neighbor.make_14_table(n_atoms, tors_idx, exclusion_mask, molecule_id)

    topo = Topology(
        n_atoms=n_atoms,
        bonds=jnp.array(bond_idx, dtype=int),
        angles=jnp.array(angle_idx, dtype=int),
        torsions=jnp.array(tors_idx, dtype=int),
        impropers=jnp.array(impr_idx, dtype=int),
        exclusion_mask=exclusion_mask,
        pair_14_mask=pair_14_mask,
    )

    bonded_params = SimpleNamespace(
        bond_k=jnp.array(k_b),
        bond_r0=jnp.array(r0),
        angle_k=jnp.array(k_theta),
        angle_theta0=jnp.array(theta0),
        torsion_k=jnp.array(tors_k),
        torsion_n=jnp.array(tors_n),
        torsion_gamma=jnp.array(tors_gamma),
        improper_k=jnp.array(impr_k),
        improper_n=jnp.array(impr_n),
        improper_gamma=jnp.array(impr_gamma),
    )
    nonbonded_params = SimpleNamespace(
        charges=jnp.array(charges),
        sigma=jnp.array(sigmas),
        epsilon=jnp.array(epsilons),
    )

    params = Parameters(bonded=bonded_params, nonbonded=nonbonded_params)

    coulomb_handler = CutoffCoulomb(r_cut=r_cut)
    nb_options = NonbondedOptions(r_cut=r_cut, dr_threshold=dr_threshold)

    energy_fn, neighbor_fn, disp_fn = oplsaa.energy(topo, params, box, coulomb_handler, nb_options)
    nlist = neighbor_fn.allocate(R)

    return FFBundle(
        topo=topo,
        params=params,
        energy_fn=energy_fn,
        neighbor_fn=neighbor_fn,
        disp_fn=disp_fn,
        nlist=nlist,
        coulomb_handler=coulomb_handler,
        nb_options=nb_options,
    )

