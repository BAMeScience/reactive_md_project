# reactive_md/forcefield.py
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np

from jax_md import space
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
    # Padded parameter arrays and the settings that stay fixed during a run.
    # Used by the cached MD loops, so a new force field does not need a recompile.
    jit_arrays: dict | None = None
    jit_static: object = None


class FFStatic(NamedTuple):
    """Settings that do not change when the topology changes (hashable, used as a cache key)."""
    box: tuple
    box_shape: tuple
    nb_options: NonbondedOptions


# Bonded term lists are padded up to a multiple of this, so their shapes rarely change.
_PAD_BLOCK = 64

# One displacement / neighbor-list function per box and cutoff, shared by all force fields.
_SPACE_FNS = {}


def _space_fns(box, r_cut, dr_threshold):
    box_np = np.asarray(box, dtype=np.float64)
    key = (tuple(box_np.ravel().tolist()), box_np.shape, float(r_cut), float(dr_threshold))
    if key not in _SPACE_FNS:
        disp_fn, _ = space.periodic(box)
        _SPACE_FNS[key] = (disp_fn, neighbor.create_neighbor_list(disp_fn, box, r_cut, dr_threshold))
    return _SPACE_FNS[key]


# First cell_list_fn seen for each neighbor function.
_CELL_LIST_FNS = {}


def stable_nlist(nlist):
    """Give the neighbor list the same cell_list_fn object every time.

    jax-md builds a new (but identical) cell_list_fn on each allocate. It is static
    data, so jit would treat every new list as a new input and compile again.
    """
    shared = _CELL_LIST_FNS.setdefault(nlist.update_fn, nlist.cell_list_fn)
    if nlist.cell_list_fn is shared:
        return nlist
    return dataclasses.replace(nlist, cell_list_fn=shared)


def _pad_terms(idx, values, width):
    """Pad bonded terms by repeating the last term with k = 0, so they add no energy or force.

    values[0] must be the force constant. Empty term lists stay empty.
    """
    idx = np.asarray(idx, dtype=np.int32).reshape(-1, width)
    values = [np.asarray(v, dtype=np.float32).reshape(-1) for v in values]
    n = idx.shape[0]
    if n == 0:
        return jnp.asarray(idx), [jnp.asarray(v) for v in values]
    extra = -(-n // _PAD_BLOCK) * _PAD_BLOCK - n
    idx = np.concatenate([idx, np.repeat(idx[-1:], extra, axis=0)])
    values = [np.concatenate([v, np.repeat(v[-1:], extra)]) for v in values]
    values[0][n:] = 0.0
    return jnp.asarray(idx), [jnp.asarray(v) for v in values]


def energy_from_arrays(static: FFStatic, arrays: dict, positions, nlist):
    """Total OPLS-AA energy with the topology and parameters passed in as arrays.

    Gives the same value as FFBundle.energy_fn(...)["total"], but can sit inside a
    cached jit, because nothing topology-specific is baked into the compiled code.
    """
    box = jnp.asarray(np.asarray(static.box).reshape(static.box_shape))
    topo = Topology(
        n_atoms=positions.shape[0],
        bonds=arrays["bonds"],
        angles=arrays["angles"],
        torsions=arrays["torsions"],
        impropers=arrays["impropers"],
        exclusion_mask=arrays["exclusion_mask"],
        pair_14_mask=arrays["pair_14_mask"],
    )
    bonded = SimpleNamespace(
        bond_k=arrays["bond_k"], bond_r0=arrays["bond_r0"],
        angle_k=arrays["angle_k"], angle_theta0=arrays["angle_theta0"],
        torsion_k=arrays["torsion_k"], torsion_n=arrays["torsion_n"],
        torsion_gamma=arrays["torsion_gamma"],
        improper_k=arrays["improper_k"], improper_n=arrays["improper_n"],
        improper_gamma=arrays["improper_gamma"],
    )
    nonbonded = SimpleNamespace(
        charges=arrays["charges"], sigma=arrays["sigma"], epsilon=arrays["epsilon"],
    )
    energy_fn, _, _ = oplsaa.energy(
        topo, Parameters(bonded=bonded, nonbonded=nonbonded), box,
        CutoffCoulomb(r_cut=static.nb_options.r_cut), static.nb_options,
    )
    return energy_fn(positions, nlist)["total"] + arrays["offset"]


# faster: only allocate the neighbor list when someone reads ff.nlist, since
# reaction trials overwrite it with the relaxed list and never use the first one.
class _LazyNlistFFBundle(FFBundle):
    @property
    def nlist(self):
        if self._nlist is None and self._nlist_positions is not None:
            self._nlist = self.neighbor_fn.allocate(self._nlist_positions)
            self._nlist_positions = None
        return self._nlist

    @nlist.setter
    def nlist(self, value):
        self._nlist = value
        self._nlist_positions = None


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
    dr_threshold: float,
    use_shift_lj: bool | None = None,
    energy_offset: float | None = None,
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
    nb_kwargs = {
      "r_cut": r_cut,
      "dr_threshold": dr_threshold,
    }

    if use_shift_lj is not None:
       nb_kwargs["use_shift_lj"] = use_shift_lj

    nb_options = NonbondedOptions(**nb_kwargs)

    # energy_fn, neighbor_fn, disp_fn = oplsaa.energy(topo, params, box, coulomb_handler, nb_options)
    # faster: reuse one neighbor/displacement function per box, so their compiled code is shared.
    energy_fn, _, _ = oplsaa.energy(topo, params, box, coulomb_handler, nb_options)
    disp_fn, neighbor_fn = _space_fns(box, r_cut, dr_threshold)
    if energy_offset is not None:
       base_energy_fn = energy_fn
       offset = jnp.asarray(energy_offset)

       def energy_fn(positions, nlist):
           energies = base_energy_fn(positions, nlist)
           return {
              **energies,
              "offset": offset,
              "total": energies["total"] + offset,
           }
    # nlist = neighbor_fn.allocate(R)
    #
    # return FFBundle(
    #     topo=topo,
    #     params=params,
    #     energy_fn=energy_fn,
    #     neighbor_fn=neighbor_fn,
    #     disp_fn=disp_fn,
    #     nlist=nlist,
    #     coulomb_handler=coulomb_handler,
    #     nb_options=nb_options,
    # )

    # faster: same bundle, but the neighbor list is built on first use (see _LazyNlistFFBundle).
    ff = _LazyNlistFFBundle(
        topo=topo,
        params=params,
        energy_fn=energy_fn,
        neighbor_fn=neighbor_fn,
        disp_fn=disp_fn,
        nlist=None,
        coulomb_handler=coulomb_handler,
        nb_options=nb_options,
    )
    ff._nlist_positions = R

    # faster: padded copies of the parameters for the cached MD loops (relaxation.py, engine/nvt.py).
    bonds, (bk, br0) = _pad_terms(bond_idx, [k_b, r0], 2)
    angles, (ak, at0) = _pad_terms(angle_idx, [k_theta, theta0], 3)
    tors, (tk, tn, tg) = _pad_terms(tors_idx, [tors_k, tors_n, tors_gamma], 4)
    imps, (ik, in_, ig) = _pad_terms(impr_idx, [impr_k, impr_n, impr_gamma], 4)
    ff.jit_arrays = {
        "bonds": bonds, "bond_k": bk, "bond_r0": br0,
        "angles": angles, "angle_k": ak, "angle_theta0": at0,
        "torsions": tors, "torsion_k": tk, "torsion_n": tn, "torsion_gamma": tg,
        "impropers": imps, "improper_k": ik, "improper_n": in_, "improper_gamma": ig,
        "exclusion_mask": exclusion_mask, "pair_14_mask": pair_14_mask,
        "charges": jnp.asarray(charges, dtype=jnp.float32),
        "sigma": jnp.asarray(sigmas, dtype=jnp.float32),
        "epsilon": jnp.asarray(epsilons, dtype=jnp.float32),
        "offset": jnp.asarray(0.0 if energy_offset is None else energy_offset, dtype=jnp.float32),
    }
    box_np = np.asarray(box, dtype=np.float64)
    ff.jit_static = FFStatic(tuple(box_np.ravel().tolist()), box_np.shape, nb_options)
    return ff

