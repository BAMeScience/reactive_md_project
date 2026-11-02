"""A2 MC potential-energy accounting, grounded in the original rs@md C++.

Original source:
  src/parser/energyParserGMX.cpp: local + solvation =
      (U_reacting_products - U_reacting_reactants)
    + (U_reacting-solvent_products - U_reacting-solvent_reactants).

For the A2-only model, the same quantity is obtained exactly by subtracting
spectator-only potential from the full potential at EACH configuration. This
removes spectator-spectator energy changes during Langevin relaxation while
retaining the reacting pair's internal and solvent interactions. The explicit
V0 offset is already present in the full-system JAX-MD force field and is not
added again. This module is not a general-purpose molecular energy partition.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def local_solvation_energy(ff, positions, system, reacting_pair, box):
    """Return U_full - U_spectators for one A2 reacting pair.

    Only valid for an A2 topology with no bonded term connecting a reacting
    atom to a spectator. A2 bond offsets on spectator dimers need not be
    included in U_spectators: they are unchanged by an A2 trial and cancel in
    the before/after difference.
    """
    from reactive_md.reactions.a2_metropolis import _potential, _valid_nlist
    from reactive_md.forcefield import build_forcefield

    full = _potential(ff, positions, _valid_nlist(ff, positions))
    n = int(positions.shape[0])
    pair = {int(reacting_pair[0]), int(reacting_pair[1])}
    if len(pair) != 2 or any(i < 0 or i >= n for i in pair):
        raise ValueError('reacting_pair must contain two distinct valid atom indices')
    spectator = np.asarray([i for i in range(n) if i not in pair], dtype=np.int32)
    if len(spectator) < 2:
        return full

    remap = np.full(n, -1, dtype=np.int32)
    remap[spectator] = np.arange(len(spectator), dtype=np.int32)

    def subset_terms(terms, arity):
        indices = np.asarray(terms[0], dtype=np.int32).reshape(-1, arity)
        if len(indices) == 0:
            keep = np.empty(0, dtype=bool)
        else:
            spectators_in_term = remap[indices] >= 0
            # Cross-boundary bonded terms invalidate this A2-only partition.
            if np.any(np.any(spectators_in_term, axis=1) &
                      ~np.all(spectators_in_term, axis=1)):
                raise ValueError('A2 local energy requires no reacting-spectator bonded terms')
            keep = np.all(spectators_in_term, axis=1)
        return (remap[indices[keep]],) + tuple(np.asarray(x)[keep] for x in terms[1:])

    bonds = subset_terms(system.bonds, 2)
    angles = subset_terms(system.angles, 3)
    torsions = subset_terms(system.torsions, 4)
    impropers = subset_terms(system.impropers, 4)
    pos_s = jnp.asarray(positions)[spectator]
    spectator_ff = build_forcefield(
        pos_s, box,
        bond_idx=bonds[0], k_b=bonds[1], r0=bonds[2],
        angle_idx=angles[0], k_theta=angles[1], theta0=angles[2],
        torsions=torsions, impropers=impropers,
        charges=jnp.asarray(system.charges)[spectator],
        sigmas=jnp.asarray(system.sigmas)[spectator],
        epsilons=jnp.asarray(system.epsilons)[spectator],
        molecule_id=jnp.asarray(system.molecule_id)[spectator],
        r_cut=float(ff.nb_options.r_cut),
        dr_threshold=float(ff.nb_options.dr_threshold),
        use_shift_lj=ff.nb_options.use_shift_lj,
        energy_offset=0.0,
    )
    return full - _potential(spectator_ff, pos_s, _valid_nlist(spectator_ff, pos_s))
