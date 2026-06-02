# reactive_md/mace_mc_energy.py
from __future__ import annotations

import jax
import jax.numpy as jnp


class MaceJaxEnergyEvaluator:
    def __init__(self, *, neighbor_fn, make_energy_fn, box0):
        self.neighbor_fn = neighbor_fn
        self.make_energy_fn = make_energy_fn
        self.box0 = jnp.asarray(box0)

        @jax.jit
        def _energy(R, nbrs):
            energy_fn = self.make_energy_fn(nbrs)
            return energy_fn(R, box=self.box0)

        self._energy = _energy

    def energy(self, R) -> float:
        R = jnp.asarray(R)

        # Allocation is intentionally outside jit.
        nbrs = self.neighbor_fn.allocate(R, extra_capacity=2)
        nbrs = nbrs.update(R)

        E = self._energy(R, nbrs)
        E.block_until_ready()
        return float(E)
