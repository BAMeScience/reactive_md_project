# reactive_md/mace_mc_energy.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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

        # Temporary memory diagnostics
        idx = jax.device_get(nbrs.idx)
        valid_edges = int((idx < R.shape[0]).sum())

        print("MACE coordinate dtype:", R.dtype)
        print("MACE box dtype:", self.box0.dtype)
        print("Neighbor array shape:", idx.shape)
        print("Neighbor array dtype:", idx.dtype)
        print("Allocated neighbor slots:", idx.size)
        print("Occupied neighbor slots:", valid_edges)
        print("Neighbor overflow:", bool(jax.device_get(nbrs.did_buffer_overflow)))
        print("Starting MACE energy evaluation...", flush=True)

        idx = np.asarray(jax.device_get(nbrs.idx))
        counts = np.sum(idx < R.shape[0], axis=1)

        print("Mean neighbors per atom:", counts.mean(), flush=True)
        print("95th percentile neighbors:", np.percentile(counts, 95), flush=True)
        print("Maximum neighbors per atom:", counts.max(), flush=True)
        print("Atoms using all 153 slots:", np.sum(counts == idx.shape[1]), flush=True)

        E = self._energy(R, nbrs)
        E.block_until_ready()

        print("MACE energy evaluation completed.", flush=True)
        return float(E)
