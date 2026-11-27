# reactive_md/mace_mc_energy.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from .performance import PerformanceTimer

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
        perf = PerformanceTimer()
        R = jnp.asarray(R)

        # Allocation and overflow recovery are intentionally outside jit.
        extra_capacity = 2
        for attempt in range(13):
            with perf.measure(
                'mace_mc_neighbor_allocate', attempt=attempt, extra_capacity=extra_capacity,
            ) as wait:
                nbrs = self.neighbor_fn.allocate(
                    R, extra_capacity=extra_capacity
                )
                nbrs = nbrs.update(R)
                wait(nbrs)

            if not bool(np.asarray(nbrs.did_buffer_overflow)):
                break

            extra_capacity = max(8, 2 * extra_capacity)
        else:
            raise RuntimeError(
                "MACE MC neighbor-list allocation overflowed "
                "after capacity growth"
            )

        with perf.measure('mace_mc_energy_kernel'):
            E = self._energy(R, nbrs)
            E.block_until_ready()

        print("MACE energy evaluation completed.", flush=True)
        return float(E)
