"""Independent two-A-particle configurational equilibrium reference.

Integrate the minimum-image relative displacement over the full periodic cube.
No production reactive_md or JAX imports. The unbound LJ is explicitly
UNSHIFTED and truncated at r_cut; the bonded pair has a harmonic potential
and replaces the LJ interaction. No standard-state or chemical-potential term.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TwoAtomModel:
    box: float = 7.0
    r_cut: float = 3.0
    association_cutoff: float = 3.0
    sigma: float = 1.2
    epsilon: float = 0.2
    bond_k: float = 1.5
    bond_r0: float = 2.0
    bond_offset: float = -0.5
    temperature_k: float = 300.0
    kb_real: float = 0.0019872041

    @property
    def kt(self):
        return self.temperature_k * self.kb_real

    def validate(self):
        if not (0 < self.r_cut < self.box / 2):
            raise ValueError('Require r_cut < box/2 for the minimum-image reference')
        if not (0 < self.association_cutoff < self.box / 2):
            raise ValueError('Require association_cutoff < box/2')
        if min(self.sigma, self.epsilon, self.bond_k, self.bond_r0, self.kt) <= 0:
            raise ValueError('Invalid model parameters')


def energy_unbound(r, model: TwoAtomModel):
    r = np.asarray(r, dtype=np.float64)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        x6 = (model.sigma / np.maximum(r, 1e-12)) ** 6
        lj = 4.0 * model.epsilon * (x6 * x6 - x6)
    return np.where(r < model.r_cut, lj, 0.0)


def energy_bonded(r, model: TwoAtomModel):
    r = np.asarray(r, dtype=np.float64)
    return model.bond_k * (r - model.bond_r0) ** 2 + model.bond_offset


def integrate_reference(model: TwoAtomModel, order=96):
    """Tensor-product Gauss-Legendre quadrature over the full relative cube.

    Each coordinate is uniformly distributed in [-L/2,L/2] for two
    independent particles in a periodic cube. Translation contributes a
    common volume factor, which cancels between topology probabilities.
    """
    model.validate()
    if order < 8:
        raise ValueError('Quadrature order must be >= 8')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    xyz = nodes * model.box / 2
    w = weights * model.box / 2
    yz2 = xyz[:, None] ** 2 + xyz[None, :] ** 2
    wyz = w[:, None] * w[None, :]
    zu = zb = 0.0
    ru = rb = 0.0
    for x, wx in zip(xyz, w):
        r = np.sqrt(x*x + yz2)
        bu = np.exp(-np.clip(energy_unbound(r, model) / model.kt, -700, 700))
        bb = np.exp(-np.clip(energy_bonded(r, model) / model.kt, -700, 700))
        # Clipping positive LJ energies at 700 is harmless to float64 precision.
        zu += wx * np.sum(wyz * bu)
        zb += wx * np.sum(wyz * bb)
        ru += wx * np.sum(wyz * bu * r)
        rb += wx * np.sum(wyz * bb * r)
    return {
        'quadrature_order': int(order),
        'partition_unbound': float(zu),
        'partition_bonded': float(zb),
        'p_bonded': float(zb / (zu + zb)),
        'mean_r_unbound': float(ru / zu),
        'mean_r_bonded': float(rb / zb),
        'model': vars(model),
    }
