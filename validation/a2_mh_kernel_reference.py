"""Independent, NumPy-only two-atom MH kernel reference.

No reactive_md/JAX imports. For two labeled atoms there is one candidate
in each permitted direction, so the proposal ratio is exactly one.
Outside the association cutoff, both topology transitions have zero
probability: dissociation is blocked by the reverse-reachability test.
"""
from __future__ import annotations

import math

from validation.a2_mh_moving_reference import (
    TwoAtomModel, energy_bonded, energy_unbound,
)


def kernel_at_distance(distance: float, model: TwoAtomModel | None = None) -> dict:
    model = model or TwoAtomModel()
    r = float(distance)
    if not math.isfinite(r) or r < 0:
        raise ValueError('distance must be finite and nonnegative')
    u0 = float(energy_unbound(r, model))
    u1 = float(energy_bonded(r, model))
    delta = u1 - u0
    # Stable logistic probability for the bonded topology at fixed R.
    z = delta / model.kt
    conditional = (math.exp(-z) / (1.0 + math.exp(-z))
                   if z >= 0 else 1.0 / (1.0 + math.exp(z)))
    eligible = r <= model.association_cutoff
    pa = math.exp(min(0.0, -delta / model.kt)) if eligible else 0.0
    pd = math.exp(min(0.0, +delta / model.kt)) if eligible else 0.0
    # Fluxes use unnormalized equilibrium weights and direction probability 1/2.
    flux_a = math.exp(-u0 / model.kt) * 0.5 * pa
    flux_d = math.exp(-u1 / model.kt) * 0.5 * pd
    return {
        'distance': r, 'energy_unbound': u0, 'energy_bonded': u1,
        'delta_energy_association': delta,
        'conditional_bonded_probability': conditional,
        'association_eligible': eligible,
        'association_acceptance': pa, 'dissociation_acceptance': pd,
        'forward_flux': flux_a, 'reverse_flux': flux_d,
    }
