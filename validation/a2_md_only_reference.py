"""Pure-NumPy helpers for fixed-topology two-atom MD diagnostics.

A fixed-topology MD run must be compared with P(r | topology), not with the
joint topology-changing equilibrium distribution. No JAX or production imports.
"""
from __future__ import annotations

import numpy as np

from validation.a2_mh_mixing_reference import integrated_autocorrelation


def initial_displacement(geometry: str):
    if geometry == 'near':
        return np.array([2.0, 0.0, 0.0], dtype=float)
    if geometry == 'far':
        return np.array([2.2, 2.2, 2.2], dtype=float)
    raise ValueError('geometry must be near or far')


def kinetic_diagnostics(momentum, *, mass: float, kt: float):
    """Dimensionless kinetic energies, with no COM-motion subtraction.

    K_total/(3 kT) corresponds to six unconstrained momentum components.
    K_relative/(1.5 kT) corresponds to the three relative components.
    These ratios are *diagnostics*, not instantaneous thermodynamic T.
    """
    p = np.asarray(momentum, dtype=float)
    if p.shape != (2, 3) or not np.isfinite(p).all():
        raise ValueError('momentum must be a finite (2, 3) array')
    if mass <= 0 or kt <= 0:
        raise ValueError('mass and kt must be positive')
    total_ke = float(np.sum(p * p) / (2.0 * mass))
    relative_p = (p[1] - p[0]) / 2.0
    relative_ke = float(np.sum(relative_p ** 2) / mass)
    return {
        'kinetic_ratio_6dof': total_ke / (3.0 * kt),
        'relative_kinetic_ratio_3dof': relative_ke / (1.5 * kt),
        'com_momentum_norm': float(np.linalg.norm(p.sum(axis=0))),
    }


def conditional_radial_summary(distances, *, reference, bonded: bool):
    """Compare observed distances with the canonical *conditional* radial law."""
    x = np.asarray(distances, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('distances must be a nonempty finite 1D array')
    edges = np.asarray(reference['bin_edges_A'], dtype=float)
    if np.any(x < edges[0] - 1e-5) or np.any(x > edges[-1] + 1e-5):
        raise ValueError('distance outside periodic reference support')
    expected = np.asarray(reference[
        'prob_r_given_bonded' if bonded else 'prob_r_given_unbound'
    ], dtype=float)
    observed = np.histogram(x, bins=edges)[0] / len(x)
    half = len(x) // 2
    return {
        'mean_r_A': float(x.mean()),
        'min_r_A': float(x.min()), 'max_r_A': float(x.max()),
        'first_half_mean_r_A': float(x[:half].mean()) if half else None,
        'second_half_mean_r_A': float(x[half:].mean()),
        'conditional_radial_tv': float(0.5 * np.sum(np.abs(observed - expected))),
        'observed_conditional_shells': observed.tolist(),
        'expected_conditional_shells': expected.tolist(),
        'distance_iat': integrated_autocorrelation(x),
    }
