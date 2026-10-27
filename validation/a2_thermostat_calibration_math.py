"""Analytic momentum statistics for two equal-mass particles; NumPy only.

For canonical momenta p_i ~ N(0, m kT I_3), independent:
K_total/(3 kT), K_relative/(1.5 kT), K_COM/(1.5 kT) all have mean 1.
The last two use P=p0+p1 and q=(p1-p0)/2, respectively.
"""
from __future__ import annotations
import numpy as np

AMU_TO_KCAL_MOL_PS2_A2 = 0.002390057361376673  # CODATA-level approximation


def mass_from_amu(amu: float) -> float:
    if not np.isfinite(amu) or amu <= 0:
        raise ValueError('amu must be finite and positive')
    return float(amu * AMU_TO_KCAL_MOL_PS2_A2)


def momentum_ratios(p, mass: float, kT: float):
    """Return per-replica ratios; p shape (..., 2, 3)."""
    p = np.asarray(p, dtype=float)
    if p.ndim < 2 or p.shape[-2:] != (2, 3) or not np.isfinite(p).all():
        raise ValueError('p must be finite with trailing shape (2,3)')
    if not (np.isfinite(mass) and mass > 0 and np.isfinite(kT) and kT > 0):
        raise ValueError('mass and kT must be positive and finite')
    total = np.sum(p * p, axis=(-2, -1)) / (2 * mass)
    relative = np.sum((p[..., 1, :] - p[..., 0, :]) ** 2, axis=-1) / (4 * mass)
    com = np.sum((p[..., 1, :] + p[..., 0, :]) ** 2, axis=-1) / (4 * mass)
    return {'total': total / (3 * kT),
            'relative': relative / (1.5 * kT),
            'com': com / (1.5 * kT)}


def summarize_momenta(p, mass: float, kT: float):
    """p: (n_samples, n_independent_replicas, 2, 3).

    Error bars use *replica means*, never treat time samples as independent.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 4 or p.shape[2:] != (2, 3) or p.shape[1] < 2:
        raise ValueError('expected (samples, replicas>=2, 2, 3)')
    stats = momentum_ratios(p, mass, kT)
    out = {}
    for name, values in stats.items():
        means = np.mean(values, axis=0)
        avg = float(np.mean(means))
        se = float(np.std(means, ddof=1) / np.sqrt(len(means)))
        out[name] = {'mean': avg, 'replica_se': se,
                     'deviation_sigma': float(abs(avg - 1) / se) if se > 0 else None}
    out['momentum_component_second_moment_ratio'] = float(np.mean(p * p) / (mass * kT))
    out['momentum_component_mean_in_thermal_units'] = float(np.mean(p) / np.sqrt(mass * kT))
    out['replicas'] = int(p.shape[1])
    out['time_samples_per_replica'] = int(p.shape[0])
    return out


def assess_langevin(summary, *, tolerance=0.15, sigma=5.0):
    """Exploratory threshold; not a formal statistical acceptance test."""
    keys = ('total', 'relative', 'com')
    deviations = {k: abs(summary[k]['mean'] - 1) for k in keys}
    limits = {k: max(tolerance, sigma * summary[k]['replica_se']) for k in keys}
    return {'status': 'CONSISTENT_WITH_CANONICAL_MOMENTA'
            if all(deviations[k] <= limits[k] for k in keys)
            else 'INVESTIGATE_KINETIC_NORMALIZATION',
            'absolute_deviations': deviations, 'limits': limits}
