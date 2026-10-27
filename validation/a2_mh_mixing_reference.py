"""Independent NumPy-only reference and conservative mixing diagnostics.

No production reactive_md/JAX imports. These are diagnostics, not an
automatic proof of convergence or canonical sampling.
"""
from __future__ import annotations

import numpy as np

from validation.a2_mh_moving_reference import (
    TwoAtomModel, energy_bonded, energy_unbound,
)


def radial_reference(model: TwoAtomModel, *, order=96, bins=12):
    """Topology-resolved radial shell probabilities in the periodic cube.

    Uses the same independent quadrature as the full-box reference, but
    explicitly integrates weighted radial histograms. The binning is an
    approximation; discontinuous bin boundaries converge more slowly than
    the partition functions. Do not use this as a strict pass/fail threshold.
    """
    model.validate()
    if order < 8 or bins < 2:
        raise ValueError('order >= 8 and bins >= 2 required')
    edges = np.linspace(0, model.box * np.sqrt(3.0) / 2, bins + 1)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    xyz = nodes * model.box / 2
    w = weights * model.box / 2
    yz2 = xyz[:, None] ** 2 + xyz[None, :] ** 2
    wyz = w[:, None] * w[None, :]
    hist_u = np.zeros(bins, dtype=float)
    hist_b = np.zeros(bins, dtype=float)
    for x, wx in zip(xyz, w):
        r = np.sqrt(x * x + yz2)
        wu = wx * wyz * np.exp(-np.clip(energy_unbound(r, model) / model.kt, -700, 700))
        wb = wx * wyz * np.exp(-np.clip(energy_bonded(r, model) / model.kt, -700, 700))
        hist_u += np.histogram(r, bins=edges, weights=wu)[0]
        hist_b += np.histogram(r, bins=edges, weights=wb)[0]
    zu, zb = hist_u.sum(), hist_b.sum()
    total = zu + zb
    return {
        'bin_edges_A': edges.tolist(),
        'prob_unbound_by_shell': (hist_u / total).tolist(),
        'prob_bonded_by_shell': (hist_b / total).tolist(),
        'prob_r_by_shell': ((hist_u + hist_b) / total).tolist(),
        'prob_r_given_unbound': (hist_u / zu).tolist(),
        'prob_r_given_bonded': (hist_b / zb).tolist(),
        'p_bonded': float(zb / total),
        'order': int(order),
        'bins': int(bins),
    }


def integrated_autocorrelation(series, *, max_lag=None):
    """Heuristic initial-positive-sequence IAT/ESS; unreliable for short chains."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 8 or np.var(x) < 1e-16:
        return {'tau_checks': None, 'ess': None, 'max_lag': 0}
    centered = x - x.mean()
    max_lag = min(n // 5, 1000) if max_lag is None else min(int(max_lag), n - 1)
    if max_lag < 1:
        return {'tau_checks': None, 'ess': None, 'max_lag': 0}
    denom = float(np.dot(centered, centered))
    tau = 1.0
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        if rho <= 0:
            break
        tau += 2 * rho
    return {'tau_checks': float(tau), 'ess': float(n / tau),
            'max_lag': int(max_lag)}


def batch_means_se(chains, *, block_size):
    """SE from non-overlapping batches, treating batches as approximately independent.

    This is a diagnostic only. Report None when fewer than four batches exist.
    """
    if block_size < 1:
        raise ValueError('block_size must be positive')
    means = []
    for chain in chains:
        x = np.asarray(chain, dtype=float)
        for start in range(0, len(x) - block_size + 1, block_size):
            means.append(float(np.mean(x[start:start + block_size])))
    if len(means) < 4:
        return {'se': None, 'n_batches': len(means), 'block_size': block_size}
    return {'se': float(np.std(means, ddof=1) / np.sqrt(len(means))),
            'n_batches': len(means), 'block_size': block_size}


def observed_radial(samples, reference):
    """Joint topology/radius histogram and radial TV distance."""
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        raise ValueError('samples must have shape (n, 2) and be nonempty')
    edges = np.asarray(reference['bin_edges_A'], dtype=float)
    h0 = np.histogram(arr[arr[:, 0] == 0, 1], bins=edges)[0] / len(arr)
    h1 = np.histogram(arr[arr[:, 0] == 1, 1], bins=edges)[0] / len(arr)
    href = np.asarray(reference['prob_r_by_shell'])
    radial = h0 + h1
    return {
        'observed_unbound_by_shell': h0.tolist(),
        'observed_bonded_by_shell': h1.tolist(),
        'observed_r_by_shell': radial.tolist(),
        'radial_total_variation': float(0.5 * np.sum(np.abs(radial - href))),
        'joint_total_variation': float(0.5 * (
            np.sum(np.abs(h0 - reference['prob_unbound_by_shell'])) +
            np.sum(np.abs(h1 - reference['prob_bonded_by_shell'])))),
    }
