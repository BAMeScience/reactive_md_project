"""Independent finite-state reference and uncertainty estimates for A2 MH sampling.

This module intentionally imports no production reactive_md or JAX code.
"""
from __future__ import annotations

import math
import numpy as np

from validation.a2_mh_exact_reference import STATES, free_pairs


def boltzmann(energies, kt: float) -> np.ndarray:
    e = np.asarray(energies, dtype=np.float64)
    if e.shape != (len(STATES),) or not np.isfinite(e).all() or not np.isfinite(kt) or kt <= 0:
        raise ValueError('Expected 10 finite energies and positive finite kT')
    w = np.exp(-(e - e.min()) / kt)
    return w / w.sum()


def transition_matrix(energies, *, p_association: float, kt: float) -> np.ndarray:
    """Exact transition kernel, constructed independently of the production handler."""
    if not 0.0 < p_association < 1.0:
        raise ValueError('Association direction probability must lie in (0,1)')
    energies = np.asarray(energies, dtype=np.float64)
    pi = boltzmann(energies, kt)
    n = len(STATES)
    p = np.zeros((n, n), dtype=np.float64)
    for i, source in enumerate(STATES):
        for j, target in enumerate(STATES):
            if i == j:
                continue
            added = set(target) - set(source)
            removed = set(source) - set(target)
            if len(added) == 1 and not removed and len(target) == len(source) + 1:
                pair = next(iter(added))
                if pair not in free_pairs(source):
                    continue
                q_forward = p_association / len(free_pairs(source))
                q_reverse = (1.0 - p_association) / len(target)
            elif len(removed) == 1 and not added and len(target) == len(source) - 1:
                pair = next(iter(removed))
                if pair not in free_pairs(target):
                    continue
                q_forward = (1.0 - p_association) / len(source)
                q_reverse = p_association / len(free_pairs(target))
            else:
                continue
            log_alpha = -(energies[j] - energies[i]) / kt + math.log(q_reverse / q_forward)
            p[i, j] = q_forward * math.exp(min(0.0, log_alpha))
        p[i, i] = 1.0 - p[i].sum()
    if np.min(np.diag(p)) < -1e-12:
        raise AssertionError('Invalid transition matrix')
    if not np.allclose(pi[:, None] * p, pi[None, :] * p.T, atol=1e-12):
        raise AssertionError('Exact kernel violates detailed balance')
    return p


def occupation_standard_errors(p: np.ndarray, pi: np.ndarray, *, samples_per_chain: int,
                               chains: int) -> np.ndarray:
    """Exact stationary variance of empirical state frequencies for independent chains.

    For state i, Cov(1[X_0=i], 1[X_h=i]) = pi_i (P^h_ii - pi_i).
    Includes all lags with the finite-sample (1-h/N) correction.
    Assumes burn-in has brought each chain close to stationarity.
    """
    if samples_per_chain < 1 or chains < 1:
        raise ValueError('Samples and chains must be positive')
    n = samples_per_chain
    pi = np.asarray(pi, dtype=np.float64)
    variance = pi * (1.0 - pi) / n
    power = np.eye(len(pi), dtype=np.float64)
    for lag in range(1, n):
        power = power @ p
        variance += (2.0 / n) * (1.0 - lag / n) * pi * (np.diag(power) - pi)
    return np.sqrt(np.maximum(variance / chains, 0.0))


def sampling_diagnostics(counts, energies, *, p_association: float, kt: float,
                         samples_per_chain: int, chains: int) -> dict:
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (len(STATES),) or np.any(counts < 0):
        raise ValueError('Invalid state counts')
    if counts.sum() != samples_per_chain * chains:
        raise ValueError('Counts do not match requested sample total')
    pi = boltzmann(energies, kt)
    kernel = transition_matrix(energies, p_association=p_association, kt=kt)
    se = occupation_standard_errors(kernel, pi, samples_per_chain=samples_per_chain, chains=chains)
    observed = counts / counts.sum()
    errors = np.abs(observed - pi)
    # A floor prevents spurious failures for very rare states in short chains.
    bounds = np.maximum(5.0 * se, 0.01)
    tv = 0.5 * errors.sum()
    return dict(expected=pi, observed=observed, standard_error=se, errors=errors,
                bounds=bounds, total_variation=float(tv),
                within_bounds=bool(np.all(errors <= bounds)),
                kernel=kernel)
