"""Independent shifted-LJ spectator reference for the monatomic A2 audit.

Only for uncharged spectators without bonded interactions.
"""
import numpy as np


def spectator_lj_reference(positions, box, pair, *, sigma, epsilon, cutoff=9.0):
    """Independent NumPy shifted-LJ spectator energy for this monatomic model.

    This reference assumes zero charges and no spectator bonded terms, as in
    the controlled Phase 7 setup. It is NOT a general molecular partition.
    """
    R = np.asarray(positions, dtype=np.float64)
    L = np.asarray(box, dtype=np.float64)
    keep = [i for i in range(len(R)) if i not in set(pair)]
    total = 0.0
    for a, i in enumerate(keep):
        for j in keep[a + 1:]:
            dr = R[i] - R[j]
            dr -= L * np.round(dr / L)
            r = float(np.linalg.norm(dr))
            if r < 1.0e-12:
                raise ValueError('Coincident spectators in reference LJ energy')
            if r >= cutoff:
                continue
            sr6 = (sigma / r) ** 6
            sc6 = (sigma / cutoff) ** 6
            total += 4.0 * epsilon * (sr6 * sr6 - sr6 - sc6 * sc6 + sc6)
    return float(total)

