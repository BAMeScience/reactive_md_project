"""Independent, dependency-light finite-state MH reference for four A atoms.

Fixed positions are assumed to put all six A-A pairs inside the association
cutoff. The ten states are all matchings of K4: 0, 1, or 2 dimers.
No JAX, JAX-MD, or production reactive_md code is imported.
"""

from __future__ import annotations

import math
from itertools import combinations

PAIRS = tuple(combinations(range(4), 2))
STATES = ((),) + tuple((pair,) for pair in PAIRS) + (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)


def free_pairs(state):
    bonded = {atom for pair in state for atom in pair}
    return tuple(pair for pair in PAIRS if not (set(pair) & bonded))


def energy(state):
    """Synthetic potential with pair-specific and non-additive contributions."""
    weights = {(0, 1): -0.9, (0, 2): 0.3, (0, 3): -0.2,
               (1, 2): 0.6, (1, 3): -0.5, (2, 3): 0.1}
    return sum(weights[pair] for pair in state) + 0.17 * len(state) ** 2


def move_probability(state, target, *, p_association, kT):
    """Off-diagonal transition probability for one attempted MC step."""
    a, b = set(state), set(target)
    if len(b) == len(a) + 1 and a.issubset(b):
        added = next(iter(b - a))
        forward_candidates = free_pairs(state)
        if added not in forward_candidates:
            return 0.0
        q_forward = p_association / len(forward_candidates)
        q_reverse = (1.0 - p_association) / len(target)
    elif len(a) == len(b) + 1 and b.issubset(a):
        removed = next(iter(a - b))
        if removed not in state or removed not in free_pairs(target):
            return 0.0
        q_forward = (1.0 - p_association) / len(state)
        q_reverse = p_association / len(free_pairs(target))
    else:
        return 0.0
    alpha = min(1.0, math.exp(min(0.0, -(energy(target) - energy(state)) / kT
                                     + math.log(q_reverse / q_forward))))
    return q_forward * alpha


def transition_matrix(*, p_association=0.5, kT=1.0):
    if not 0.0 < p_association < 1.0 or kT <= 0:
        raise ValueError("Invalid MC probability or thermal energy")
    matrix = []
    for source in STATES:
        row = [move_probability(source, target, p_association=p_association,
                                kT=kT) if target != source else 0.0
               for target in STATES]
        row[STATES.index(source)] = 1.0 - sum(row)
        matrix.append(tuple(row))
    return tuple(matrix)


def boltzmann_weights(*, kT=1.0):
    unnormalized = [math.exp(-energy(state) / kT) for state in STATES]
    total = sum(unnormalized)
    return tuple(w / total for w in unnormalized)
