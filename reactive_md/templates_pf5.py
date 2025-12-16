# reactive_md/templates_pf5.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PF5Template:
    bond_idx_local: np.ndarray
    angle_idx_local: np.ndarray
    angle_type: np.ndarray

    # Bond params (k, r0) and per-term arrays (k_b, r0)
    bond_coeff_1: tuple[float, float]
    k_b: np.ndarray
    r0: np.ndarray

    # Angle params and per-term arrays (k_theta, theta0 [rad])
    angle_coeff: dict[int, tuple[float, float]]
    k_theta: np.ndarray
    theta0: np.ndarray

    # Nonbonded & charges for PF5 atoms
    pair: dict[str, tuple[float, float]]  # eps, sigma
    q: dict[str, float]                   # partial charges

@dataclass(frozen=True)
class LiFTemplate:
    nb: dict  # {"Li": {"q":..,"eps":..,"sigma":..}, "F": {...}}

def make_pf5_template() -> PF5Template:
    PF5_BOND_IDX_LOCAL = np.array(
        [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]], dtype=np.int32
    )

    PF5_ANGLE_IDX_LOCAL = np.array(
        [
            [1, 0, 4],
            [1, 0, 2],
            [1, 0, 3],
            [1, 0, 5],
            [4, 0, 2],
            [4, 0, 3],
            [4, 0, 5],
            [2, 0, 3],
            [3, 0, 5],
            [2, 0, 5],
        ],
        dtype=np.int32,
    )

    PF5_ANGLE_TYPE = np.array([2, 1, 1, 1, 1, 1, 1, 3, 3, 3], dtype=np.int32)

    # bond_coeff 1 370.4589 1.606
    BOND_COEFF_1 = (370.4589, 1.606)  # (k, r0)

    # angle_coeff type k theta0
    ANGLE_COEFF = {
        1: (139.2209, 90.00),
        2: (34.77535, 180.00),
        3: (139.2209, 120.00),
    }

    # PF5 pair coeffs from settings
    PF5_PAIR = {
        "P": (0.2000000, 3.740),  # (eps, sigma)
        "F": (0.0610000, 3.118),
    }

    # PF5 charges
    PF5_Q = {"P": 1.34, "F": -0.268}

    # Build PF5 bonded parameter arrays
    PF5_K_B = np.full((5,), BOND_COEFF_1[0], dtype=np.float32)
    PF5_R0 = np.full((5,), BOND_COEFF_1[1], dtype=np.float32)

    PF5_K_TH = np.array([ANGLE_COEFF[int(t)][0] for t in PF5_ANGLE_TYPE], dtype=np.float32)
    PF5_TH0_DEG = np.array([ANGLE_COEFF[int(t)][1] for t in PF5_ANGLE_TYPE], dtype=np.float32)
    PF5_TH0 = np.deg2rad(PF5_TH0_DEG).astype(np.float32)

    return PF5Template(
        bond_idx_local=PF5_BOND_IDX_LOCAL,
        angle_idx_local=PF5_ANGLE_IDX_LOCAL,
        angle_type=PF5_ANGLE_TYPE,
        bond_coeff_1=BOND_COEFF_1,
        k_b=PF5_K_B,
        r0=PF5_R0,
        angle_coeff=ANGLE_COEFF,
        k_theta=PF5_K_TH,
        theta0=PF5_TH0,
        pair=PF5_PAIR,
        q=PF5_Q,
    )

def make_lif_template() -> LiFTemplate:
    LIF_NB = {
        "Li": {"q": +1.0, "eps": 0.0182792, "sigma": 2.12645},
        "F":  {"q": -1.0, "eps": 0.72,      "sigma": 2.73295},
    }
    return LiFTemplate(nb=LIF_NB)

