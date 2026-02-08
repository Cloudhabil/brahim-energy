"""
Brahim Energy Constants
=======================

Canonical source of all Brahim constants used across the energy SDK.
Zero external dependencies — stdlib ``math`` only.

Constants
---------
PHI : float
    Golden ratio (1 + sqrt(5)) / 2.
GENESIS_CONSTANT : float
    Threshold for triggering demand response (2 / 901).
BETA_SECURITY : float
    Optimal compression / peak-reduction ratio (sqrt(5) - 2 ≈ 0.236).
GAMMA : float
    Damping constant (1 / PHI**4).
BRAHIM_SEQUENCE : tuple[int, ...]
    10-element resonance sequence.
BRAHIM_SUM : int
    Mirror constant (214).
BRAHIM_CENTER : int
    Centre constant (107).
LUCAS_NUMBERS : list[int]
    Lucas numbers for dimensions 1-12.
TOTAL_STATES : int
    Sum of Lucas numbers (840).
DIMENSION_NAMES : list[str]
    Names of the 12 cognitive/energy dimensions.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Core constants
# ---------------------------------------------------------------------------

PHI: float = (1 + math.sqrt(5)) / 2              # 1.6180339887498949
GENESIS_CONSTANT: float = 2 / 901                  # 0.00221975...
BETA_SECURITY: float = math.sqrt(5) - 2            # 0.23606797749978969
GAMMA: float = 1 / PHI ** 4                        # 0.14589803375031546
REGULARITY_THRESHOLD: float = 0.0219

# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

BRAHIM_SEQUENCE: tuple[int, ...] = (27, 42, 60, 75, 97, 117, 139, 154, 172, 187)
BRAHIM_SUM: int = 214
BRAHIM_CENTER: int = 107

LUCAS_NUMBERS: list[int] = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]
TOTAL_STATES: int = sum(LUCAS_NUMBERS)  # 840

DIMENSION_NAMES: list[str] = [
    "PERCEPTION",
    "ATTENTION",
    "SECURITY",
    "STABILITY",
    "COMPRESSION",
    "HARMONY",
    "REASONING",
    "PREDICTION",
    "CREATIVITY",
    "WISDOM",
    "INTEGRATION",
    "UNIFICATION",
]

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def D(x: float) -> float:
    """Map a value *x* (0 < x ≤ 1) to its Brahim dimension.

    ``D(x) = -log(x) / log(PHI)``
    """
    if x <= 0:
        raise ValueError("D(x) requires x > 0")
    return -math.log(x) / math.log(PHI)


def x_from_D(d: float) -> float:
    """Inverse of :func:`D` — recover *x* from dimension *d*.

    ``x = 1 / PHI**d``
    """
    return 1.0 / PHI ** d
