"""
brahim-energy — Unified Energy Intelligence SDK
================================================

Grid optimisation, battery materials, demand response, CO2 tracking,
and photosynthesis / MOF carbon capture.  **Zero external dependencies.**

Quick start::

    from brahim_energy.grid.eu import BrahimGridStressCalculator
    calc = BrahimGridStressCalculator("DE")
    result = calc.simulate_24h(40_000)

    from brahim_energy.battery import BrahimBatteryCalculator
    bc = BrahimBatteryCalculator()
    best = bc.find_optimal_battery("grid_storage", "medium", budget_per_kwh=200)

    from brahim_energy.carbon import PhotosynthesisCascadeAnalyzer
    pca = PhotosynthesisCascadeAnalyzer()
    cascade = pca.analyze_cascade()
"""

from brahim_energy.constants import (
    BETA_SECURITY,
    BRAHIM_CENTER,
    BRAHIM_SEQUENCE,
    BRAHIM_SUM,
    DIMENSION_NAMES,
    GAMMA,
    GENESIS_CONSTANT,
    LUCAS_NUMBERS,
    PHI,
    TOTAL_STATES,
    D,
    x_from_D,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # constants
    "BETA_SECURITY",
    "BRAHIM_CENTER",
    "BRAHIM_SEQUENCE",
    "BRAHIM_SUM",
    "D",
    "DIMENSION_NAMES",
    "GAMMA",
    "GENESIS_CONSTANT",
    "LUCAS_NUMBERS",
    "PHI",
    "TOTAL_STATES",
    "x_from_D",
]
