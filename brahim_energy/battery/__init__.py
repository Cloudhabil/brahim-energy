"""Battery optimization and material engine sub-package."""

from __future__ import annotations

from brahim_energy.battery.materials import (
    AgentState,
    AgentTask,
    CostOptimizer,
    DegradationForecaster,
    IndustryStandard,
    IntegrityScorer,
    MaterialEngineAgent,
    MaterialPropertyPredictor,
    MLModel,
    SafetyClassifier,
    StandardCompliance,
)
from brahim_energy.battery.optimizer import (
    BATTERY_CHEMISTRIES,
    MATERIALS_DB,
    ApplicationScale,
    BatteryChemistry,
    BatteryStressResult,
    BrahimBatteryCalculator,
    DurationClass,
    IntegrityRating,
    MaterialIntegrityScore,
    MaterialProperties,
    OptimalBatteryResult,
    StorageCategory,
    visualize_chemistry_deep_dive,
    visualize_comparison,
    visualize_materials,
)

__all__ = [
    # optimizer
    "BatteryChemistry",
    "BatteryStressResult",
    "BrahimBatteryCalculator",
    "MaterialIntegrityScore",
    "MaterialProperties",
    "OptimalBatteryResult",
    "StorageCategory",
    "ApplicationScale",
    "DurationClass",
    "IntegrityRating",
    "BATTERY_CHEMISTRIES",
    "MATERIALS_DB",
    "visualize_comparison",
    "visualize_materials",
    "visualize_chemistry_deep_dive",
    # materials
    "MaterialEngineAgent",
    "MaterialPropertyPredictor",
    "DegradationForecaster",
    "SafetyClassifier",
    "CostOptimizer",
    "IntegrityScorer",
    "IndustryStandard",
    "StandardCompliance",
    "MLModel",
    "AgentState",
    "AgentTask",
]
