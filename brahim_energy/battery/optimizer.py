#!/usr/bin/env python3
"""
Brahim Battery Materials Optimizer
===================================

Applies Brahim's mathematical framework to find optimal battery materials
for grid-scale energy storage, integrating integrity principles.

CORE FORMULAS:

1. Battery Stress (degradation prediction):
   B(t) = Σ(1/(max_cycles - used_cycles)²) × exp(-λ × depth_of_discharge)

2. Material Resonance (grid matching):
   R = Σ(response_time × grid_frequency) × exp(-λ × mismatch)

3. Integrity Score (ethical sustainability):
   I = (1 - supply_risk) × (1 - environmental_harm) × transparency

4. Optimal Material Score:
   M = (energy_density × power_density × cycle_life × efficiency) /
       (cost × B(t)) × R × I

STORAGE TECHNOLOGIES EVALUATED:
├── Electrochemical
│   ├── Lithium-ion (LFP, NMC, NCA, LTO)
│   ├── Sodium-ion
│   ├── Solid-state
│   └── Flow batteries (Vanadium, Zinc-Bromine, Iron)
├── Metal-Air
│   ├── Iron-air
│   ├── Zinc-air
│   └── Aluminum-air
├── Mechanical
│   ├── Pumped hydro
│   ├── Compressed air (CAES)
│   └── Gravity storage
├── Thermal
│   ├── Molten salt
│   ├── Ice storage
│   └── Hot water
└── Chemical
    ├── Green hydrogen
    ├── Ammonia
    └── Synthetic methane

INTEGRITY PRINCIPLES:
- Transparency: Open formulas, no black-box decisions
- Consistency: Same evaluation criteria for all materials
- Sustainability: Environmental and social impact weighted
- Availability: No single-source dependencies (anti-oppression)

AUTHOR: GPIA Cognitive Ecosystem
DATE: 2026-01-26
VERSION: 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from brahim_energy.constants import BETA_SECURITY, GENESIS_CONSTANT, PHI

# =============================================================================
# BATTERY-SPECIFIC DERIVED CONSTANTS
# =============================================================================

# Fibonacci sequence used for charge/discharge intervals
FIBONACCI_INTERVALS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# Battery-specific Brahim extensions
CYCLE_RESONANCE = PHI / 1000   # Optimal cycle frequency
DOD_OPTIMAL = 1 - 2 * BETA_SECURITY  # 52.8% usable (23.6% margin each end)


# =============================================================================
# ENUMS AND CLASSIFICATIONS
# =============================================================================

class StorageCategory(Enum):
    """Main storage technology categories."""
    ELECTROCHEMICAL = "electrochemical"
    METAL_AIR = "metal_air"
    MECHANICAL = "mechanical"
    THERMAL = "thermal"
    CHEMICAL = "chemical"


class ApplicationScale(Enum):
    """Scale of application."""
    RESIDENTIAL = "residential"      # 5-20 kWh
    COMMERCIAL = "commercial"        # 100 kWh - 1 MWh
    UTILITY = "utility"              # 1 MWh - 1 GWh
    GRID = "grid"                    # 1 GWh+


class DurationClass(Enum):
    """Storage duration classification."""
    SHORT = "short"      # < 4 hours
    MEDIUM = "medium"    # 4-12 hours
    LONG = "long"        # 12-24 hours
    SEASONAL = "seasonal"  # Days to months


class IntegrityRating(Enum):
    """Ethical/sustainability rating."""
    EXCELLENT = "excellent"   # Score > 0.8
    GOOD = "good"             # 0.6 - 0.8
    MODERATE = "moderate"     # 0.4 - 0.6
    POOR = "poor"             # 0.2 - 0.4
    CRITICAL = "critical"     # < 0.2


# =============================================================================
# MATERIAL DATA STRUCTURES
# =============================================================================

@dataclass
class MaterialProperties:
    """Raw material properties for battery components."""
    name: str
    symbol: str
    abundance_ppm: float          # Earth's crust abundance
    price_usd_kg: float           # Current market price
    supply_concentration: float    # 0-1 (1 = single country controls 90%+)
    mining_co2_kg_per_kg: float   # CO2 footprint of extraction
    recyclability: float          # 0-1 (1 = fully recyclable)
    toxicity: float               # 0-1 (1 = highly toxic)
    countries: List[str]          # Major producing countries


# Key battery materials database
MATERIALS_DB: Dict[str, MaterialProperties] = {
    "lithium": MaterialProperties(
        name="Lithium", symbol="Li",
        abundance_ppm=20,
        price_usd_kg=70,
        supply_concentration=0.75,  # Australia, Chile, China dominate
        mining_co2_kg_per_kg=15,
        recyclability=0.5,
        toxicity=0.2,
        countries=["Australia", "Chile", "China", "Argentina"],
    ),
    "cobalt": MaterialProperties(
        name="Cobalt", symbol="Co",
        abundance_ppm=25,
        price_usd_kg=35,
        supply_concentration=0.90,  # DRC controls 70%
        mining_co2_kg_per_kg=35,
        recyclability=0.6,
        toxicity=0.6,
        countries=["DRC", "Russia", "Australia"],
    ),
    "nickel": MaterialProperties(
        name="Nickel", symbol="Ni",
        abundance_ppm=84,
        price_usd_kg=18,
        supply_concentration=0.50,
        mining_co2_kg_per_kg=12,
        recyclability=0.7,
        toxicity=0.4,
        countries=["Indonesia", "Philippines", "Russia", "Canada"],
    ),
    "iron": MaterialProperties(
        name="Iron", symbol="Fe",
        abundance_ppm=50000,  # Very abundant
        price_usd_kg=0.1,
        supply_concentration=0.20,  # Well distributed
        mining_co2_kg_per_kg=2,
        recyclability=0.95,
        toxicity=0.05,
        countries=["Australia", "Brazil", "China", "India"],
    ),
    "sodium": MaterialProperties(
        name="Sodium", symbol="Na",
        abundance_ppm=23600,  # Extremely abundant
        price_usd_kg=0.3,
        supply_concentration=0.05,  # Everywhere (salt)
        mining_co2_kg_per_kg=1,
        recyclability=0.9,
        toxicity=0.1,
        countries=["Global"],  # Available everywhere
    ),
    "vanadium": MaterialProperties(
        name="Vanadium", symbol="V",
        abundance_ppm=120,
        price_usd_kg=25,
        supply_concentration=0.65,
        mining_co2_kg_per_kg=20,
        recyclability=0.85,
        toxicity=0.3,
        countries=["China", "Russia", "South Africa"],
    ),
    "zinc": MaterialProperties(
        name="Zinc", symbol="Zn",
        abundance_ppm=70,
        price_usd_kg=2.5,
        supply_concentration=0.35,
        mining_co2_kg_per_kg=4,
        recyclability=0.9,
        toxicity=0.15,
        countries=["China", "Peru", "Australia", "USA"],
    ),
    "manganese": MaterialProperties(
        name="Manganese", symbol="Mn",
        abundance_ppm=950,
        price_usd_kg=2,
        supply_concentration=0.40,
        mining_co2_kg_per_kg=3,
        recyclability=0.8,
        toxicity=0.2,
        countries=["South Africa", "Australia", "Gabon", "China"],
    ),
    "graphite": MaterialProperties(
        name="Graphite", symbol="C",
        abundance_ppm=200,
        price_usd_kg=1.5,
        supply_concentration=0.70,  # China dominates
        mining_co2_kg_per_kg=5,
        recyclability=0.7,
        toxicity=0.05,
        countries=["China", "Mozambique", "Brazil"],
    ),
    "aluminum": MaterialProperties(
        name="Aluminum", symbol="Al",
        abundance_ppm=82300,  # Third most abundant
        price_usd_kg=2.3,
        supply_concentration=0.25,
        mining_co2_kg_per_kg=12,  # Energy intensive
        recyclability=0.95,
        toxicity=0.05,
        countries=["China", "India", "Russia", "Canada"],
    ),
}


@dataclass
class BatteryChemistry:
    """Complete battery chemistry specification."""
    name: str
    code: str
    category: StorageCategory

    # Performance metrics
    energy_density_wh_kg: float      # Gravimetric
    energy_density_wh_l: float       # Volumetric
    power_density_w_kg: float
    cycle_life: int                  # Cycles to 80% capacity
    calendar_life_years: float
    round_trip_efficiency: float     # 0-1
    self_discharge_per_month: float  # 0-1

    # Operating parameters
    dod_max: float                   # Maximum depth of discharge
    charge_rate_c: float             # Maximum C-rate
    discharge_rate_c: float
    temp_min_c: float
    temp_max_c: float

    # Economics
    cost_per_kwh: float              # $/kWh installed
    cost_trend: float                # Annual % decrease

    # Materials
    primary_materials: List[str]
    critical_materials: List[str]    # Supply-constrained

    # Application fit
    duration_class: DurationClass
    best_applications: List[ApplicationScale]

    # Response characteristics
    response_time_ms: float          # Time to full power
    ramp_rate_per_second: float      # % per second


# =============================================================================
# BATTERY CHEMISTRY DATABASE
# =============================================================================

BATTERY_CHEMISTRIES: Dict[str, BatteryChemistry] = {
    # === LITHIUM-ION VARIANTS ===
    "lfp": BatteryChemistry(
        name="Lithium Iron Phosphate",
        code="LFP",
        category=StorageCategory.ELECTROCHEMICAL,
        energy_density_wh_kg=160,
        energy_density_wh_l=325,
        power_density_w_kg=2500,
        cycle_life=6000,
        calendar_life_years=15,
        round_trip_efficiency=0.94,
        self_discharge_per_month=0.02,
        dod_max=0.90,
        charge_rate_c=1.0,
        discharge_rate_c=3.0,
        temp_min_c=-20,
        temp_max_c=60,
        cost_per_kwh=150,
        cost_trend=-0.08,
        primary_materials=["lithium", "iron", "graphite"],
        critical_materials=["lithium"],
        duration_class=DurationClass.SHORT,
        best_applications=[ApplicationScale.UTILITY, ApplicationScale.COMMERCIAL],
        response_time_ms=20,
        ramp_rate_per_second=10.0,
    ),
    "nmc": BatteryChemistry(
        name="Lithium Nickel Manganese Cobalt",
        code="NMC",
        category=StorageCategory.ELECTROCHEMICAL,
        energy_density_wh_kg=250,
        energy_density_wh_l=650,
        power_density_w_kg=1500,
        cycle_life=3000,
        calendar_life_years=12,
        round_trip_efficiency=0.92,
        self_discharge_per_month=0.03,
        dod_max=0.85,
        charge_rate_c=0.7,
        discharge_rate_c=2.0,
        temp_min_c=-10,
        temp_max_c=45,
        cost_per_kwh=180,
        cost_trend=-0.10,
        primary_materials=["lithium", "nickel", "manganese", "cobalt", "graphite"],
        critical_materials=["lithium", "cobalt", "nickel"],
        duration_class=DurationClass.SHORT,
        best_applications=[ApplicationScale.RESIDENTIAL, ApplicationScale.COMMERCIAL],
        response_time_ms=20,
        ramp_rate_per_second=8.0,
    ),

    # === SODIUM-ION ===
    "sodium_ion": BatteryChemistry(
        name="Sodium-Ion",
        code="Na-ion",
        category=StorageCategory.ELECTROCHEMICAL,
        energy_density_wh_kg=140,
        energy_density_wh_l=280,
        power_density_w_kg=2000,
        cycle_life=5000,
        calendar_life_years=15,
        round_trip_efficiency=0.92,
        self_discharge_per_month=0.02,
        dod_max=0.90,
        charge_rate_c=1.0,
        discharge_rate_c=3.0,
        temp_min_c=-40,  # Excellent cold performance
        temp_max_c=60,
        cost_per_kwh=80,  # Much cheaper
        cost_trend=-0.15,
        primary_materials=["sodium", "iron", "manganese"],
        critical_materials=[],  # No critical materials!
        duration_class=DurationClass.SHORT,
        best_applications=[ApplicationScale.UTILITY, ApplicationScale.GRID],
        response_time_ms=25,
        ramp_rate_per_second=10.0,
    ),

    # === SOLID-STATE ===
    "solid_state": BatteryChemistry(
        name="Solid-State Lithium",
        code="SSB",
        category=StorageCategory.ELECTROCHEMICAL,
        energy_density_wh_kg=400,
        energy_density_wh_l=900,
        power_density_w_kg=3000,
        cycle_life=10000,
        calendar_life_years=20,
        round_trip_efficiency=0.96,
        self_discharge_per_month=0.005,
        dod_max=0.95,
        charge_rate_c=3.0,
        discharge_rate_c=5.0,
        temp_min_c=-30,
        temp_max_c=80,
        cost_per_kwh=350,  # Still expensive (2026)
        cost_trend=-0.20,
        primary_materials=["lithium"],
        critical_materials=["lithium"],
        duration_class=DurationClass.SHORT,
        best_applications=[ApplicationScale.RESIDENTIAL, ApplicationScale.COMMERCIAL],
        response_time_ms=10,
        ramp_rate_per_second=15.0,
    ),

    # === FLOW BATTERIES ===
    "vanadium_flow": BatteryChemistry(
        name="Vanadium Redox Flow",
        code="VRFB",
        category=StorageCategory.ELECTROCHEMICAL,
        energy_density_wh_kg=25,
        energy_density_wh_l=35,
        power_density_w_kg=100,
        cycle_life=20000,
        calendar_life_years=25,
        round_trip_efficiency=0.75,
        self_discharge_per_month=0.001,
        dod_max=1.0,  # Full DoD possible
        charge_rate_c=0.25,
        discharge_rate_c=0.25,
        temp_min_c=10,
        temp_max_c=40,
        cost_per_kwh=400,
        cost_trend=-0.05,
        primary_materials=["vanadium"],
        critical_materials=["vanadium"],
        duration_class=DurationClass.LONG,
        best_applications=[ApplicationScale.UTILITY, ApplicationScale.GRID],
        response_time_ms=100,
        ramp_rate_per_second=2.0,
    ),
    "iron_flow": BatteryChemistry(
        name="Iron-Air Flow",
        code="Fe-Air",
        category=StorageCategory.ELECTROCHEMICAL,
        energy_density_wh_kg=100,
        energy_density_wh_l=150,
        power_density_w_kg=50,
        cycle_life=10000,
        calendar_life_years=30,
        round_trip_efficiency=0.50,  # Lower efficiency
        self_discharge_per_month=0.001,
        dod_max=1.0,
        charge_rate_c=0.1,
        discharge_rate_c=0.1,
        temp_min_c=0,
        temp_max_c=50,
        cost_per_kwh=25,  # Extremely cheap
        cost_trend=-0.10,
        primary_materials=["iron"],
        critical_materials=[],  # No critical materials
        duration_class=DurationClass.SEASONAL,
        best_applications=[ApplicationScale.GRID],
        response_time_ms=1000,
        ramp_rate_per_second=0.5,
    ),
    "zinc_bromine": BatteryChemistry(
        name="Zinc-Bromine Flow",
        code="Zn-Br",
        category=StorageCategory.ELECTROCHEMICAL,
        energy_density_wh_kg=60,
        energy_density_wh_l=70,
        power_density_w_kg=150,
        cycle_life=10000,
        calendar_life_years=20,
        round_trip_efficiency=0.70,
        self_discharge_per_month=0.005,
        dod_max=1.0,
        charge_rate_c=0.2,
        discharge_rate_c=0.3,
        temp_min_c=5,
        temp_max_c=45,
        cost_per_kwh=200,
        cost_trend=-0.08,
        primary_materials=["zinc"],
        critical_materials=[],
        duration_class=DurationClass.MEDIUM,
        best_applications=[ApplicationScale.COMMERCIAL, ApplicationScale.UTILITY],
        response_time_ms=200,
        ramp_rate_per_second=1.5,
    ),

    # === METAL-AIR ===
    "iron_air": BatteryChemistry(
        name="Iron-Air",
        code="Fe-Air",
        category=StorageCategory.METAL_AIR,
        energy_density_wh_kg=1200,  # Theoretical very high
        energy_density_wh_l=2000,
        power_density_w_kg=50,
        cycle_life=3000,
        calendar_life_years=20,
        round_trip_efficiency=0.45,
        self_discharge_per_month=0.01,
        dod_max=0.80,
        charge_rate_c=0.05,
        discharge_rate_c=0.1,
        temp_min_c=10,
        temp_max_c=50,
        cost_per_kwh=20,  # Target price
        cost_trend=-0.15,
        primary_materials=["iron"],
        critical_materials=[],
        duration_class=DurationClass.SEASONAL,
        best_applications=[ApplicationScale.GRID],
        response_time_ms=5000,
        ramp_rate_per_second=0.2,
    ),

    # === MECHANICAL ===
    "pumped_hydro": BatteryChemistry(
        name="Pumped Hydro Storage",
        code="PHS",
        category=StorageCategory.MECHANICAL,
        energy_density_wh_kg=1,  # Very low
        energy_density_wh_l=1,
        power_density_w_kg=10,
        cycle_life=50000,
        calendar_life_years=80,
        round_trip_efficiency=0.80,
        self_discharge_per_month=0.001,
        dod_max=1.0,
        charge_rate_c=0.1,
        discharge_rate_c=0.2,
        temp_min_c=-20,
        temp_max_c=40,
        cost_per_kwh=150,  # Very site-dependent
        cost_trend=0.0,
        primary_materials=["iron", "aluminum"],  # Turbines
        critical_materials=[],
        duration_class=DurationClass.LONG,
        best_applications=[ApplicationScale.GRID],
        response_time_ms=60000,  # Minutes to start
        ramp_rate_per_second=0.5,
    ),
    "gravity_storage": BatteryChemistry(
        name="Gravity Energy Storage",
        code="GES",
        category=StorageCategory.MECHANICAL,
        energy_density_wh_kg=0.5,
        energy_density_wh_l=2,
        power_density_w_kg=5,
        cycle_life=100000,
        calendar_life_years=50,
        round_trip_efficiency=0.85,
        self_discharge_per_month=0.0,
        dod_max=1.0,
        charge_rate_c=0.2,
        discharge_rate_c=0.3,
        temp_min_c=-40,
        temp_max_c=60,
        cost_per_kwh=180,
        cost_trend=-0.05,
        primary_materials=["iron"],  # Weights
        critical_materials=[],
        duration_class=DurationClass.MEDIUM,
        best_applications=[ApplicationScale.UTILITY, ApplicationScale.GRID],
        response_time_ms=2000,
        ramp_rate_per_second=1.0,
    ),
    "compressed_air": BatteryChemistry(
        name="Compressed Air (CAES)",
        code="CAES",
        category=StorageCategory.MECHANICAL,
        energy_density_wh_kg=30,
        energy_density_wh_l=6,
        power_density_w_kg=20,
        cycle_life=30000,
        calendar_life_years=40,
        round_trip_efficiency=0.70,
        self_discharge_per_month=0.005,
        dod_max=1.0,
        charge_rate_c=0.1,
        discharge_rate_c=0.2,
        temp_min_c=-20,
        temp_max_c=50,
        cost_per_kwh=120,
        cost_trend=-0.03,
        primary_materials=["iron"],
        critical_materials=[],
        duration_class=DurationClass.LONG,
        best_applications=[ApplicationScale.GRID],
        response_time_ms=300000,  # 5 minutes
        ramp_rate_per_second=0.3,
    ),

    # === THERMAL ===
    "molten_salt": BatteryChemistry(
        name="Molten Salt Thermal",
        code="MST",
        category=StorageCategory.THERMAL,
        energy_density_wh_kg=100,
        energy_density_wh_l=200,
        power_density_w_kg=20,
        cycle_life=30000,
        calendar_life_years=30,
        round_trip_efficiency=0.40,  # Thermal to electric
        self_discharge_per_month=0.02,
        dod_max=1.0,
        charge_rate_c=0.1,
        discharge_rate_c=0.2,
        temp_min_c=200,  # Must stay hot
        temp_max_c=600,
        cost_per_kwh=30,
        cost_trend=-0.05,
        primary_materials=["sodium"],  # Sodium nitrate
        critical_materials=[],
        duration_class=DurationClass.LONG,
        best_applications=[ApplicationScale.GRID],
        response_time_ms=60000,
        ramp_rate_per_second=0.2,
    ),

    # === CHEMICAL (HYDROGEN) ===
    "green_hydrogen": BatteryChemistry(
        name="Green Hydrogen",
        code="H2",
        category=StorageCategory.CHEMICAL,
        energy_density_wh_kg=33000,  # Very high
        energy_density_wh_l=3000,    # Compressed
        power_density_w_kg=500,
        cycle_life=50000,
        calendar_life_years=25,
        round_trip_efficiency=0.35,  # Electrolysis + fuel cell
        self_discharge_per_month=0.005,
        dod_max=1.0,
        charge_rate_c=0.2,
        discharge_rate_c=0.5,
        temp_min_c=-40,
        temp_max_c=80,
        cost_per_kwh=500,  # Still expensive
        cost_trend=-0.12,
        primary_materials=["iron", "nickel"],  # Electrolyzer
        critical_materials=[],
        duration_class=DurationClass.SEASONAL,
        best_applications=[ApplicationScale.GRID],
        response_time_ms=10000,
        ramp_rate_per_second=0.5,
    ),
}


# =============================================================================
# BRAHIM BATTERY CALCULATOR
# =============================================================================

@dataclass
class BatteryStressResult:
    """Result of battery stress calculation."""
    chemistry: str
    stress: float
    cycles_remaining: int
    health_percent: float
    optimal_dod: float
    recommended_charge_rate: float
    temperature_factor: float


@dataclass
class MaterialIntegrityScore:
    """Integrity score for a material."""
    material: str
    supply_risk: float       # 0 = diverse supply, 1 = monopoly
    environmental: float     # 0 = clean, 1 = harmful
    social_risk: float       # 0 = ethical, 1 = conflict
    recyclability: float     # 0 = waste, 1 = circular
    overall: float           # Combined integrity score
    rating: IntegrityRating


@dataclass
class OptimalBatteryResult:
    """Result of optimal battery selection."""
    rank: int
    chemistry: BatteryChemistry
    brahim_score: float
    integrity_score: float
    grid_resonance: float
    cost_effectiveness: float
    co2_savings_kg_per_kwh: float
    recommendation: str


class BrahimBatteryCalculator:
    """
    Brahim's Battery Materials Optimizer.

    Uses mathematical framework to find optimal battery chemistry
    based on application requirements and integrity principles.
    """

    def __init__(self):
        self.chemistries = BATTERY_CHEMISTRIES
        self.materials = MATERIALS_DB

    def calculate_battery_stress(
        self,
        chemistry_code: str,
        current_cycles: int,
        depth_of_discharge: float,
        temperature_c: float,
    ) -> BatteryStressResult:
        """
        Calculate battery stress using Brahim formula.

        B(t) = Σ(1/(max_cycles - used_cycles)²) × exp(-λ × DoD)
        """
        chem = self.chemistries.get(chemistry_code)
        if not chem:
            raise ValueError(f"Unknown chemistry: {chemistry_code}")

        # Remaining cycles
        remaining = max(1, chem.cycle_life - current_cycles)

        # Base stress (cycle degradation)
        cycle_stress = 1.0 / (remaining ** 2)

        # DoD factor (deeper discharge = more stress)
        dod_factor = math.exp(-GENESIS_CONSTANT * depth_of_discharge * 1000)

        # Temperature factor
        optimal_temp = (chem.temp_min_c + chem.temp_max_c) / 2
        temp_deviation = abs(temperature_c - optimal_temp) / 20
        temp_factor = 1 + temp_deviation * BETA_SECURITY

        # Combined stress
        stress = cycle_stress * dod_factor * temp_factor

        # Health estimation
        health = 100 * (remaining / chem.cycle_life) * (1 - stress * 100)
        health = max(0, min(100, health))

        # Optimal DoD for this state
        if stress > 0.01:
            optimal_dod = chem.dod_max * DOD_OPTIMAL  # Reduce DoD under stress
        else:
            optimal_dod = chem.dod_max

        # Recommended charge rate
        if stress > 0.001:
            recommended_c = chem.charge_rate_c * 0.5
        else:
            recommended_c = chem.charge_rate_c

        return BatteryStressResult(
            chemistry=chemistry_code,
            stress=stress,
            cycles_remaining=remaining,
            health_percent=health,
            optimal_dod=optimal_dod,
            recommended_charge_rate=recommended_c,
            temperature_factor=temp_factor,
        )

    def calculate_material_integrity(
        self,
        material_name: str,
    ) -> MaterialIntegrityScore:
        """
        Calculate integrity score for a battery material.

        Integrity = (1 - supply_risk) × (1 - environmental) × recyclability
        """
        mat = self.materials.get(material_name)
        if not mat:
            # Unknown material - assume moderate
            return MaterialIntegrityScore(
                material=material_name,
                supply_risk=0.5,
                environmental=0.5,
                social_risk=0.5,
                recyclability=0.5,
                overall=0.5,
                rating=IntegrityRating.MODERATE,
            )

        supply_risk = mat.supply_concentration
        environmental = (mat.mining_co2_kg_per_kg / 50) * 0.5 + mat.toxicity * 0.5
        environmental = min(1.0, environmental)

        # Social risk based on source countries
        high_risk_countries = ["DRC", "Russia", "Myanmar"]
        social_risk = sum(1 for c in mat.countries if c in high_risk_countries) / max(len(mat.countries), 1)

        recyclability = mat.recyclability

        # Brahim integrity formula
        overall = (1 - supply_risk) * (1 - environmental) * (1 - social_risk * 0.5) * recyclability
        overall = math.pow(overall, 0.5)  # Square root to normalize

        # Rating
        if overall > 0.8:
            rating = IntegrityRating.EXCELLENT
        elif overall > 0.6:
            rating = IntegrityRating.GOOD
        elif overall > 0.4:
            rating = IntegrityRating.MODERATE
        elif overall > 0.2:
            rating = IntegrityRating.POOR
        else:
            rating = IntegrityRating.CRITICAL

        return MaterialIntegrityScore(
            material=material_name,
            supply_risk=supply_risk,
            environmental=environmental,
            social_risk=social_risk,
            recyclability=recyclability,
            overall=overall,
            rating=rating,
        )

    def calculate_chemistry_integrity(
        self,
        chemistry_code: str,
    ) -> float:
        """Calculate overall integrity for a battery chemistry."""
        chem = self.chemistries.get(chemistry_code)
        if not chem:
            return 0.5

        # Weight critical materials more heavily
        scores = []
        for mat in chem.primary_materials:
            integrity = self.calculate_material_integrity(mat)
            weight = 2.0 if mat in chem.critical_materials else 1.0
            scores.append((integrity.overall, weight))

        if not scores:
            return 0.8  # No materials = probably good

        total_weight = sum(w for _, w in scores)
        weighted_sum = sum(s * w for s, w in scores)

        return weighted_sum / total_weight

    def calculate_grid_resonance(
        self,
        chemistry_code: str,
        grid_frequency_hz: float = 50/60,  # EU/US ratio
        renewable_variability: float = 0.3,
    ) -> float:
        """
        Calculate how well battery responds to grid needs.

        R = exp(-λ × |response_time - optimal_time|)
        """
        chem = self.chemistries.get(chemistry_code)
        if not chem:
            return 0.5

        # Optimal response time for renewable variability
        optimal_response_ms = 1000 / (renewable_variability * 10)

        # Response mismatch
        mismatch = abs(chem.response_time_ms - optimal_response_ms) / 1000

        # Resonance score
        resonance = math.exp(-GENESIS_CONSTANT * mismatch * 100)

        # Ramp rate bonus
        ramp_bonus = min(1.0, chem.ramp_rate_per_second / 5)

        return resonance * 0.7 + ramp_bonus * 0.3

    def find_optimal_battery(
        self,
        application: ApplicationScale,
        duration: DurationClass,
        budget_per_kwh: float = 500,
        prioritize_integrity: bool = True,
        min_efficiency: float = 0.5,
    ) -> List[OptimalBatteryResult]:
        """
        Find optimal battery chemistry for given requirements.

        Brahim Optimal Score:
        M = (energy × power × cycles × efficiency) / (cost × stress) × resonance × integrity
        """
        results = []

        for code, chem in self.chemistries.items():
            # Filter by application and duration
            if application not in chem.best_applications:
                continue
            if chem.duration_class != duration and duration != DurationClass.SHORT:
                # Allow flexibility for short duration
                if duration == DurationClass.MEDIUM and chem.duration_class == DurationClass.SHORT:
                    pass
                elif duration == DurationClass.LONG and chem.duration_class in [DurationClass.MEDIUM, DurationClass.SEASONAL]:
                    pass
                else:
                    continue

            # Filter by budget
            if chem.cost_per_kwh > budget_per_kwh:
                continue

            # Filter by efficiency
            if chem.round_trip_efficiency < min_efficiency:
                continue

            # Calculate scores
            integrity = self.calculate_chemistry_integrity(code)
            resonance = self.calculate_grid_resonance(code)

            # Brahim optimal score
            performance = (
                chem.energy_density_wh_kg / 300 *  # Normalized
                chem.power_density_w_kg / 2000 *
                chem.cycle_life / 10000 *
                chem.round_trip_efficiency
            )

            cost_factor = chem.cost_per_kwh / 200  # Normalized

            # Combined score
            brahim_score = (performance / cost_factor) * resonance

            if prioritize_integrity:
                brahim_score *= (0.5 + integrity * 0.5)

            # CO2 savings (vs coal at 900g/kWh)
            co2_savings = 0.9 * chem.round_trip_efficiency  # kg CO2 per kWh cycled

            # Recommendation
            if brahim_score > 0.5:
                rec = "Highly recommended"
            elif brahim_score > 0.2:
                rec = "Good option"
            elif brahim_score > 0.1:
                rec = "Consider for specific needs"
            else:
                rec = "Not optimal for requirements"

            results.append(OptimalBatteryResult(
                rank=0,
                chemistry=chem,
                brahim_score=brahim_score,
                integrity_score=integrity,
                grid_resonance=resonance,
                cost_effectiveness=performance / cost_factor,
                co2_savings_kg_per_kwh=co2_savings,
                recommendation=rec,
            ))

        # Sort by Brahim score
        results.sort(key=lambda r: r.brahim_score, reverse=True)

        # Assign ranks
        for i, r in enumerate(results):
            r.rank = i + 1

        return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_comparison(results: List[OptimalBatteryResult]) -> str:
    """Generate ASCII comparison of battery options."""
    lines = []
    lines.append("")
    lines.append("  Optimal Battery Selection (Brahim Calculator)")
    lines.append("  " + "=" * 75)
    lines.append("")
    lines.append("  Rank  Chemistry          Score    Integrity  $/kWh  Efficiency  Duration")
    lines.append("  " + "-" * 75)

    for r in results[:10]:
        chem = r.chemistry

        # Score bar
        bar_len = int(r.brahim_score * 20)
        bar = "\u2588" * bar_len + "\u2591" * (10 - bar_len)

        integrity_stars = "\u2605" * int(r.integrity_score * 5) + "\u2606" * (5 - int(r.integrity_score * 5))

        lines.append(
            f"  {r.rank:2d}.   {chem.name[:18]:18s}  {bar}  {integrity_stars}  "
            f"${chem.cost_per_kwh:3.0f}   {chem.round_trip_efficiency*100:4.0f}%      {chem.duration_class.value[:6]}"
        )

    lines.append("")
    lines.append("  Score: \u2588 = Brahim optimization score (higher = better)")
    lines.append("  Integrity: \u2605 = Supply security + sustainability + ethics")
    lines.append("")

    return "\n".join(lines)


def visualize_materials() -> str:
    """Show material integrity analysis."""
    calc = BrahimBatteryCalculator()

    lines = []
    lines.append("")
    lines.append("  Battery Material Integrity Analysis (Brahim Framework)")
    lines.append("  " + "=" * 70)
    lines.append("")
    lines.append("  Material      Supply   Environ   Social   Recycle   Overall   Rating")
    lines.append("  " + "-" * 70)

    for mat_name in MATERIALS_DB.keys():
        score = calc.calculate_material_integrity(mat_name)

        overall_bar = "\u2588" * int(score.overall * 10) + "\u2591" * (10 - int(score.overall * 10))

        lines.append(
            f"  {mat_name:12s}  "
            f"{(1-score.supply_risk)*100:5.0f}%   "
            f"{(1-score.environmental)*100:5.0f}%   "
            f"{(1-score.social_risk)*100:5.0f}%   "
            f"{score.recyclability*100:5.0f}%    "
            f"{overall_bar}  {score.rating.value[:6]}"
        )

    lines.append("")
    lines.append("  Higher % = Better | \u2588\u2588\u2588 = Integrity Score")
    lines.append("")
    lines.append("  ANTI-OPPRESSION PRINCIPLE:")
    lines.append("  Materials with high supply concentration (monopoly) score lower.")
    lines.append("  Diversified supply chains protect against price manipulation.")
    lines.append("")

    return "\n".join(lines)


def visualize_chemistry_deep_dive(code: str) -> str:
    """Deep dive into a specific chemistry."""
    calc = BrahimBatteryCalculator()
    chem = BATTERY_CHEMISTRIES.get(code)

    if not chem:
        return f"Unknown chemistry: {code}"

    integrity = calc.calculate_chemistry_integrity(code)
    resonance = calc.calculate_grid_resonance(code)

    lines = []
    lines.append("")
    lines.append(f"  {chem.name} ({chem.code})")
    lines.append("  " + "=" * 60)
    lines.append("")

    lines.append("  PERFORMANCE:")
    lines.append(f"    Energy density:    {chem.energy_density_wh_kg} Wh/kg")
    lines.append(f"    Power density:     {chem.power_density_w_kg} W/kg")
    lines.append(f"    Cycle life:        {chem.cycle_life:,} cycles")
    lines.append(f"    Efficiency:        {chem.round_trip_efficiency*100:.0f}%")
    lines.append(f"    Calendar life:     {chem.calendar_life_years} years")
    lines.append("")

    lines.append("  ECONOMICS:")
    lines.append(f"    Cost per kWh:      ${chem.cost_per_kwh}")
    lines.append(f"    Cost trend:        {chem.cost_trend*100:+.0f}%/year")
    lines.append(f"    Levelized cost:    ${chem.cost_per_kwh / chem.cycle_life * 1000:.2f}/MWh")
    lines.append("")

    lines.append("  BRAHIM SCORES:")
    bar_full = "\u2588"
    bar_empty = "\u2591"
    int_bar = bar_full * int(integrity * 10) + bar_empty * (10 - int(integrity * 10))
    res_bar = bar_full * int(resonance * 10) + bar_empty * (10 - int(resonance * 10))
    lines.append(f"    Integrity:         {int_bar} ({integrity:.2f})")
    lines.append(f"    Grid resonance:    {res_bar} ({resonance:.2f})")
    lines.append(f"    Response time:     {chem.response_time_ms} ms")
    lines.append("")

    lines.append("  MATERIALS:")
    for mat in chem.primary_materials:
        mat_int = calc.calculate_material_integrity(mat)
        critical = " [CRITICAL]" if mat in chem.critical_materials else ""
        lines.append(f"    {mat:12s}: {mat_int.rating.value}{critical}")
    lines.append("")

    lines.append("  BEST FOR:")
    for app in chem.best_applications:
        lines.append(f"    - {app.value.title()}")
    lines.append("")

    return "\n".join(lines)
