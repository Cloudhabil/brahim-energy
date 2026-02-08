"""
European Grid Stress Formula
=============================

Brahim's Grid Stress Formula applied to European electricity grids.
Self-contained module for calculating optimal load shifting windows
across any EU member state grid.

FORMULA:
    G(t) = Sigma(1/(capacity - demand)^2) * exp(-lambda * renewable_fraction)

WHERE:
    G(t)               = Grid stress at time t (dimensionless, 0 to infinity)
    capacity           = Total available generation (MW)
    demand             = Current consumption (MW)
    renewable_fraction = Renewable generation / total demand
    lambda             = GENESIS_CONSTANT (0.0022)

INTERPRETATION:
    G < 0.0001  : Excellent - Heavy load shifting recommended
    G < 0.001   : Good - Normal operations, load shifting beneficial
    G < 0.01    : Moderate - Monitor closely
    G < 0.1     : High - Defer non-critical loads
    G >= 0.1    : Critical - Emergency demand response

DERIVED FROM:
    Traffic Congestion Formula: C(t) = Sigma(1/(capacity - flow)^2) * exp(-lambda*t)
    Brahim's Theorem adaptation for electrical grid load balancing

AUTHOR: GPIA Cognitive Ecosystem
DATE: 2026-01-26
VERSION: 1.0.0
LICENSE: MIT
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from brahim_energy.constants import BETA_SECURITY, GENESIS_CONSTANT

# Fibonacci-based intervals (distinct from the main BRAHIM_SEQUENCE)
FIBONACCI_INTERVALS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


# =============================================================================
# STRESS LEVEL CLASSIFICATION
# =============================================================================

class StressLevel(Enum):
    """Grid stress level classification."""
    EXCELLENT = "excellent"    # G < 0.0001 - Heavy load shifting
    GOOD = "good"              # G < 0.001  - Normal, shifting beneficial
    MODERATE = "moderate"      # G < 0.01   - Monitor closely
    HIGH = "high"              # G < 0.1    - Defer non-critical
    CRITICAL = "critical"      # G >= 0.1   - Emergency response


def classify_stress(stress: float) -> StressLevel:
    """Classify stress value into levels."""
    if stress < 0.0001:
        return StressLevel.EXCELLENT
    elif stress < 0.001:
        return StressLevel.GOOD
    elif stress < 0.01:
        return StressLevel.MODERATE
    elif stress < 0.1:
        return StressLevel.HIGH
    else:
        return StressLevel.CRITICAL


# =============================================================================
# EU COUNTRY CONFIGURATIONS
# =============================================================================

@dataclass
class CountryGridConfig:
    """Configuration for a country's electricity grid."""
    name: str
    code: str  # ISO 3166-1 alpha-2
    installed_capacity_gw: float
    peak_demand_gw: float
    wind_capacity_gw: float
    solar_capacity_gw: float
    nuclear_capacity_gw: float
    hydro_capacity_gw: float
    fossil_capacity_gw: float
    renewable_target_2030: float  # Percentage
    co2_per_capita_tons: float
    population_millions: float
    dominant_source: str  # "solar", "wind", "nuclear", "hydro", "fossil"


# European Union country grid configurations (2024-2025 data)
EU_GRID_CONFIGS: Dict[str, CountryGridConfig] = {
    "DE": CountryGridConfig(
        name="Germany",
        code="DE",
        installed_capacity_gw=240.0,
        peak_demand_gw=82.0,
        wind_capacity_gw=65.0,
        solar_capacity_gw=60.0,
        nuclear_capacity_gw=0.0,  # Phased out 2023
        hydro_capacity_gw=5.5,
        fossil_capacity_gw=80.0,
        renewable_target_2030=0.80,
        co2_per_capita_tons=8.1,
        population_millions=83.2,
        dominant_source="wind",
    ),
    "ES": CountryGridConfig(
        name="Spain",
        code="ES",
        installed_capacity_gw=120.0,
        peak_demand_gw=42.0,
        wind_capacity_gw=30.0,
        solar_capacity_gw=25.0,
        nuclear_capacity_gw=7.0,
        hydro_capacity_gw=17.0,
        fossil_capacity_gw=35.0,
        renewable_target_2030=0.74,
        co2_per_capita_tons=5.2,
        population_millions=47.4,
        dominant_source="solar",
    ),
    "FR": CountryGridConfig(
        name="France",
        code="FR",
        installed_capacity_gw=145.0,
        peak_demand_gw=88.0,
        wind_capacity_gw=22.0,
        solar_capacity_gw=18.0,
        nuclear_capacity_gw=61.0,
        hydro_capacity_gw=25.0,
        fossil_capacity_gw=12.0,
        renewable_target_2030=0.40,
        co2_per_capita_tons=4.5,
        population_millions=67.8,
        dominant_source="nuclear",
    ),
    "IT": CountryGridConfig(
        name="Italy",
        code="IT",
        installed_capacity_gw=125.0,
        peak_demand_gw=56.0,
        wind_capacity_gw=12.0,
        solar_capacity_gw=28.0,
        nuclear_capacity_gw=0.0,
        hydro_capacity_gw=19.0,
        fossil_capacity_gw=58.0,
        renewable_target_2030=0.55,
        co2_per_capita_tons=5.3,
        population_millions=59.0,
        dominant_source="solar",
    ),
    "PL": CountryGridConfig(
        name="Poland",
        code="PL",
        installed_capacity_gw=55.0,
        peak_demand_gw=28.0,
        wind_capacity_gw=9.0,
        solar_capacity_gw=16.0,
        nuclear_capacity_gw=0.0,
        hydro_capacity_gw=2.4,
        fossil_capacity_gw=33.0,  # High coal dependency
        renewable_target_2030=0.32,
        co2_per_capita_tons=8.5,
        population_millions=37.7,
        dominant_source="fossil",
    ),
    "NL": CountryGridConfig(
        name="Netherlands",
        code="NL",
        installed_capacity_gw=45.0,
        peak_demand_gw=18.0,
        wind_capacity_gw=10.0,
        solar_capacity_gw=22.0,
        nuclear_capacity_gw=0.5,
        hydro_capacity_gw=0.0,
        fossil_capacity_gw=18.0,
        renewable_target_2030=0.70,
        co2_per_capita_tons=8.0,
        population_millions=17.5,
        dominant_source="wind",
    ),
    "SE": CountryGridConfig(
        name="Sweden",
        code="SE",
        installed_capacity_gw=42.0,
        peak_demand_gw=26.0,
        wind_capacity_gw=14.0,
        solar_capacity_gw=2.5,
        nuclear_capacity_gw=6.9,
        hydro_capacity_gw=16.5,
        fossil_capacity_gw=1.5,
        renewable_target_2030=0.65,
        co2_per_capita_tons=3.5,
        population_millions=10.4,
        dominant_source="hydro",
    ),
    "AT": CountryGridConfig(
        name="Austria",
        code="AT",
        installed_capacity_gw=28.0,
        peak_demand_gw=12.0,
        wind_capacity_gw=3.5,
        solar_capacity_gw=5.0,
        nuclear_capacity_gw=0.0,
        hydro_capacity_gw=14.5,
        fossil_capacity_gw=5.0,
        renewable_target_2030=0.80,
        co2_per_capita_tons=6.8,
        population_millions=9.0,
        dominant_source="hydro",
    ),
    "BE": CountryGridConfig(
        name="Belgium",
        code="BE",
        installed_capacity_gw=25.0,
        peak_demand_gw=13.5,
        wind_capacity_gw=5.5,
        solar_capacity_gw=8.0,
        nuclear_capacity_gw=4.0,
        hydro_capacity_gw=1.4,
        fossil_capacity_gw=8.0,
        renewable_target_2030=0.40,
        co2_per_capita_tons=8.2,
        population_millions=11.6,
        dominant_source="nuclear",
    ),
    "PT": CountryGridConfig(
        name="Portugal",
        code="PT",
        installed_capacity_gw=22.0,
        peak_demand_gw=8.5,
        wind_capacity_gw=5.6,
        solar_capacity_gw=3.0,
        nuclear_capacity_gw=0.0,
        hydro_capacity_gw=7.2,
        fossil_capacity_gw=4.5,
        renewable_target_2030=0.80,
        co2_per_capita_tons=4.0,
        population_millions=10.3,
        dominant_source="hydro",
    ),
    "DK": CountryGridConfig(
        name="Denmark",
        code="DK",
        installed_capacity_gw=18.0,
        peak_demand_gw=6.5,
        wind_capacity_gw=7.5,
        solar_capacity_gw=3.5,
        nuclear_capacity_gw=0.0,
        hydro_capacity_gw=0.0,
        fossil_capacity_gw=4.0,
        renewable_target_2030=1.00,  # 100% target
        co2_per_capita_tons=4.5,
        population_millions=5.9,
        dominant_source="wind",
    ),
    "GR": CountryGridConfig(
        name="Greece",
        code="GR",
        installed_capacity_gw=22.0,
        peak_demand_gw=10.0,
        wind_capacity_gw=5.0,
        solar_capacity_gw=7.0,
        nuclear_capacity_gw=0.0,
        hydro_capacity_gw=3.4,
        fossil_capacity_gw=8.0,
        renewable_target_2030=0.60,
        co2_per_capita_tons=5.5,
        population_millions=10.4,
        dominant_source="solar",
    ),
}


# =============================================================================
# GRID STRESS CALCULATOR
# =============================================================================

@dataclass
class GridState:
    """Current state of the electricity grid."""
    timestamp: datetime
    capacity_mw: float
    demand_mw: float
    wind_mw: float
    solar_mw: float
    hydro_mw: float
    nuclear_mw: float
    fossil_mw: float
    imports_mw: float = 0.0
    exports_mw: float = 0.0

    @property
    def total_generation_mw(self) -> float:
        return (
            self.wind_mw + self.solar_mw + self.hydro_mw +
            self.nuclear_mw + self.fossil_mw + self.imports_mw
        )

    @property
    def renewable_mw(self) -> float:
        return self.wind_mw + self.solar_mw + self.hydro_mw

    @property
    def renewable_fraction(self) -> float:
        if self.demand_mw <= 0:
            return 0.0
        return min(1.0, self.renewable_mw / self.demand_mw)

    @property
    def margin_mw(self) -> float:
        return self.capacity_mw - self.demand_mw

    @property
    def margin_percentage(self) -> float:
        if self.capacity_mw <= 0:
            return 0.0
        return (self.margin_mw / self.capacity_mw) * 100


@dataclass
class StressResult:
    """Result of grid stress calculation."""
    stress: float
    level: StressLevel
    margin_mw: float
    margin_percentage: float
    renewable_fraction: float
    co2_intensity_kg_kwh: float
    recommended_action: str
    can_add_load_mw: float
    optimal_shift_hours: int

    def to_dict(self) -> dict:
        return {
            "stress": round(self.stress, 8),
            "level": self.level.value,
            "margin_mw": round(self.margin_mw, 1),
            "margin_percentage": round(self.margin_percentage, 2),
            "renewable_fraction": round(self.renewable_fraction, 4),
            "co2_intensity_kg_kwh": round(self.co2_intensity_kg_kwh, 4),
            "recommended_action": self.recommended_action,
            "can_add_load_mw": round(self.can_add_load_mw, 1),
            "optimal_shift_hours": self.optimal_shift_hours,
        }


class BrahimGridStressCalculator:
    """
    Brahim's Grid Stress Calculator for European Electricity Grids.

    Implements the formula:
        G(t) = Sigma(1/(capacity - demand)^2) * exp(-lambda * renewable_fraction)

    Features:
    - Country-specific configurations
    - CO2 intensity calculation
    - Load shifting recommendations
    - 24-hour forecast capability
    """

    def __init__(self, country_code: str = "DE"):
        """
        Initialize calculator for a specific country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code
        """
        if country_code not in EU_GRID_CONFIGS:
            raise ValueError(
                f"Country '{country_code}' not supported. "
                f"Available: {list(EU_GRID_CONFIGS.keys())}"
            )

        self.country_code = country_code
        self.config = EU_GRID_CONFIGS[country_code]

    def calculate_stress(
        self,
        capacity_mw: float,
        demand_mw: float,
        renewable_fraction: float = 0.0,
    ) -> float:
        """
        Calculate grid stress using Brahim's formula.

        G(t) = Sigma(1/(capacity - demand)^2) * exp(-lambda * renewable_fraction)

        Args:
            capacity_mw: Total available generation capacity (MW)
            demand_mw: Current electricity demand (MW)
            renewable_fraction: Fraction of demand met by renewables (0-1)

        Returns:
            Grid stress value (dimensionless, 0 to infinity)
        """
        # Safety check: avoid division by zero
        if capacity_mw <= demand_mw:
            return float('inf')

        margin = capacity_mw - demand_mw

        # Core Brahim formula
        base_stress = 1.0 / (margin ** 2)

        # Renewable discount factor
        # Higher renewable fraction = lower stress
        renewable_discount = math.exp(-GENESIS_CONSTANT * renewable_fraction * 1000)

        stress = base_stress * renewable_discount

        return stress

    def calculate_from_state(self, state: GridState) -> StressResult:
        """
        Calculate comprehensive stress result from grid state.

        Args:
            state: Current grid state

        Returns:
            Detailed stress result with recommendations
        """
        # Calculate base stress
        stress = self.calculate_stress(
            capacity_mw=state.capacity_mw,
            demand_mw=state.demand_mw,
            renewable_fraction=state.renewable_fraction,
        )

        level = classify_stress(stress)

        # Calculate CO2 intensity
        co2 = self._calculate_co2_intensity(state)

        # Determine recommended action
        action = self._get_recommended_action(level, state)

        # Calculate how much load can be added
        can_add = self._calculate_available_headroom(state)

        # Calculate optimal hours to shift
        shift_hours = self._calculate_shift_hours(level)

        return StressResult(
            stress=stress,
            level=level,
            margin_mw=state.margin_mw,
            margin_percentage=state.margin_percentage,
            renewable_fraction=state.renewable_fraction,
            co2_intensity_kg_kwh=co2,
            recommended_action=action,
            can_add_load_mw=can_add,
            optimal_shift_hours=shift_hours,
        )

    def _calculate_co2_intensity(self, state: GridState) -> float:
        """Calculate CO2 intensity based on generation mix."""
        # CO2 factors (kg/MWh)
        co2_factors = {
            "wind": 0,
            "solar": 0,
            "hydro": 0,
            "nuclear": 12,
            "fossil": 450,  # Average of gas (400) and coal (900)
        }

        if state.total_generation_mw <= 0:
            return 0.40  # Default

        total_co2 = (
            state.wind_mw * co2_factors["wind"] +
            state.solar_mw * co2_factors["solar"] +
            state.hydro_mw * co2_factors["hydro"] +
            state.nuclear_mw * co2_factors["nuclear"] +
            state.fossil_mw * co2_factors["fossil"]
        )

        # Convert from kg/MWh to kg/kWh
        co2_per_mwh = total_co2 / state.total_generation_mw
        co2_per_kwh = co2_per_mwh / 1000

        # Apply Brahim correction for grid losses (~5%)
        co2_per_kwh *= (1 + GENESIS_CONSTANT * 25)

        return co2_per_kwh

    def _get_recommended_action(self, level: StressLevel, state: GridState) -> str:
        """Get recommended action based on stress level."""
        actions = {
            StressLevel.EXCELLENT: (
                f"SHIFT NOW: Add heavy loads. "
                f"{state.renewable_fraction*100:.0f}% renewable, "
                f"{state.margin_mw:.0f} MW headroom."
            ),
            StressLevel.GOOD: (
                f"Good window for EV charging and industrial loads. "
                f"{state.margin_mw:.0f} MW available."
            ),
            StressLevel.MODERATE: (
                f"Monitor grid. Only add essential loads. "
                f"Margin: {state.margin_percentage:.1f}%"
            ),
            StressLevel.HIGH: (
                f"DEFER non-critical loads. "
                f"Wait {self._calculate_shift_hours(level)} hours for better window."
            ),
            StressLevel.CRITICAL: (
                f"EMERGENCY: Reduce consumption immediately. "
                f"Grid at {100-state.margin_percentage:.1f}% capacity."
            ),
        }
        return actions.get(level, "Unknown state")

    def _calculate_available_headroom(self, state: GridState) -> float:
        """Calculate how much additional load can be safely added."""
        # Safe margin: BETA_SECURITY of capacity
        safe_margin = state.capacity_mw * BETA_SECURITY
        available = state.margin_mw - safe_margin

        return max(0, available)

    def _calculate_shift_hours(self, level: StressLevel) -> int:
        """Calculate optimal hours to wait before using power."""
        shift_map = {
            StressLevel.EXCELLENT: 0,
            StressLevel.GOOD: 0,
            StressLevel.MODERATE: 2,
            StressLevel.HIGH: 4,
            StressLevel.CRITICAL: 8,
        }
        return shift_map.get(level, 0)

    def simulate_24h(
        self,
        base_demand_mw: Optional[float] = None,
    ) -> List[Tuple[datetime, StressResult]]:
        """
        Simulate grid stress for 24 hours.

        Args:
            base_demand_mw: Base demand (default: country peak * 0.7)

        Returns:
            List of (datetime, StressResult) tuples
        """
        if base_demand_mw is None:
            base_demand_mw = self.config.peak_demand_gw * 1000 * 0.7

        results = []
        now = datetime.now().replace(minute=0, second=0, microsecond=0)

        for hour_offset in range(24):
            check_time = now + timedelta(hours=hour_offset)
            hour = check_time.hour

            # Simulate demand pattern
            demand_factor = self._get_demand_factor(hour)
            demand = base_demand_mw * demand_factor

            # Simulate renewable output
            wind, solar = self._simulate_renewables(hour)

            # Build grid state
            state = GridState(
                timestamp=check_time,
                capacity_mw=self.config.installed_capacity_gw * 1000,
                demand_mw=demand,
                wind_mw=wind,
                solar_mw=solar,
                hydro_mw=self.config.hydro_capacity_gw * 1000 * 0.4,
                nuclear_mw=self.config.nuclear_capacity_gw * 1000 * 0.9,
                fossil_mw=max(0, demand - wind - solar -
                             self.config.hydro_capacity_gw * 1000 * 0.4 -
                             self.config.nuclear_capacity_gw * 1000 * 0.9),
            )

            result = self.calculate_from_state(state)
            results.append((check_time, result))

        return results

    def _get_demand_factor(self, hour: int) -> float:
        """Get demand factor by hour (0-23)."""
        # Typical European demand curve
        factors = {
            0: 0.65, 1: 0.60, 2: 0.58, 3: 0.57, 4: 0.58, 5: 0.62,
            6: 0.72, 7: 0.85, 8: 0.92, 9: 0.95, 10: 0.97, 11: 0.98,
            12: 0.96, 13: 0.94, 14: 0.92, 15: 0.93, 16: 0.95, 17: 0.98,
            18: 1.00, 19: 0.98, 20: 0.92, 21: 0.85, 22: 0.78, 23: 0.70,
        }
        return factors.get(hour, 0.85)

    def _simulate_renewables(self, hour: int) -> Tuple[float, float]:
        """Simulate wind and solar output by hour."""
        # Wind: more at night, varies by country
        wind_base = self.config.wind_capacity_gw * 1000
        wind_hour_factor = 0.8 + 0.4 * math.sin(math.pi * (hour + 6) / 12)
        wind = wind_base * wind_hour_factor * 0.35  # Capacity factor

        # Solar: obvious day/night cycle
        solar_base = self.config.solar_capacity_gw * 1000
        if 6 <= hour <= 20:
            solar_factor = math.sin(math.pi * (hour - 6) / 14)
            solar = solar_base * solar_factor * 0.7  # Peak capacity factor
        else:
            solar = 0

        return (wind, solar)

    def find_optimal_windows(
        self,
        duration_hours: float = 2.0,
        max_stress: float = 0.001,
    ) -> List[Tuple[datetime, datetime, float]]:
        """
        Find optimal windows for load shifting.

        Args:
            duration_hours: Required duration
            max_stress: Maximum acceptable stress

        Returns:
            List of (start, end, avg_stress) tuples
        """
        simulation = self.simulate_24h()
        windows = []

        i = 0
        while i < len(simulation) - int(duration_hours):
            # Check if this window is below max stress
            window_results = simulation[i:i + int(duration_hours) + 1]
            avg_stress = sum(r[1].stress for r in window_results) / len(window_results)

            if avg_stress <= max_stress:
                start = window_results[0][0]
                end = window_results[-1][0]
                windows.append((start, end, avg_stress))
                i += int(duration_hours)  # Skip to avoid overlapping
            else:
                i += 1

        # Sort by stress (best first)
        windows.sort(key=lambda w: w[2])

        return windows

    def calculate_co2_savings(
        self,
        load_kwh: float,
        from_stress: StressResult,
        to_stress: StressResult,
    ) -> float:
        """
        Calculate CO2 savings from load shifting.

        Args:
            load_kwh: Energy to shift (kWh)
            from_stress: Original time slot
            to_stress: New time slot

        Returns:
            CO2 saved (kg)
        """
        co2_before = load_kwh * from_stress.co2_intensity_kg_kwh
        co2_after = load_kwh * to_stress.co2_intensity_kg_kwh

        return co2_before - co2_after


# =============================================================================
# VISUALIZATION (ASCII)
# =============================================================================

def visualize_24h(calculator: BrahimGridStressCalculator) -> str:
    """Generate ASCII visualization of 24h stress."""
    simulation = calculator.simulate_24h()

    lines = []
    lines.append("")
    lines.append(f"  24-Hour Grid Stress Forecast: {calculator.config.name}")
    lines.append("  " + "=" * 68)
    lines.append("")
    lines.append("  Hour  Stress     Level      Renewable  CO2       Action")
    lines.append("  " + "-" * 68)

    for time, result in simulation:
        hour = time.strftime("%H:00")

        # Stress bar
        if result.stress < 0.0001:
            bar = "█" * 1
        elif result.stress < 0.001:
            bar = "█" * 2
        elif result.stress < 0.01:
            bar = "█" * 4
        elif result.stress < 0.1:
            bar = "█" * 6
        else:
            bar = "█" * 8

        level_short = {
            StressLevel.EXCELLENT: "EXCEL",
            StressLevel.GOOD: "GOOD ",
            StressLevel.MODERATE: "MOD  ",
            StressLevel.HIGH: "HIGH ",
            StressLevel.CRITICAL: "CRIT ",
        }.get(result.level, "?????")

        renewable_pct = f"{result.renewable_fraction * 100:5.1f}%"
        co2 = f"{result.co2_intensity_kg_kwh:.3f}"

        # Short action
        if result.level == StressLevel.EXCELLENT:
            action = ">>> SHIFT NOW <<<"
        elif result.level == StressLevel.GOOD:
            action = "Good for EVs"
        elif result.level == StressLevel.MODERATE:
            action = "Monitor"
        elif result.level == StressLevel.HIGH:
            action = "DEFER"
        else:
            action = "!!! EMERGENCY !!!"

        lines.append(
            f"  {hour}  {bar:8s}  {level_short}  {renewable_pct}   {co2}    {action}"
        )

    lines.append("")
    lines.append("  Legend: █ = Stress level (fewer = better)")
    lines.append("")

    return "\n".join(lines)


def visualize_eu_comparison() -> str:
    """Compare grid stress across EU countries."""
    lines = []
    lines.append("")
    lines.append("  European Grid Stress Comparison (Current Hour)")
    lines.append("  " + "=" * 70)
    lines.append("")
    lines.append("  Country        Stress     Level      Renewable  Dominant Source")
    lines.append("  " + "-" * 70)

    results = []

    for code, config in EU_GRID_CONFIGS.items():
        calc = BrahimGridStressCalculator(code)
        simulation = calc.simulate_24h()

        # Get current hour
        current_hour = datetime.now().hour
        _, result = simulation[current_hour]

        results.append((config.name, result, config.dominant_source))

    # Sort by stress (best first)
    results.sort(key=lambda r: r[1].stress)

    for name, result, dominant in results:
        if result.stress < 0.0001:
            bar = "▓" * 2
        elif result.stress < 0.001:
            bar = "▓" * 4
        elif result.stress < 0.01:
            bar = "▓" * 6
        else:
            bar = "▓" * 8

        level = result.level.value[:6].upper()
        renewable = f"{result.renewable_fraction * 100:5.1f}%"

        lines.append(f"  {name:12s}   {bar:8s}  {level:6s}   {renewable}    {dominant}")

    lines.append("")

    return "\n".join(lines)
