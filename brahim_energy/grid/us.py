"""
US Grid Stress Formula
======================

Brahim's Grid Stress Formula applied to United States electricity grids.
Covers all major ISOs/RTOs across the three interconnections.

FORMULA:
    G(t) = Sigma(1/(capacity - demand)^2) x exp(-lambda x renewable_fraction)

US GRID STRUCTURE:
    +-------------------------------------------------------------+
    |                    WESTERN INTERCONNECTION                    |
    |  +---------+  +---------+  +---------+  +---------+          |
    |  |  CAISO  |  |   SPP   |  |  WAPA   |  |   BPA   |          |
    |  | (Calif) |  | (Plains)|  | (Hydro) |  | (PNW)   |          |
    |  +---------+  +---------+  +---------+  +---------+          |
    +-------------------------------------------------------------+
    +-------------------------------------------------------------+
    |                    EASTERN INTERCONNECTION                    |
    |  +---------+  +---------+  +---------+  +---------+          |
    |  |   PJM   |  |  MISO   |  |  NYISO  |  | ISO-NE  |          |
    |  |(13 states)| |(Midwest)|  | (NY)    |  |(New Eng)|          |
    |  +---------+  +---------+  +---------+  +---------+          |
    +-------------------------------------------------------------+
    +-------------------------------------------------------------+
    |                    TEXAS INTERCONNECTION                      |
    |                      +-------------+                         |
    |                      |   ERCOT     |  (Isolated - no imports)|
    |                      |  (Texas)    |                         |
    |                      +-------------+                         |
    +-------------------------------------------------------------+

SPECIAL FEATURES:
    - Duck Curve detection (CAISO solar surplus)
    - Polar Vortex stress alerts (ERCOT, MISO, PJM)
    - Heat Dome detection (Western US)
    - Wind Corridor optimization (SPP, ERCOT, MISO)

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

from brahim_energy.constants import BETA_SECURITY, GENESIS_CONSTANT, PHI

# =============================================================================
# LOCAL FIBONACCI SEQUENCE (distinct from BRAHIM_SEQUENCE in constants)
# =============================================================================

FIBONACCI_INTERVALS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


# =============================================================================
# STRESS LEVEL CLASSIFICATION
# =============================================================================

class StressLevel(Enum):
    """Grid stress level classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"  # US-specific: rolling blackout risk


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
    elif stress < 1.0:
        return StressLevel.CRITICAL
    else:
        return StressLevel.EMERGENCY


# =============================================================================
# US INTERCONNECTIONS AND ISOs
# =============================================================================

class Interconnection(Enum):
    """US Grid Interconnections."""
    WESTERN = "western"
    EASTERN = "eastern"
    TEXAS = "texas"


class ISORegion(Enum):
    """US Independent System Operators / Regional Transmission Organizations."""
    # Western Interconnection
    CAISO = "caiso"      # California ISO
    SPP = "spp"          # Southwest Power Pool (also Eastern)
    BPA = "bpa"          # Bonneville Power Administration
    WAPA = "wapa"        # Western Area Power Administration

    # Eastern Interconnection
    PJM = "pjm"          # PJM Interconnection (13 states + DC)
    MISO = "miso"        # Midcontinent ISO
    NYISO = "nyiso"      # New York ISO
    ISONE = "isone"      # ISO New England
    TVA = "tva"          # Tennessee Valley Authority
    SOCO = "soco"        # Southern Company

    # Texas Interconnection
    ERCOT = "ercot"      # Electric Reliability Council of Texas


# =============================================================================
# ISO CONFIGURATIONS
# =============================================================================

@dataclass
class ISOConfig:
    """Configuration for an ISO/RTO region."""
    name: str
    code: str
    interconnection: Interconnection
    installed_capacity_gw: float
    peak_demand_gw: float
    wind_capacity_gw: float
    solar_capacity_gw: float
    nuclear_capacity_gw: float
    hydro_capacity_gw: float
    gas_capacity_gw: float
    coal_capacity_gw: float
    population_millions: float
    states: List[str]
    api_url: str
    dominant_source: str
    has_duck_curve: bool = False
    polar_vortex_risk: bool = False
    heat_dome_risk: bool = False


US_ISO_CONFIGS: Dict[ISORegion, ISOConfig] = {
    ISORegion.CAISO: ISOConfig(
        name="California ISO",
        code="CAISO",
        interconnection=Interconnection.WESTERN,
        installed_capacity_gw=85.0,
        peak_demand_gw=52.0,
        wind_capacity_gw=6.5,
        solar_capacity_gw=20.0,  # Largest solar in US
        nuclear_capacity_gw=2.2,  # Diablo Canyon
        hydro_capacity_gw=14.0,
        gas_capacity_gw=38.0,
        coal_capacity_gw=0.0,  # Phased out
        population_millions=39.5,
        states=["CA"],
        api_url="https://api.caiso.com/oasis/",
        dominant_source="solar",
        has_duck_curve=True,
        heat_dome_risk=True,
    ),
    ISORegion.ERCOT: ISOConfig(
        name="ERCOT (Texas)",
        code="ERCOT",
        interconnection=Interconnection.TEXAS,
        installed_capacity_gw=140.0,
        peak_demand_gw=85.0,
        wind_capacity_gw=38.0,  # #1 in US
        solar_capacity_gw=18.0,
        nuclear_capacity_gw=5.1,
        hydro_capacity_gw=0.7,
        gas_capacity_gw=55.0,
        coal_capacity_gw=14.0,
        population_millions=30.0,
        states=["TX"],
        api_url="https://www.ercot.com/api/",
        dominant_source="wind",
        polar_vortex_risk=True,  # Feb 2021 crisis
        heat_dome_risk=True,
    ),
    ISORegion.PJM: ISOConfig(
        name="PJM Interconnection",
        code="PJM",
        interconnection=Interconnection.EASTERN,
        installed_capacity_gw=185.0,
        peak_demand_gw=150.0,
        wind_capacity_gw=4.5,
        solar_capacity_gw=5.0,
        nuclear_capacity_gw=33.0,  # Largest nuclear fleet
        hydro_capacity_gw=8.0,
        gas_capacity_gw=75.0,
        coal_capacity_gw=45.0,
        population_millions=65.0,
        states=["DE", "IL", "IN", "KY", "MD", "MI", "NJ", "NC", "OH", "PA", "TN", "VA", "WV", "DC"],
        api_url="https://api.pjm.com/api/v1/",
        dominant_source="nuclear",
        polar_vortex_risk=True,
    ),
    ISORegion.MISO: ISOConfig(
        name="Midcontinent ISO",
        code="MISO",
        interconnection=Interconnection.EASTERN,
        installed_capacity_gw=195.0,
        peak_demand_gw=127.0,
        wind_capacity_gw=32.0,  # Wind corridor
        solar_capacity_gw=4.0,
        nuclear_capacity_gw=12.0,
        hydro_capacity_gw=1.5,
        gas_capacity_gw=70.0,
        coal_capacity_gw=50.0,
        population_millions=45.0,
        states=["AR", "IL", "IN", "IA", "KY", "LA", "MI", "MN", "MS", "MO", "MT", "ND", "SD", "TX", "WI"],
        api_url="https://api.misoenergy.org/",
        dominant_source="wind",
        polar_vortex_risk=True,
    ),
    ISORegion.SPP: ISOConfig(
        name="Southwest Power Pool",
        code="SPP",
        interconnection=Interconnection.WESTERN,  # Spans both
        installed_capacity_gw=105.0,
        peak_demand_gw=55.0,
        wind_capacity_gw=35.0,  # Wind corridor heartland
        solar_capacity_gw=8.0,
        nuclear_capacity_gw=2.5,
        hydro_capacity_gw=2.5,
        gas_capacity_gw=40.0,
        coal_capacity_gw=20.0,
        population_millions=18.0,
        states=["AR", "KS", "LA", "MO", "NE", "NM", "OK", "TX"],
        api_url="https://marketplace.spp.org/",
        dominant_source="wind",
    ),
    ISORegion.NYISO: ISOConfig(
        name="New York ISO",
        code="NYISO",
        interconnection=Interconnection.EASTERN,
        installed_capacity_gw=42.0,
        peak_demand_gw=33.0,
        wind_capacity_gw=2.5,
        solar_capacity_gw=4.5,
        nuclear_capacity_gw=5.3,
        hydro_capacity_gw=6.0,  # Niagara Falls
        gas_capacity_gw=22.0,
        coal_capacity_gw=0.0,  # Phased out
        population_millions=19.5,
        states=["NY"],
        api_url="https://www.nyiso.com/public/webservices/",
        dominant_source="hydro",
        polar_vortex_risk=True,
    ),
    ISORegion.ISONE: ISOConfig(
        name="ISO New England",
        code="ISO-NE",
        interconnection=Interconnection.EASTERN,
        installed_capacity_gw=35.0,
        peak_demand_gw=28.0,
        wind_capacity_gw=1.5,
        solar_capacity_gw=6.0,
        nuclear_capacity_gw=3.4,
        hydro_capacity_gw=4.0,
        gas_capacity_gw=18.0,
        coal_capacity_gw=0.5,
        population_millions=14.8,
        states=["CT", "MA", "ME", "NH", "RI", "VT"],
        api_url="https://webservices.iso-ne.com/",
        dominant_source="gas",
        polar_vortex_risk=True,
    ),
    ISORegion.BPA: ISOConfig(
        name="Bonneville Power Administration",
        code="BPA",
        interconnection=Interconnection.WESTERN,
        installed_capacity_gw=35.0,
        peak_demand_gw=12.0,
        wind_capacity_gw=6.0,
        solar_capacity_gw=0.5,
        nuclear_capacity_gw=1.2,
        hydro_capacity_gw=22.0,  # Columbia River dams
        gas_capacity_gw=3.0,
        coal_capacity_gw=1.5,
        population_millions=13.0,
        states=["WA", "OR", "ID", "MT"],
        api_url="https://transmission.bpa.gov/",
        dominant_source="hydro",
    ),
}


# =============================================================================
# EXTREME WEATHER SCENARIOS
# =============================================================================

@dataclass
class ExtremeWeatherEvent:
    """Extreme weather event affecting grid stress."""
    name: str
    description: str
    affected_isos: List[ISORegion]
    demand_multiplier: float  # 1.0 = normal
    renewable_reduction: float  # 0.0 = no reduction, 1.0 = complete loss
    outage_probability: float  # 0-1
    historical_example: str


EXTREME_WEATHER_SCENARIOS: Dict[str, ExtremeWeatherEvent] = {
    "polar_vortex": ExtremeWeatherEvent(
        name="Polar Vortex",
        description="Arctic air mass causing extreme cold and heating demand surge",
        affected_isos=[ISORegion.ERCOT, ISORegion.MISO, ISORegion.PJM, ISORegion.ISONE],
        demand_multiplier=1.45,  # 45% increase
        renewable_reduction=0.60,  # Wind turbines freeze
        outage_probability=0.15,
        historical_example="February 2021 Texas Crisis - 4.5M without power",
    ),
    "heat_dome": ExtremeWeatherEvent(
        name="Heat Dome",
        description="Persistent high pressure causing extreme heat and AC demand",
        affected_isos=[ISORegion.CAISO, ISORegion.ERCOT, ISORegion.SPP],
        demand_multiplier=1.35,
        renewable_reduction=0.10,  # Solar still works, slight efficiency loss
        outage_probability=0.08,
        historical_example="June 2021 Pacific Northwest - 116F in Portland",
    ),
    "hurricane": ExtremeWeatherEvent(
        name="Hurricane",
        description="Major tropical storm causing infrastructure damage",
        affected_isos=[ISORegion.ERCOT, ISORegion.MISO],
        demand_multiplier=0.70,  # Evacuations reduce demand
        renewable_reduction=0.90,  # Wind/solar shut down
        outage_probability=0.40,
        historical_example="Hurricane Harvey 2017 - 300,000+ without power",
    ),
    "wildfire": ExtremeWeatherEvent(
        name="Wildfire PSPS",
        description="Public Safety Power Shutoffs during extreme fire weather",
        affected_isos=[ISORegion.CAISO],
        demand_multiplier=0.85,
        renewable_reduction=0.30,  # Smoke reduces solar
        outage_probability=0.25,
        historical_example="2020 California PSPS - 500,000+ shutoffs",
    ),
    "bomb_cyclone": ExtremeWeatherEvent(
        name="Bomb Cyclone",
        description="Rapidly intensifying storm with high winds",
        affected_isos=[ISORegion.ISONE, ISORegion.NYISO, ISORegion.PJM],
        demand_multiplier=1.25,
        renewable_reduction=0.50,  # Wind too strong, turbines brake
        outage_probability=0.12,
        historical_example="December 2022 Buffalo Blizzard",
    ),
}


# =============================================================================
# DUCK CURVE MODEL (CAISO-SPECIFIC)
# =============================================================================

@dataclass
class DuckCurveStatus:
    """California Duck Curve status."""
    timestamp: datetime
    net_load_mw: float  # Total load - solar generation
    solar_generation_mw: float
    ramp_rate_mw_per_hour: float
    is_belly: bool  # Midday solar surplus
    is_neck: bool   # Evening ramp
    curtailment_mw: float  # Solar being wasted
    negative_prices: bool


def calculate_duck_curve(hour: int, base_load_mw: float, solar_capacity_mw: float) -> DuckCurveStatus:
    """
    Calculate California duck curve position.

    The "duck curve" shows net load (demand minus solar) creating:
    - Morning ramp (5-9 AM): demand rises faster than solar
    - Belly (10 AM - 4 PM): solar surplus, net load drops
    - Neck/Evening ramp (4-9 PM): solar drops, demand peaks
    """
    # Demand pattern (pre-solar era baseline)
    demand_factors = {
        0: 0.60, 1: 0.55, 2: 0.52, 3: 0.50, 4: 0.52, 5: 0.58,
        6: 0.68, 7: 0.80, 8: 0.88, 9: 0.92, 10: 0.95, 11: 0.98,
        12: 1.00, 13: 0.98, 14: 0.96, 15: 0.95, 16: 0.97, 17: 1.02,
        18: 1.05, 19: 1.00, 20: 0.92, 21: 0.82, 22: 0.72, 23: 0.65,
    }

    gross_load = base_load_mw * demand_factors.get(hour, 0.85)

    # Solar generation
    if 6 <= hour <= 19:
        # Parabolic solar curve peaking at 12:30
        solar_factor = max(0, math.sin(math.pi * (hour - 5.5) / 14))
        solar_gen = solar_capacity_mw * solar_factor * 0.85  # Capacity factor
    else:
        solar_gen = 0

    net_load = gross_load - solar_gen

    # Ramp rate (change from previous hour)
    prev_hour = (hour - 1) % 24
    prev_demand = base_load_mw * demand_factors.get(prev_hour, 0.85)
    if 6 <= prev_hour <= 19:
        prev_solar = solar_capacity_mw * max(0, math.sin(math.pi * (prev_hour - 5.5) / 14)) * 0.85
    else:
        prev_solar = 0
    prev_net = prev_demand - prev_solar
    ramp_rate = net_load - prev_net

    # Duck curve positions
    is_belly = 10 <= hour <= 15 and solar_gen > gross_load * 0.3
    is_neck = 17 <= hour <= 20 and ramp_rate > 2000

    # Curtailment (oversupply)
    curtailment = max(0, solar_gen - gross_load * 0.5) if is_belly else 0

    return DuckCurveStatus(
        timestamp=datetime.now().replace(hour=hour, minute=0),
        net_load_mw=net_load,
        solar_generation_mw=solar_gen,
        ramp_rate_mw_per_hour=ramp_rate,
        is_belly=is_belly,
        is_neck=is_neck,
        curtailment_mw=curtailment,
        negative_prices=curtailment > 1000,
    )


# =============================================================================
# GRID STATE AND STRESS CALCULATION
# =============================================================================

@dataclass
class USGridState:
    """Current state of a US grid region."""
    timestamp: datetime
    iso: ISORegion
    capacity_mw: float
    demand_mw: float
    wind_mw: float
    solar_mw: float
    hydro_mw: float
    nuclear_mw: float
    gas_mw: float
    coal_mw: float
    imports_mw: float = 0.0
    exports_mw: float = 0.0
    temperature_f: Optional[float] = None
    is_extreme_weather: bool = False
    weather_event: Optional[str] = None

    @property
    def total_generation_mw(self) -> float:
        return (
            self.wind_mw + self.solar_mw + self.hydro_mw +
            self.nuclear_mw + self.gas_mw + self.coal_mw +
            self.imports_mw - self.exports_mw
        )

    @property
    def renewable_mw(self) -> float:
        return self.wind_mw + self.solar_mw + self.hydro_mw

    @property
    def fossil_mw(self) -> float:
        return self.gas_mw + self.coal_mw

    @property
    def renewable_fraction(self) -> float:
        if self.demand_mw <= 0:
            return 0.0
        return min(1.0, self.renewable_mw / self.demand_mw)

    @property
    def margin_mw(self) -> float:
        return self.capacity_mw - self.demand_mw


@dataclass
class USStressResult:
    """Result of US grid stress calculation."""
    iso: ISORegion
    stress: float
    level: StressLevel
    margin_mw: float
    margin_percentage: float
    renewable_fraction: float
    co2_intensity_lb_mwh: float  # US uses lb/MWh
    price_estimate_usd_mwh: float
    recommended_action: str
    can_add_load_mw: float
    duck_curve: Optional[DuckCurveStatus] = None
    extreme_weather: Optional[ExtremeWeatherEvent] = None

    def to_dict(self) -> dict:
        return {
            "iso": self.iso.value,
            "stress": round(self.stress, 8),
            "level": self.level.value,
            "margin_mw": round(self.margin_mw, 1),
            "margin_percentage": round(self.margin_percentage, 2),
            "renewable_fraction": round(self.renewable_fraction, 4),
            "co2_intensity_lb_mwh": round(self.co2_intensity_lb_mwh, 1),
            "price_estimate_usd_mwh": round(self.price_estimate_usd_mwh, 2),
            "recommended_action": self.recommended_action,
            "can_add_load_mw": round(self.can_add_load_mw, 1),
        }


# =============================================================================
# US GRID STRESS CALCULATOR
# =============================================================================

class USGridStressCalculator:
    """
    Brahim's Grid Stress Calculator for US Electricity Grids.

    Specialized for US ISOs/RTOs with:
    - Duck curve detection (CAISO)
    - Extreme weather stress testing
    - Three interconnection awareness
    - LMP (Locational Marginal Pricing) estimates
    """

    def __init__(self, iso: ISORegion = ISORegion.CAISO):
        """Initialize calculator for a specific ISO."""
        self.iso = iso
        self.config = US_ISO_CONFIGS[iso]

    def calculate_stress(
        self,
        capacity_mw: float,
        demand_mw: float,
        renewable_fraction: float = 0.0,
    ) -> float:
        """
        Calculate grid stress using Brahim's formula.

        G(t) = Sigma(1/(capacity - demand)^2) x exp(-lambda x renewable_fraction)
        """
        if capacity_mw <= demand_mw:
            return float('inf')

        margin = capacity_mw - demand_mw
        base_stress = 1.0 / (margin ** 2)
        renewable_discount = math.exp(-GENESIS_CONSTANT * renewable_fraction * 1000)

        return base_stress * renewable_discount

    def calculate_from_state(self, state: USGridState) -> USStressResult:
        """Calculate comprehensive stress result from grid state."""
        stress = self.calculate_stress(
            capacity_mw=state.capacity_mw,
            demand_mw=state.demand_mw,
            renewable_fraction=state.renewable_fraction,
        )

        level = classify_stress(stress)
        co2 = self._calculate_co2_intensity(state)
        price = self._estimate_lmp(state, stress)
        action = self._get_recommended_action(level, state)
        headroom = self._calculate_headroom(state)

        # Duck curve for CAISO
        duck = None
        if self.iso == ISORegion.CAISO:
            duck = calculate_duck_curve(
                hour=state.timestamp.hour,
                base_load_mw=self.config.peak_demand_gw * 1000 * 0.75,
                solar_capacity_mw=self.config.solar_capacity_gw * 1000,
            )

        # Extreme weather
        extreme = None
        if state.is_extreme_weather and state.weather_event:
            extreme = EXTREME_WEATHER_SCENARIOS.get(state.weather_event)

        return USStressResult(
            iso=self.iso,
            stress=stress,
            level=level,
            margin_mw=state.margin_mw,
            margin_percentage=(state.margin_mw / state.capacity_mw) * 100,
            renewable_fraction=state.renewable_fraction,
            co2_intensity_lb_mwh=co2,
            price_estimate_usd_mwh=price,
            recommended_action=action,
            can_add_load_mw=headroom,
            duck_curve=duck,
            extreme_weather=extreme,
        )

    def _calculate_co2_intensity(self, state: USGridState) -> float:
        """Calculate CO2 intensity in lb/MWh (US standard)."""
        # US EPA CO2 factors (lb/MWh)
        co2_factors = {
            "wind": 0,
            "solar": 0,
            "hydro": 0,
            "nuclear": 0,
            "gas": 898,    # NGCC average
            "coal": 2133,  # Coal average
        }

        if state.total_generation_mw <= 0:
            return 800

        total_co2 = (
            state.wind_mw * co2_factors["wind"] +
            state.solar_mw * co2_factors["solar"] +
            state.hydro_mw * co2_factors["hydro"] +
            state.nuclear_mw * co2_factors["nuclear"] +
            state.gas_mw * co2_factors["gas"] +
            state.coal_mw * co2_factors["coal"]
        )

        return total_co2 / state.total_generation_mw

    def _estimate_lmp(self, state: USGridState, stress: float) -> float:
        """Estimate Locational Marginal Price ($/MWh)."""
        # Base price varies by region
        base_prices = {
            ISORegion.CAISO: 45,
            ISORegion.ERCOT: 35,
            ISORegion.PJM: 40,
            ISORegion.MISO: 32,
            ISORegion.SPP: 28,
            ISORegion.NYISO: 55,
            ISORegion.ISONE: 50,
            ISORegion.BPA: 25,
        }

        base = base_prices.get(self.iso, 40)

        # Stress multiplier
        if stress < 0.0001:
            price = base * 0.5  # Possible negative
        elif stress < 0.001:
            price = base * 0.8
        elif stress < 0.01:
            price = base
        elif stress < 0.1:
            price = base * 1.5
        else:
            price = base * 3.0  # Scarcity pricing

        # Renewable discount
        price *= (1 - state.renewable_fraction * 0.3)

        return max(-10, price)  # Can go negative

    def _get_recommended_action(self, level: StressLevel, state: USGridState) -> str:
        """Get recommended action for stress level."""
        actions = {
            StressLevel.EXCELLENT: (
                f"SHIFT NOW: Perfect for EV charging & data centers. "
                f"{state.renewable_fraction*100:.0f}% clean energy."
            ),
            StressLevel.GOOD: (
                f"Good conditions. {state.margin_mw/1000:.1f} GW headroom available."
            ),
            StressLevel.MODERATE: (
                "Normal operations. Monitor for changes."
            ),
            StressLevel.HIGH: (
                "Conservation advisory. Defer non-essential loads."
            ),
            StressLevel.CRITICAL: (
                "FLEX ALERT: Reduce consumption 4-9 PM. Avoid major appliances."
            ),
            StressLevel.EMERGENCY: (
                "EMERGENCY: Rolling blackouts possible. Minimize ALL usage."
            ),
        }
        return actions.get(level, "Unknown")

    def _calculate_headroom(self, state: USGridState) -> float:
        """Calculate available load headroom."""
        reserve_margin = state.capacity_mw * 0.15  # 15% reserve requirement
        return max(0, state.margin_mw - reserve_margin)

    def simulate_24h(
        self,
        base_demand_mw: Optional[float] = None,
        weather_event: Optional[str] = None,
    ) -> List[Tuple[datetime, USStressResult]]:
        """Simulate 24 hours of grid stress."""
        if base_demand_mw is None:
            base_demand_mw = self.config.peak_demand_gw * 1000 * 0.7

        # Apply extreme weather if specified
        demand_mult = 1.0
        renewable_red = 0.0
        if weather_event and weather_event in EXTREME_WEATHER_SCENARIOS:
            event = EXTREME_WEATHER_SCENARIOS[weather_event]
            if self.iso in event.affected_isos:
                demand_mult = event.demand_multiplier
                renewable_red = event.renewable_reduction

        results = []
        now = datetime.now().replace(minute=0, second=0, microsecond=0)

        for hour_offset in range(24):
            check_time = now + timedelta(hours=hour_offset)
            hour = check_time.hour

            demand_factor = self._get_demand_factor(hour)
            demand = base_demand_mw * demand_factor * demand_mult

            wind, solar = self._simulate_renewables(hour, renewable_red)

            # Calculate fossil needed
            clean_gen = (
                wind + solar +
                self.config.hydro_capacity_gw * 1000 * 0.5 +
                self.config.nuclear_capacity_gw * 1000 * 0.92
            )
            fossil_needed = max(0, demand - clean_gen)

            # Split between gas and coal
            gas_ratio = 0.7 if self.config.coal_capacity_gw < 5 else 0.5
            gas = fossil_needed * gas_ratio
            coal = fossil_needed * (1 - gas_ratio)

            state = USGridState(
                timestamp=check_time,
                iso=self.iso,
                capacity_mw=self.config.installed_capacity_gw * 1000,
                demand_mw=demand,
                wind_mw=wind,
                solar_mw=solar,
                hydro_mw=self.config.hydro_capacity_gw * 1000 * 0.5,
                nuclear_mw=self.config.nuclear_capacity_gw * 1000 * 0.92,
                gas_mw=gas,
                coal_mw=coal,
                is_extreme_weather=weather_event is not None,
                weather_event=weather_event,
            )

            result = self.calculate_from_state(state)
            results.append((check_time, result))

        return results

    def _get_demand_factor(self, hour: int) -> float:
        """Get demand factor by hour (US patterns)."""
        # US has more pronounced dual peaks
        factors = {
            0: 0.62, 1: 0.58, 2: 0.55, 3: 0.53, 4: 0.54, 5: 0.60,
            6: 0.72, 7: 0.85, 8: 0.92, 9: 0.94, 10: 0.93, 11: 0.92,
            12: 0.93, 13: 0.94, 14: 0.96, 15: 0.98, 16: 1.00, 17: 1.02,
            18: 1.00, 19: 0.95, 20: 0.88, 21: 0.80, 22: 0.72, 23: 0.65,
        }
        return factors.get(hour, 0.85)

    def _simulate_renewables(
        self,
        hour: int,
        reduction: float = 0.0,
    ) -> Tuple[float, float]:
        """Simulate wind and solar output."""
        wind_base = self.config.wind_capacity_gw * 1000

        # Wind patterns vary by ISO
        if self.iso in [ISORegion.ERCOT, ISORegion.SPP, ISORegion.MISO]:
            # Wind corridor: stronger at night
            wind_factor = 0.5 + 0.3 * math.cos(math.pi * hour / 12)
        else:
            wind_factor = 0.3 + 0.1 * math.sin(math.pi * hour / 12)

        wind = wind_base * wind_factor * (1 - reduction)

        # Solar
        solar_base = self.config.solar_capacity_gw * 1000
        if 6 <= hour <= 19:
            solar_factor = math.sin(math.pi * (hour - 5.5) / 14)
            solar = solar_base * max(0, solar_factor) * 0.75 * (1 - reduction * 0.3)
        else:
            solar = 0

        return (wind, solar)

    def find_optimal_windows(
        self,
        duration_hours: float = 2.0,
        max_stress: float = 0.001,
    ) -> List[Tuple[datetime, datetime, float, float]]:
        """Find optimal windows (includes price)."""
        simulation = self.simulate_24h()
        windows = []

        i = 0
        while i < len(simulation) - int(duration_hours):
            window_results = simulation[i:i + int(duration_hours) + 1]
            avg_stress = sum(r[1].stress for r in window_results) / len(window_results)
            avg_price = sum(r[1].price_estimate_usd_mwh for r in window_results) / len(window_results)

            if avg_stress <= max_stress:
                start = window_results[0][0]
                end = window_results[-1][0]
                windows.append((start, end, avg_stress, avg_price))
                i += int(duration_hours)
            else:
                i += 1

        windows.sort(key=lambda w: w[2])
        return windows

    def calculate_annual_co2_savings(
        self,
        ev_count: int = 0,
        data_center_mw: float = 0,
        industrial_mw: float = 0,
        residential_homes: int = 0,
    ) -> Dict[str, float]:
        """Calculate annual CO2 savings from load shifting (in tons)."""
        savings = {}

        # EV: 12,000 miles/year, 3 mi/kWh = 4,000 kWh/year per EV
        # Shift from 800 lb/MWh to 200 lb/MWh = 600 lb/MWh saved
        ev_mwh = ev_count * 4.0  # 4 MWh/year per EV
        savings["ev_charging"] = (ev_mwh * 600) / 2000  # Convert lb to tons

        # Data centers: 24/7 but can shift 30% of load
        dc_mwh = data_center_mw * 8760 * 0.30  # 30% flexible
        savings["data_centers"] = (dc_mwh * 400) / 2000

        # Industrial: 50% shiftable
        ind_mwh = industrial_mw * 8760 * 0.50
        savings["industrial"] = (ind_mwh * 500) / 2000

        # Residential: 10,000 kWh/year, 20% shiftable
        res_mwh = residential_homes * 10 * 0.20
        savings["residential"] = (res_mwh * 300) / 2000

        # Brahim efficiency factor
        brahim_factor = 1 + (PHI - 1) * BETA_SECURITY
        for key in savings:
            savings[key] *= brahim_factor

        savings["total"] = sum(savings.values())
        return savings


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_24h(calculator: USGridStressCalculator, weather: Optional[str] = None) -> str:
    """Generate ASCII visualization of 24h stress."""
    simulation = calculator.simulate_24h(weather_event=weather)

    lines = []
    lines.append("")
    title = f"  24-Hour Grid Stress: {calculator.config.name}"
    if weather:
        title += f" [{weather.upper()}]"
    lines.append(title)
    lines.append("  " + "=" * 72)
    lines.append("")
    lines.append("  Hour  Stress     Level      Renew%   CO2 lb   $/MWh   Action")
    lines.append("  " + "-" * 72)

    for time, result in simulation:
        hour = time.strftime("%H:00")

        # Stress bar
        if result.stress < 0.0001:
            bar = "▓" * 1
        elif result.stress < 0.001:
            bar = "▓" * 2
        elif result.stress < 0.01:
            bar = "▓" * 4
        elif result.stress < 0.1:
            bar = "▓" * 6
        elif result.stress < 1.0:
            bar = "▓" * 8
        else:
            bar = "█" * 10

        level_map = {
            StressLevel.EXCELLENT: "EXCEL",
            StressLevel.GOOD: "GOOD ",
            StressLevel.MODERATE: "MOD  ",
            StressLevel.HIGH: "HIGH ",
            StressLevel.CRITICAL: "CRIT ",
            StressLevel.EMERGENCY: "EMERG",
        }
        level = level_map.get(result.level, "?????")

        renew = f"{result.renewable_fraction * 100:5.1f}%"
        co2 = f"{result.co2_intensity_lb_mwh:6.0f}"
        price = f"${result.price_estimate_usd_mwh:5.1f}"

        # Short action
        if result.level == StressLevel.EXCELLENT:
            action = ">>> CHARGE <<<"
        elif result.level == StressLevel.GOOD:
            action = "Good window"
        elif result.level == StressLevel.MODERATE:
            action = "Monitor"
        elif result.level == StressLevel.HIGH:
            action = "Conserve"
        elif result.level == StressLevel.CRITICAL:
            action = "FLEX ALERT"
        else:
            action = "!EMERGENCY!"

        lines.append(
            f"  {hour}  {bar:10s}  {level}  {renew}  {co2}  {price}  {action}"
        )

    # Duck curve note for CAISO
    if calculator.iso == ISORegion.CAISO:
        lines.append("")
        lines.append("  Duck Curve: Belly (10-15), Neck ramp (17-20)")

    lines.append("")
    return "\n".join(lines)


def visualize_duck_curve() -> str:
    """Visualize California duck curve."""
    lines = []
    lines.append("")
    lines.append("  California Duck Curve - Net Load Profile")
    lines.append("  " + "=" * 60)
    lines.append("")
    lines.append("  Load (MW)")
    lines.append("     ↑")

    base_load = 35000  # MW
    solar_cap = 20000  # MW

    for hour in range(24):
        duck = calculate_duck_curve(hour, base_load, solar_cap)
        net = duck.net_load_mw

        # Scale to 40 chars
        bar_len = int((net / 40000) * 40)
        bar = "█" * max(1, bar_len)

        marker = ""
        if duck.is_belly:
            marker = " ← BELLY (shift loads here)"
        elif duck.is_neck:
            marker = " ← NECK (avoid!)"
        elif duck.curtailment_mw > 0:
            marker = f" ← Curtailing {duck.curtailment_mw/1000:.1f}GW"

        lines.append(f"  {hour:02d}:00 |{bar}{marker}")

    lines.append("     └" + "─" * 45 + "→ Hour")
    lines.append("")
    lines.append("  Solar Generation (GW):  █ = 5GW")
    lines.append("  Best charging window: 10:00 - 15:00 (belly)")
    lines.append("  Avoid: 17:00 - 21:00 (neck ramp)")
    lines.append("")

    return "\n".join(lines)


def visualize_us_comparison() -> str:
    """Compare all US ISOs."""
    lines = []
    lines.append("")
    lines.append("  US Grid Stress Comparison (Current Hour)")
    lines.append("  " + "=" * 75)
    lines.append("")
    lines.append("  ISO          Stress      Level    Renew%   CO2 lb    $/MWh   Dominant")
    lines.append("  " + "-" * 75)

    results = []
    for iso in US_ISO_CONFIGS.keys():
        calc = USGridStressCalculator(iso)
        simulation = calc.simulate_24h()
        current_hour = datetime.now().hour
        _, result = simulation[current_hour]
        results.append((calc.config, result))

    results.sort(key=lambda r: r[1].stress)

    for config, result in results:
        name = config.code[:10]

        if result.stress < 0.0001:
            bar = "▓" * 2
        elif result.stress < 0.001:
            bar = "▓" * 4
        elif result.stress < 0.01:
            bar = "▓" * 6
        else:
            bar = "▓" * 8

        level = result.level.value[:5].upper()
        renew = f"{result.renewable_fraction * 100:5.1f}%"
        co2 = f"{result.co2_intensity_lb_mwh:6.0f}"
        price = f"${result.price_estimate_usd_mwh:5.1f}"

        lines.append(
            f"  {name:10s}  {bar:8s}  {level:5s}  {renew}  {co2}  {price}   {config.dominant_source}"
        )

    lines.append("")
    return "\n".join(lines)


def visualize_extreme_weather(iso: ISORegion, event_name: str) -> str:
    """Show extreme weather impact."""
    if event_name not in EXTREME_WEATHER_SCENARIOS:
        return f"Unknown event: {event_name}"

    event = EXTREME_WEATHER_SCENARIOS[event_name]
    calc = USGridStressCalculator(iso)

    lines = []
    lines.append("")
    lines.append(f"  EXTREME WEATHER STRESS TEST: {event.name}")
    lines.append("  " + "=" * 60)
    lines.append(f"  {event.description}")
    lines.append(f"  Historical: {event.historical_example}")
    lines.append("")
    lines.append(f"  Impact on {calc.config.name}:")
    lines.append(f"    Demand increase: +{(event.demand_multiplier-1)*100:.0f}%")
    lines.append(f"    Renewable loss: -{event.renewable_reduction*100:.0f}%")
    lines.append(f"    Outage probability: {event.outage_probability*100:.0f}%")
    lines.append("")

    # Compare normal vs extreme
    normal = calc.simulate_24h()
    extreme = calc.simulate_24h(weather_event=event_name)

    lines.append("  Hour   Normal Stress   Extreme Stress   Δ Level")
    lines.append("  " + "-" * 55)

    for (t1, r1), (t2, r2) in zip(normal, extreme):
        hour = t1.strftime("%H:00")
        s1 = f"{r1.stress:.6f}"
        s2 = f"{r2.stress:.6f}"

        if r2.level.value != r1.level.value:
            delta = f"{r1.level.value[:4]} → {r2.level.value[:4]}"
        else:
            delta = "—"

        lines.append(f"  {hour}   {s1:14s}   {s2:14s}   {delta}")

    lines.append("")
    return "\n".join(lines)
