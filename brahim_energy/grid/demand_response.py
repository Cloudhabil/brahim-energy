"""
Demand Response Orchestrator — CO2-Aware Load Shifting
======================================================

Intelligent demand response using Brahim mathematics for optimal CO2
reduction through load shifting.

Key mechanisms:

1. Peak Shaving — shift loads from peak (dirty) to off-peak (cleaner).
2. Renewable Integration — pre-position loads for solar / wind windows.
3. EV Smart Charging — optimise EV charging for grid carbon intensity.
4. Emergency Response — rapid load curtailment during stress events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from brahim_energy.constants import (
    BETA_SECURITY,
    GENESIS_CONSTANT,
    PHI,
)
from brahim_energy.grid.optimizer import (
    GridNode,
    GridSnapshot,
    GridStatus,
    NodeType,
    OnionGridOptimizer,
)

logger = logging.getLogger("brahim_energy.grid.demand_response")


# =============================================================================
# DATA CLASSES
# =============================================================================

class LoadShiftType(Enum):
    """Types of load shifting actions."""
    DEFER = auto()
    ADVANCE = auto()
    CURTAIL = auto()
    MODULATE = auto()
    INTERRUPT = auto()


class CO2IntensityLevel(Enum):
    """Grid carbon intensity levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class LoadShiftCommand:
    """Command to shift load at a specific node."""
    command_id: str
    node_id: str
    shift_type: LoadShiftType
    amount_kw: float
    from_time: datetime
    to_time: datetime
    duration_minutes: float
    priority: int = 5
    co2_savings_kg: float = 0.0
    cost_savings_eur: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "node_id": self.node_id,
            "shift_type": self.shift_type.name,
            "amount_kw": self.amount_kw,
            "from_time": self.from_time.isoformat(),
            "to_time": self.to_time.isoformat(),
            "duration_minutes": self.duration_minutes,
            "priority": self.priority,
            "co2_savings_kg": self.co2_savings_kg,
            "cost_savings_eur": self.cost_savings_eur,
            "reason": self.reason,
        }


@dataclass
class CO2Forecast:
    """Carbon intensity forecast for a time period."""
    timestamp: datetime
    intensity_kg_per_kwh: float
    level: CO2IntensityLevel
    renewable_fraction: float
    confidence: float = 0.8


@dataclass
class DemandResponseEvent:
    """Record of a demand response event."""
    event_id: str
    timestamp: datetime
    trigger: str
    status: GridStatus
    commands_issued: List[LoadShiftCommand]
    total_reduction_kw: float
    total_co2_saved_kg: float
    duration_minutes: float


# =============================================================================
# CO2 CALCULATOR
# =============================================================================

class CO2Calculator:
    """Calculate CO2 emissions and savings from load shifting.

    Uses time-of-day carbon intensity curves based on typical
    generation mix patterns.

    Spanish Grid Average (2024)::

        Night     (00-06): 0.25 kg/kWh (nuclear + wind)
        Morning   (06-12): 0.35 kg/kWh (ramp up)
        Afternoon (12-18): 0.30 kg/kWh (solar peak)
        Evening   (18-24): 0.45 kg/kWh (gas peakers)
    """

    DEFAULT_INTENSITY_PROFILE: Dict[int, float] = {
        0: 0.25, 1: 0.24, 2: 0.23, 3: 0.22, 4: 0.23, 5: 0.24,
        6: 0.28, 7: 0.32, 8: 0.35, 9: 0.36, 10: 0.35, 11: 0.33,
        12: 0.30, 13: 0.28, 14: 0.27, 15: 0.28, 16: 0.30, 17: 0.35,
        18: 0.42, 19: 0.45, 20: 0.44, 21: 0.40, 22: 0.35, 23: 0.30,
    }

    def __init__(
        self,
        intensity_profile: Optional[Dict[int, float]] = None,
        electricity_price_eur_kwh: float = 0.15,
    ):
        self.intensity_profile = intensity_profile or dict(self.DEFAULT_INTENSITY_PROFILE)
        self.electricity_price = electricity_price_eur_kwh

    def get_intensity(self, timestamp: datetime) -> float:
        return self.intensity_profile.get(timestamp.hour, 0.35)

    def get_intensity_level(self, intensity: float) -> CO2IntensityLevel:
        if intensity < 0.2:
            return CO2IntensityLevel.VERY_LOW
        elif intensity < 0.3:
            return CO2IntensityLevel.LOW
        elif intensity < 0.4:
            return CO2IntensityLevel.MEDIUM
        elif intensity < 0.5:
            return CO2IntensityLevel.HIGH
        else:
            return CO2IntensityLevel.VERY_HIGH

    def forecast(
        self, start_time: datetime, hours: int = 24
    ) -> List[CO2Forecast]:
        forecasts: List[CO2Forecast] = []
        for h in range(hours):
            timestamp = start_time + timedelta(hours=h)
            intensity = self.get_intensity(timestamp)
            level = self.get_intensity_level(intensity)
            renewable = max(0.0, min(1.0, 1 - (intensity - 0.2) / 0.3))
            forecasts.append(
                CO2Forecast(
                    timestamp=timestamp,
                    intensity_kg_per_kwh=intensity,
                    level=level,
                    renewable_fraction=renewable,
                )
            )
        return forecasts

    def calculate_savings(
        self, amount_kwh: float, from_time: datetime, to_time: datetime
    ) -> Tuple[float, float]:
        """Return ``(co2_savings_kg, cost_savings_eur)``."""
        from_intensity = self.get_intensity(from_time)
        to_intensity = self.get_intensity(to_time)
        co2_savings = amount_kwh * (from_intensity - to_intensity)
        from_peak = 1.5 if 18 <= from_time.hour <= 22 else 1.0
        to_peak = 1.5 if 18 <= to_time.hour <= 22 else 1.0
        cost_savings = (
            amount_kwh * self.electricity_price * (from_peak - to_peak) / from_peak
        )
        return max(0.0, co2_savings), max(0.0, cost_savings)

    def find_optimal_shift_window(
        self,
        current_time: datetime,
        duration_hours: float,
        look_ahead_hours: int = 24,
    ) -> Tuple[datetime, float]:
        """Find the time window with lowest average CO2 intensity."""
        forecasts = self.forecast(current_time, look_ahead_hours)
        best_start = current_time
        best_intensity = float("inf")
        for i, forecast in enumerate(forecasts):
            remaining = look_ahead_hours - i
            if remaining < duration_hours:
                break
            end_idx = min(i + int(duration_hours), len(forecasts))
            # stdlib mean (replaces np.mean)
            window = [forecasts[j].intensity_kg_per_kwh for j in range(i, end_idx)]
            avg_intensity = sum(window) / len(window) if window else float("inf")
            if avg_intensity < best_intensity:
                best_intensity = avg_intensity
                best_start = forecast.timestamp
        return best_start, best_intensity


# =============================================================================
# DEMAND RESPONSE ORCHESTRATOR
# =============================================================================

class DemandResponseOrchestrator:
    """Orchestrate demand response for CO2 reduction."""

    def __init__(
        self,
        optimizer: OnionGridOptimizer,
        co2_calculator: Optional[CO2Calculator] = None,
        beta_target: float = BETA_SECURITY,
        min_shift_kw: float = 1.0,
        max_shift_duration_hours: float = 8.0,
    ):
        self.optimizer = optimizer
        self.co2_calculator = co2_calculator or CO2Calculator()
        self.beta_target = beta_target
        self.min_shift_kw = min_shift_kw
        self.max_shift_duration = timedelta(hours=max_shift_duration_hours)
        self._events: List[DemandResponseEvent] = []
        self._command_counter = 0
        self._on_command: List[Callable[[LoadShiftCommand], None]] = []

    # --- Load analysis -----------------------------------------------------

    def identify_shiftable_loads(
        self, snapshot: Optional[GridSnapshot] = None
    ) -> List[GridNode]:
        if snapshot is None:
            snapshot = self.optimizer.analyze()
        shiftable = [
            node
            for node in snapshot.nodes
            if node.controllable
            and node.current_demand_kw >= self.min_shift_kw
            and node.priority >= 3
            and node.node_type
            in (NodeType.METER, NodeType.EV_CHARGER, NodeType.LOAD_CENTER)
        ]
        shiftable.sort(key=lambda n: (-n.priority, -n.current_demand_kw))
        return shiftable

    def calculate_target_reduction(
        self, snapshot: Optional[GridSnapshot] = None
    ) -> float:
        if snapshot is None:
            snapshot = self.optimizer.analyze()
        stress_ratio = snapshot.stress / GENESIS_CONSTANT
        if stress_ratio <= 1.0:
            return 0.0
        reduction_factor = min(self.beta_target, (stress_ratio - 1) * 0.1)
        return snapshot.total_demand_kw * reduction_factor

    # --- Command generation ------------------------------------------------

    def generate_shift_commands(
        self,
        target_reduction_kw: float,
        shiftable_loads: List[GridNode],
        current_time: Optional[datetime] = None,
    ) -> List[LoadShiftCommand]:
        if current_time is None:
            current_time = datetime.utcnow()
        commands: List[LoadShiftCommand] = []
        remaining_reduction = target_reduction_kw
        for node in shiftable_loads:
            if remaining_reduction <= 0:
                break
            shift_amount = min(node.current_demand_kw * 0.5, remaining_reduction)
            if shift_amount < self.min_shift_kw:
                continue
            duration = 1.0
            optimal_time, optimal_intensity = (
                self.co2_calculator.find_optimal_shift_window(
                    current_time, duration, look_ahead_hours=24
                )
            )
            energy_kwh = shift_amount * duration
            co2_savings, cost_savings = self.co2_calculator.calculate_savings(
                energy_kwh, current_time, optimal_time
            )
            self._command_counter += 1
            command = LoadShiftCommand(
                command_id=(
                    f"DR_{current_time.strftime('%Y%m%d%H%M%S')}"
                    f"_{self._command_counter:04d}"
                ),
                node_id=node.node_id,
                shift_type=LoadShiftType.DEFER,
                amount_kw=shift_amount,
                from_time=current_time,
                to_time=optimal_time,
                duration_minutes=duration * 60,
                priority=node.priority,
                co2_savings_kg=co2_savings,
                cost_savings_eur=cost_savings,
                reason="Stress reduction: shift to low-carbon window",
                metadata={
                    "from_intensity": self.co2_calculator.get_intensity(current_time),
                    "to_intensity": optimal_intensity,
                },
            )
            commands.append(command)
            remaining_reduction -= shift_amount
        return commands

    # --- Execution ---------------------------------------------------------

    def execute_demand_response(
        self, trigger: str = "stress", force: bool = False
    ) -> Optional[DemandResponseEvent]:
        current_time = datetime.utcnow()
        snapshot = self.optimizer.analyze()
        if not force and snapshot.status in (GridStatus.OPTIMAL, GridStatus.NORMAL):
            return None
        target_reduction = self.calculate_target_reduction(snapshot)
        if target_reduction < self.min_shift_kw and not force:
            return None
        shiftable = self.identify_shiftable_loads(snapshot)
        if not shiftable:
            return None
        commands = self.generate_shift_commands(
            target_reduction, shiftable, current_time
        )
        if not commands:
            return None
        for command in commands:
            for callback in self._on_command:
                try:
                    callback(command)
                except Exception:
                    pass
        total_reduction = sum(c.amount_kw for c in commands)
        total_co2_saved = sum(c.co2_savings_kg for c in commands)
        # stdlib mean (replaces np.mean)
        dur_vals = [c.duration_minutes for c in commands]
        avg_duration = sum(dur_vals) / len(dur_vals)
        event = DemandResponseEvent(
            event_id=f"DRE_{current_time.strftime('%Y%m%d%H%M%S')}",
            timestamp=current_time,
            trigger=trigger,
            status=snapshot.status,
            commands_issued=commands,
            total_reduction_kw=total_reduction,
            total_co2_saved_kg=total_co2_saved,
            duration_minutes=avg_duration,
        )
        self._events.append(event)
        return event

    # --- Scheduled optimisation --------------------------------------------

    def optimize_for_carbon(
        self, look_ahead_hours: int = 24
    ) -> List[LoadShiftCommand]:
        current_time = datetime.utcnow()
        snapshot = self.optimizer.analyze()
        forecasts = self.co2_calculator.forecast(current_time, look_ahead_hours)
        low_carbon_windows = [
            f
            for f in forecasts
            if f.level in (CO2IntensityLevel.VERY_LOW, CO2IntensityLevel.LOW)
        ]
        if not low_carbon_windows:
            return []
        flexible_loads = [
            node
            for node in snapshot.nodes
            if node.controllable
            and node.node_type in (NodeType.EV_CHARGER, NodeType.STORAGE)
        ]
        commands: List[LoadShiftCommand] = []
        for node in flexible_loads:
            best_window = min(
                low_carbon_windows, key=lambda f: f.intensity_kg_per_kwh
            )
            current_intensity = self.co2_calculator.get_intensity(current_time)
            energy_kwh = node.current_demand_kw * 1.0
            co2_savings = energy_kwh * (
                current_intensity - best_window.intensity_kg_per_kwh
            )
            if co2_savings > 0.1:
                self._command_counter += 1
                command = LoadShiftCommand(
                    command_id=(
                        f"OPT_{current_time.strftime('%Y%m%d')}"
                        f"_{self._command_counter:04d}"
                    ),
                    node_id=node.node_id,
                    shift_type=LoadShiftType.DEFER,
                    amount_kw=node.current_demand_kw,
                    from_time=current_time,
                    to_time=best_window.timestamp,
                    duration_minutes=60.0,
                    priority=node.priority,
                    co2_savings_kg=co2_savings,
                    reason=(
                        f"Carbon optimization: move to {best_window.level.value} window"
                    ),
                )
                commands.append(command)
        return commands

    # --- Reporting ---------------------------------------------------------

    def get_statistics(
        self, since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        events = self._events
        if since:
            events = [e for e in events if e.timestamp >= since]
        if not events:
            return {
                "total_events": 0,
                "total_commands": 0,
                "total_reduction_kwh": 0,
                "total_co2_saved_kg": 0,
            }
        # stdlib mean (replaces np.mean)
        red_vals = [e.total_reduction_kw for e in events]
        co2_vals = [e.total_co2_saved_kg for e in events]
        return {
            "total_events": len(events),
            "total_commands": sum(len(e.commands_issued) for e in events),
            "total_reduction_kwh": sum(
                e.total_reduction_kw * e.duration_minutes / 60 for e in events
            ),
            "total_co2_saved_kg": sum(co2_vals),
            "average_reduction_per_event_kw": sum(red_vals) / len(red_vals),
            "average_co2_saved_per_event_kg": sum(co2_vals) / len(co2_vals),
            "triggers": {
                trigger: len([e for e in events if e.trigger == trigger])
                for trigger in set(e.trigger for e in events)
            },
            "brahim_metrics": {
                "beta_target": self.beta_target,
                "genesis_threshold": GENESIS_CONSTANT,
                "phi": PHI,
            },
        }

    def get_co2_report(self, period_hours: int = 24) -> Dict[str, Any]:
        since = datetime.utcnow() - timedelta(hours=period_hours)
        stats = self.get_statistics(since)
        co2_saved: float = stats.get("total_co2_saved_kg", 0)
        return {
            "period_hours": period_hours,
            "total_co2_saved_kg": co2_saved,
            "equivalents": {
                "car_km_avoided": co2_saved / 0.12,
                "trees_planted_equivalent": co2_saved / 22,
                "smartphone_charges": co2_saved / 0.005,
            },
            "annual_projection_kg": co2_saved * (8760 / period_hours),
            "annual_projection_tons": co2_saved * (8760 / period_hours) / 1000,
        }

    def on_command(self, callback: Callable[[LoadShiftCommand], None]) -> None:
        self._on_command.append(callback)
