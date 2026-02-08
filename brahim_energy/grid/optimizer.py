"""
Onion Grid Optimizer — Core Engine
===================================

Applies traffic congestion mathematics to electrical grid demand optimisation.
Backwards compatible with any existing hardware through protocol adapters.

Mathematical Translation::

    Traffic:  Congestion(t) = Σ(1/(road_capacity - traffic_flow)²) × exp(-λ×t)
    Grid:     Stress(t)     = Σ(1/(grid_capacity - power_demand)²) × exp(-λ×t)

When ``Stress > GENESIS_CONSTANT`` (0.0022) → trigger demand response.
Target: β = 23.6 % peak reduction (Brahim Security Constant).

Brahim Signal Timing Applied to Grid::

    Cycle Length : B[3] = 60 s (demand response window)
    Green Phase  : B[1] = 27 s (normal operation)
    Amber Phase  : |Δ4| =  3 s (ramp warning)
    Red Phase    :        30 s (load curtailment)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from brahim_energy.constants import (
    BETA_SECURITY,
    BRAHIM_CENTER,
    BRAHIM_SEQUENCE,
    BRAHIM_SUM,
    GENESIS_CONSTANT,
    PHI,
)

logger = logging.getLogger("brahim_energy.grid.optimizer")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class NodeType(Enum):
    """Types of grid nodes (analogous to road segments)."""
    TRANSFORMER = auto()
    FEEDER = auto()
    METER = auto()
    GENERATOR = auto()
    STORAGE = auto()
    EV_CHARGER = auto()
    LOAD_CENTER = auto()


class GridStatus(Enum):
    """Grid stress status (analogous to traffic Level of Service)."""
    OPTIMAL = "green"
    NORMAL = "blue"
    CAUTION = "yellow"
    STRESSED = "orange"
    CRITICAL = "red"


class DemandResponsePhase(Enum):
    """Demand response phases (analogous to traffic signal phases)."""
    GREEN = auto()
    AMBER = auto()
    RED = auto()


@dataclass
class GridNode:
    """Universal abstraction for any grid component.

    This is Layer 2 of the Onion Architecture — provides a unified
    interface regardless of the underlying hardware protocol.
    """
    node_id: str
    node_type: NodeType
    capacity_kw: float
    current_demand_kw: float = 0.0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    protocol: str = "simulation"
    controllable: bool = False
    priority: int = 5
    co2_intensity: float = 0.4
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        if self.capacity_kw <= 0:
            return 0.0
        return self.current_demand_kw / self.capacity_kw

    @property
    def headroom_kw(self) -> float:
        return max(0, self.capacity_kw - self.current_demand_kw)

    @property
    def is_overloaded(self) -> bool:
        return self.current_demand_kw > self.capacity_kw

    def stress_contribution(self, epsilon: float = 1.0) -> float:
        """Calculate this node's contribution to grid stress.

        Formula: ``1 / (capacity - demand + epsilon)²``
        """
        headroom = self.capacity_kw - self.current_demand_kw + epsilon
        if headroom <= 0:
            return float("inf")
        return 1.0 / (headroom ** 2)


@dataclass
class GridSnapshot:
    """Complete grid state at a point in time."""
    timestamp: datetime
    nodes: List[GridNode]
    total_capacity_kw: float
    total_demand_kw: float
    stress: float
    status: GridStatus
    renewable_fraction: float = 0.0
    co2_rate_kg_per_kwh: float = 0.4


@dataclass
class StressEvent:
    """Recorded stress event for analysis."""
    timestamp: datetime
    stress: float
    status: GridStatus
    top_contributors: List[Tuple[str, float]]
    recommendation: str


# =============================================================================
# GRID STRESS CALCULATOR (Traffic Math Applied to Grid)
# =============================================================================

class GridStressCalculator:
    """Calculate grid stress using traffic congestion mathematics.

    Grid Formula::

        Stress(t) = Σ(1/(capacity - demand)²) × exp(-λ×t)

    Thresholds (from Brahim constants):
        - GENESIS_CONSTANT (0.0022): Normal → Caution
        - REGULARITY_THRESHOLD (0.0219): Caution → Stressed
        - BETA_SECURITY (0.236): Emergency threshold
    """

    def __init__(
        self,
        genesis_threshold: float = GENESIS_CONSTANT,
        decay_lambda: float = GENESIS_CONSTANT,
        epsilon: float = 1.0,
    ):
        self.genesis_threshold = genesis_threshold
        self.decay_lambda = decay_lambda
        self.epsilon = epsilon
        self._stress_history: List[Tuple[datetime, float]] = []
        self._max_history = 100

    def compute_instantaneous_stress(self, nodes: List[GridNode]) -> float:
        """Compute instantaneous grid stress (no temporal smoothing)."""
        if not nodes:
            return 0.0
        total_stress = 0.0
        for node in nodes:
            contribution = node.stress_contribution(self.epsilon)
            if math.isinf(contribution):
                return float("inf")
            total_stress += contribution
        return total_stress / len(nodes)

    def compute_stress(
        self,
        nodes: List[GridNode],
        timestamp: Optional[datetime] = None,
    ) -> float:
        """Compute temporally-smoothed grid stress."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        instant_stress = self.compute_instantaneous_stress(nodes)
        if math.isinf(instant_stress):
            return instant_stress
        if self._stress_history:
            weighted_stress = instant_stress
            total_weight = 1.0
            for hist_time, hist_stress in self._stress_history[-10:]:
                delta_t = (timestamp - hist_time).total_seconds()
                if delta_t > 0:
                    weight = math.exp(-self.decay_lambda * delta_t)
                    weighted_stress += weight * hist_stress
                    total_weight += weight
            smoothed_stress = weighted_stress / total_weight
        else:
            smoothed_stress = instant_stress
        self._stress_history.append((timestamp, smoothed_stress))
        if len(self._stress_history) > self._max_history:
            self._stress_history.pop(0)
        return smoothed_stress

    def classify_status(self, stress: float) -> GridStatus:
        """Classify grid status based on stress level."""
        g = self.genesis_threshold
        if stress < 0.5 * g:
            return GridStatus.OPTIMAL
        elif stress < g:
            return GridStatus.NORMAL
        elif stress < 2 * g:
            return GridStatus.CAUTION
        elif stress < 5 * g:
            return GridStatus.STRESSED
        else:
            return GridStatus.CRITICAL

    def get_top_contributors(
        self, nodes: List[GridNode], top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Identify nodes contributing most to stress."""
        contributions = [
            (node.node_id, node.stress_contribution(self.epsilon)) for node in nodes
        ]
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions[:top_n]

    def compute_gradient(self, nodes: List[GridNode]) -> Dict[str, float]:
        """Compute stress gradient for each node (∂Stress/∂demand)."""
        gradients: Dict[str, float] = {}
        for node in nodes:
            headroom = node.capacity_kw - node.current_demand_kw + self.epsilon
            if headroom > 0:
                gradient = 2.0 / (headroom ** 3)
            else:
                gradient = float("inf")
            gradients[node.node_id] = gradient
        return gradients


# =============================================================================
# BRAHIM SIGNAL TIMING FOR DEMAND RESPONSE
# =============================================================================

class BrahimSignalTiming:
    """Apply traffic signal timing to demand response windows.

    Timing constants derived from Brahim sequence::

        Cycle : B[3] = 60 s
        Green : B[1] = 27 s
        Amber : |Δ4| =  3 s
        Red   : 30 s
    """

    CYCLE_SECONDS: int = BRAHIM_SEQUENCE[2]  # 60
    GREEN_SECONDS: int = BRAHIM_SEQUENCE[0]  # 27
    AMBER_SECONDS: int = abs(
        BRAHIM_SEQUENCE[3] + BRAHIM_SEQUENCE[6] - BRAHIM_SUM
    )  # 3
    RED_SECONDS: int = CYCLE_SECONDS - GREEN_SECONDS - AMBER_SECONDS  # 30

    def __init__(self) -> None:
        self.cycle_length = timedelta(seconds=self.CYCLE_SECONDS)
        self.green_duration = timedelta(seconds=self.GREEN_SECONDS)
        self.amber_duration = timedelta(seconds=self.AMBER_SECONDS)
        self.red_duration = timedelta(seconds=self.RED_SECONDS)

    def get_current_phase(
        self, timestamp: Optional[datetime] = None
    ) -> Tuple[DemandResponsePhase, float]:
        """Return ``(phase, seconds_remaining)``."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        cycle_position = timestamp.timestamp() % self.CYCLE_SECONDS
        if cycle_position < self.GREEN_SECONDS:
            return DemandResponsePhase.GREEN, self.GREEN_SECONDS - cycle_position
        elif cycle_position < self.GREEN_SECONDS + self.AMBER_SECONDS:
            return (
                DemandResponsePhase.AMBER,
                (self.GREEN_SECONDS + self.AMBER_SECONDS) - cycle_position,
            )
        else:
            return DemandResponsePhase.RED, self.CYCLE_SECONDS - cycle_position

    def get_next_green_window(
        self, timestamp: Optional[datetime] = None
    ) -> Tuple[datetime, datetime]:
        """Return ``(start, end)`` of the next green window."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        current_phase, remaining = self.get_current_phase(timestamp)
        if current_phase == DemandResponsePhase.GREEN:
            start = timestamp
            end = timestamp + timedelta(seconds=remaining)
        else:
            start = timestamp + timedelta(seconds=remaining)
            if current_phase == DemandResponsePhase.AMBER:
                start += self.red_duration
            end = start + self.green_duration
        return start, end


# =============================================================================
# ONION GRID OPTIMIZER (Main Engine)
# =============================================================================

class OnionGridOptimizer:
    """Main grid optimisation engine using Brahim Onion Architecture."""

    def __init__(
        self,
        stress_calculator: Optional[GridStressCalculator] = None,
        signal_timing: Optional[BrahimSignalTiming] = None,
        beta_target: float = BETA_SECURITY,
        genesis_threshold: float = GENESIS_CONSTANT,
    ):
        self.stress_calculator = stress_calculator or GridStressCalculator()
        self.signal_timing = signal_timing or BrahimSignalTiming()
        self.beta_target = beta_target
        self.genesis_threshold = genesis_threshold
        self._nodes: Dict[str, GridNode] = {}
        self._events: List[StressEvent] = []
        self._max_events = 1000
        self._on_stress_change: List[Callable[[float, GridStatus], None]] = []
        self._on_demand_response: List[Callable[[DemandResponsePhase], None]] = []

    # --- Node management ---------------------------------------------------

    def register_node(self, node: GridNode) -> None:
        self._nodes[node.node_id] = node

    def update_node(
        self,
        node_id: str,
        current_demand_kw: Optional[float] = None,
        capacity_kw: Optional[float] = None,
        **metadata: Any,
    ) -> None:
        if node_id not in self._nodes:
            return
        node = self._nodes[node_id]
        if current_demand_kw is not None:
            node.current_demand_kw = current_demand_kw
        if capacity_kw is not None:
            node.capacity_kw = capacity_kw
        if metadata:
            node.metadata.update(metadata)

    def get_node(self, node_id: str) -> Optional[GridNode]:
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[GridNode]:
        return list(self._nodes.values())

    # --- Stress analysis ---------------------------------------------------

    def analyze(self, timestamp: Optional[datetime] = None) -> GridSnapshot:
        """Perform complete grid analysis."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        nodes = self.get_all_nodes()
        if not nodes:
            return GridSnapshot(
                timestamp=timestamp,
                nodes=[],
                total_capacity_kw=0,
                total_demand_kw=0,
                stress=0,
                status=GridStatus.OPTIMAL,
            )
        total_capacity = sum(n.capacity_kw for n in nodes)
        total_demand = sum(n.current_demand_kw for n in nodes)
        stress = self.stress_calculator.compute_stress(nodes, timestamp)
        status = self.stress_calculator.classify_status(stress)
        generators = [n for n in nodes if n.node_type == NodeType.GENERATOR]
        if generators:
            renewable_capacity = sum(
                n.current_demand_kw
                for n in generators
                if n.metadata.get("renewable", False)
            )
            total_generation = sum(n.current_demand_kw for n in generators)
            renewable_fraction = (
                renewable_capacity / total_generation if total_generation > 0 else 0
            )
        else:
            renewable_fraction = 0

        # Average CO2 — stdlib mean (replaces np.mean)
        vals = [n.co2_intensity for n in nodes]
        avg_co2 = sum(vals) / len(vals) if vals else 0.4

        snapshot = GridSnapshot(
            timestamp=timestamp,
            nodes=nodes,
            total_capacity_kw=total_capacity,
            total_demand_kw=total_demand,
            stress=stress,
            status=status,
            renewable_fraction=renewable_fraction,
            co2_rate_kg_per_kwh=avg_co2,
        )
        if status in (GridStatus.CAUTION, GridStatus.STRESSED, GridStatus.CRITICAL):
            self._record_stress_event(snapshot)
        for callback in self._on_stress_change:
            try:
                callback(stress, status)
            except Exception:
                pass
        return snapshot

    def _record_stress_event(self, snapshot: GridSnapshot) -> None:
        contributors = self.stress_calculator.get_top_contributors(snapshot.nodes)
        if snapshot.status == GridStatus.CRITICAL:
            recommendation = "IMMEDIATE: Activate emergency load shedding"
        elif snapshot.status == GridStatus.STRESSED:
            recommendation = (
                f"URGENT: Reduce load at {contributors[0][0]} "
                f"by {self.beta_target * 100:.1f}%"
            )
        else:
            recommendation = f"ADVISORY: Monitor {contributors[0][0]}"
        event = StressEvent(
            timestamp=snapshot.timestamp,
            stress=snapshot.stress,
            status=snapshot.status,
            top_contributors=contributors,
            recommendation=recommendation,
        )
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events.pop(0)

    # --- Optimisation (Method of Characteristics) --------------------------

    def compute_optimal_load_shift(
        self, target_reduction_kw: float
    ) -> List[Tuple[str, float]]:
        """Compute optimal load shifting using gradient descent."""
        nodes = self.get_all_nodes()
        controllable = [
            n for n in nodes if n.controllable and n.current_demand_kw > 0
        ]
        controllable.sort(key=lambda n: (-n.priority, -n.current_demand_kw))
        if not controllable:
            return []
        reductions: List[Tuple[str, float]] = []
        remaining = target_reduction_kw
        for node in controllable:
            if remaining <= 0:
                break
            max_reduction = node.current_demand_kw * 0.5
            reduction = min(max_reduction, remaining)
            if reduction > 0:
                reductions.append((node.node_id, reduction))
                remaining -= reduction
        return reductions

    def compute_beta_target_reduction(self) -> float:
        """kW reduction needed to achieve β compression target."""
        total_demand = sum(n.current_demand_kw for n in self.get_all_nodes())
        return total_demand * self.beta_target

    # --- Demand response coordination -------------------------------------

    def get_demand_response_state(
        self, timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get current demand response state."""
        snapshot = self.analyze(timestamp)
        phase, remaining = self.signal_timing.get_current_phase(timestamp)
        dr_active = (
            snapshot.status in (GridStatus.STRESSED, GridStatus.CRITICAL)
            or phase == DemandResponsePhase.RED
        )
        if dr_active:
            target_reduction = self.compute_beta_target_reduction()
            load_shifts = self.compute_optimal_load_shift(target_reduction)
        else:
            target_reduction = 0
            load_shifts = []
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "grid_stress": snapshot.stress,
            "status": snapshot.status.value,
            "genesis_threshold": self.genesis_threshold,
            "demand_response": {
                "active": dr_active,
                "phase": phase.name,
                "phase_remaining_seconds": remaining,
                "cycle_length_seconds": self.signal_timing.CYCLE_SECONDS,
            },
            "optimization": {
                "target_reduction_kw": target_reduction,
                "beta_compression": self.beta_target,
                "recommended_shifts": load_shifts,
            },
            "totals": {
                "capacity_kw": snapshot.total_capacity_kw,
                "demand_kw": snapshot.total_demand_kw,
                "utilization": (
                    snapshot.total_demand_kw / snapshot.total_capacity_kw
                    if snapshot.total_capacity_kw > 0
                    else 0
                ),
                "renewable_fraction": snapshot.renewable_fraction,
            },
            "co2": {
                "current_rate_kg_per_kwh": snapshot.co2_rate_kg_per_kwh,
                "potential_savings_kg_per_hour": (
                    target_reduction * snapshot.co2_rate_kg_per_kwh if dr_active else 0
                ),
            },
            "brahim_metrics": {
                "sequence": list(BRAHIM_SEQUENCE),
                "sum": BRAHIM_SUM,
                "center": BRAHIM_CENTER,
                "phi": PHI,
                "beta": BETA_SECURITY,
                "genesis": GENESIS_CONSTANT,
            },
        }

    # --- Callbacks ---------------------------------------------------------

    def on_stress_change(
        self, callback: Callable[[float, GridStatus], None]
    ) -> None:
        self._on_stress_change.append(callback)

    def on_demand_response(
        self, callback: Callable[[DemandResponsePhase], None]
    ) -> None:
        self._on_demand_response.append(callback)
