"""
Lucas Energy Budget Manager
============================

Maps battery level to available energy units::

    E_available = 840 × (battery% / 100)

Where 840 = Σ L_n (sum of Lucas numbers for dimensions 1-12).

Tasks consume energy based on dimensions used::

    E_task = Σ L_n for required dimensions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from brahim_energy.constants import DIMENSION_NAMES, LUCAS_NUMBERS, TOTAL_STATES

logger = logging.getLogger("brahim_energy.budget.manager")


@dataclass
class EnergyTask:
    """A task with energy requirements."""
    task_id: str
    dimensions_required: List[int]  # 1-12
    value: float
    energy_cost: int = 0

    def __post_init__(self) -> None:
        self.energy_cost = sum(
            LUCAS_NUMBERS[d - 1]
            for d in self.dimensions_required
            if 1 <= d <= 12
        )

    @property
    def efficiency(self) -> float:
        """Value per energy unit."""
        return self.value / self.energy_cost if self.energy_cost > 0 else float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "dimensions_required": self.dimensions_required,
            "value": self.value,
            "energy_cost": self.energy_cost,
            "efficiency": self.efficiency,
        }


@dataclass
class EnergyBudget:
    """Current energy budget state."""
    battery_percent: float
    total_capacity: int
    available_units: float
    used_units: float
    tasks_completed: int
    tasks_queued: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battery_percent": self.battery_percent,
            "total_capacity": self.total_capacity,
            "available_units": self.available_units,
            "used_units": self.used_units,
            "tasks_completed": self.tasks_completed,
            "tasks_queued": self.tasks_queued,
        }


class LucasEnergyBudgetManager:
    """Lucas-based energy budget manager.

    Maps battery level → available energy units::

        E_available = 840 × (battery% / 100)
    """

    def __init__(self, initial_battery_percent: float = 100.0):
        self.battery_percent = initial_battery_percent
        self.total_capacity = TOTAL_STATES  # 840
        self._task_queue: List[EnergyTask] = []
        self._completed_tasks: List[EnergyTask] = []
        self._used_units = 0.0

    @property
    def available_units(self) -> float:
        return self.total_capacity * (self.battery_percent / 100.0) - self._used_units

    def set_battery(self, percent: float) -> None:
        self.battery_percent = max(0, min(100, percent))

    def calculate_energy_cost(self, dimensions: List[int]) -> int:
        """``E_task = Σ L_n`` for *dimensions* (1-12)."""
        return sum(
            LUCAS_NUMBERS[d - 1]
            for d in dimensions
            if 1 <= d <= 12
        )

    def add_task(
        self, task_id: str, dimensions: List[int], value: float
    ) -> EnergyTask:
        task = EnergyTask(
            task_id=task_id,
            dimensions_required=dimensions,
            value=value,
        )
        self._task_queue.append(task)
        return task

    def schedule_optimal(self) -> List[EnergyTask]:
        """Schedule tasks by efficiency (value / cost) descending."""
        available = self.available_units
        sorted_tasks = sorted(
            self._task_queue, key=lambda t: t.efficiency, reverse=True
        )
        executable: List[EnergyTask] = []
        for task in sorted_tasks:
            if task.energy_cost <= available:
                executable.append(task)
                available -= task.energy_cost
        return executable

    def execute_task(self, task_id: str) -> Optional[EnergyTask]:
        """Execute a task by id, consuming its energy."""
        task = None
        for t in self._task_queue:
            if t.task_id == task_id:
                task = t
                break
        if task is None:
            return None
        if task.energy_cost > self.available_units:
            return None
        self._used_units += task.energy_cost
        self._task_queue.remove(task)
        self._completed_tasks.append(task)
        return task

    def execute_optimal(self) -> List[EnergyTask]:
        """Execute tasks in optimal order until budget exhausted."""
        scheduled = self.schedule_optimal()
        executed: List[EnergyTask] = []
        for task in scheduled:
            if self.execute_task(task.task_id):
                executed.append(task)
        return executed

    def get_budget(self) -> EnergyBudget:
        return EnergyBudget(
            battery_percent=self.battery_percent,
            total_capacity=self.total_capacity,
            available_units=self.available_units,
            used_units=self._used_units,
            tasks_completed=len(self._completed_tasks),
            tasks_queued=len(self._task_queue),
        )

    def estimate_battery_life(self, tasks_per_minute: float) -> float:
        """Estimated minutes of battery life at *tasks_per_minute*."""
        if not self._completed_tasks or tasks_per_minute <= 0:
            return float("inf")
        avg_cost = (
            sum(t.energy_cost for t in self._completed_tasks)
            / len(self._completed_tasks)
        )
        energy_per_minute = avg_cost * tasks_per_minute
        return (
            self.available_units / energy_per_minute
            if energy_per_minute > 0
            else float("inf")
        )

    def get_dimension_cost_table(self) -> Dict[str, int]:
        """Energy cost for each dimension."""
        return {DIMENSION_NAMES[i]: LUCAS_NUMBERS[i] for i in range(12)}
