"""Tests for brahim_energy.budget.manager."""

from brahim_energy.budget.manager import (
    EnergyBudget,
    EnergyTask,
    LucasEnergyBudgetManager,
)


class TestEnergyTask:
    def test_energy_cost_single_dim(self):
        t = EnergyTask("t1", [1], 10.0)
        assert t.energy_cost == 1  # L(1) = 1

    def test_energy_cost_dim_12(self):
        t = EnergyTask("t1", [12], 10.0)
        assert t.energy_cost == 322  # L(12)

    def test_energy_cost_multi(self):
        t = EnergyTask("t1", [1, 2, 3], 10.0)
        assert t.energy_cost == 1 + 3 + 4  # 8

    def test_efficiency(self):
        t = EnergyTask("t1", [1], 10.0)
        assert t.efficiency == 10.0

    def test_to_dict(self):
        t = EnergyTask("t1", [1, 2], 5.0)
        d = t.to_dict()
        assert d["task_id"] == "t1"
        assert d["energy_cost"] == 4  # 1 + 3


class TestLucasEnergyBudgetManager:
    def test_init(self):
        mgr = LucasEnergyBudgetManager()
        assert mgr.total_capacity == 840
        assert mgr.available_units == 840

    def test_half_battery(self):
        mgr = LucasEnergyBudgetManager(50.0)
        assert abs(mgr.available_units - 420) < 1e-10

    def test_set_battery(self):
        mgr = LucasEnergyBudgetManager()
        mgr.set_battery(50)
        assert mgr.battery_percent == 50

    def test_set_battery_clamp(self):
        mgr = LucasEnergyBudgetManager()
        mgr.set_battery(150)
        assert mgr.battery_percent == 100
        mgr.set_battery(-10)
        assert mgr.battery_percent == 0

    def test_calculate_energy_cost(self):
        mgr = LucasEnergyBudgetManager()
        cost = mgr.calculate_energy_cost([1, 2, 3])
        assert cost == 8

    def test_add_and_execute_task(self):
        mgr = LucasEnergyBudgetManager()
        mgr.add_task("t1", [1, 2], 10.0)
        result = mgr.execute_task("t1")
        assert result is not None
        assert result.task_id == "t1"
        assert mgr.available_units == 840 - 4

    def test_execute_insufficient_energy(self):
        mgr = LucasEnergyBudgetManager(1.0)  # ~8.4 units
        mgr.add_task("t1", [12], 10.0)  # costs 322
        result = mgr.execute_task("t1")
        assert result is None

    def test_schedule_optimal(self):
        mgr = LucasEnergyBudgetManager(10.0)  # 84 units
        mgr.add_task("cheap", [1], 100.0)      # cost=1, eff=100
        mgr.add_task("expensive", [12], 10.0)  # cost=322, eff=0.031
        mgr.add_task("mid", [5], 50.0)         # cost=11, eff=4.5
        scheduled = mgr.schedule_optimal()
        # Should pick "cheap" first (highest efficiency)
        assert scheduled[0].task_id == "cheap"

    def test_execute_optimal(self):
        mgr = LucasEnergyBudgetManager()
        mgr.add_task("a", [1], 10.0)
        mgr.add_task("b", [2], 20.0)
        executed = mgr.execute_optimal()
        assert len(executed) == 2

    def test_get_budget(self):
        mgr = LucasEnergyBudgetManager()
        budget = mgr.get_budget()
        assert isinstance(budget, EnergyBudget)
        assert budget.total_capacity == 840

    def test_dimension_cost_table(self):
        mgr = LucasEnergyBudgetManager()
        table = mgr.get_dimension_cost_table()
        assert len(table) == 12
        assert table["PERCEPTION"] == 1
        assert table["UNIFICATION"] == 322

    def test_estimate_battery_life(self):
        mgr = LucasEnergyBudgetManager()
        mgr.add_task("t1", [1], 10.0)
        mgr.execute_task("t1")
        life = mgr.estimate_battery_life(1.0)
        assert life > 0
        assert life < float("inf")
