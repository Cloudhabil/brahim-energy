"""Tests for brahim_energy.grid.demand_response."""

from datetime import datetime, timedelta

from brahim_energy.grid.demand_response import (
    CO2Calculator,
    CO2IntensityLevel,
    DemandResponseOrchestrator,
    LoadShiftCommand,
    LoadShiftType,
)
from brahim_energy.grid.optimizer import (
    GridNode,
    NodeType,
    OnionGridOptimizer,
)


class TestCO2Calculator:
    def test_default_profile_24h(self):
        co2 = CO2Calculator()
        assert len(co2.intensity_profile) == 24

    def test_intensity_evening_peak(self):
        co2 = CO2Calculator()
        dt_evening = datetime(2026, 1, 1, 19, 0)
        dt_night = datetime(2026, 1, 1, 3, 0)
        assert co2.get_intensity(dt_evening) > co2.get_intensity(dt_night)

    def test_intensity_level(self):
        co2 = CO2Calculator()
        assert co2.get_intensity_level(0.15) == CO2IntensityLevel.VERY_LOW
        assert co2.get_intensity_level(0.25) == CO2IntensityLevel.LOW
        assert co2.get_intensity_level(0.35) == CO2IntensityLevel.MEDIUM
        assert co2.get_intensity_level(0.45) == CO2IntensityLevel.HIGH
        assert co2.get_intensity_level(0.55) == CO2IntensityLevel.VERY_HIGH

    def test_forecast_length(self):
        co2 = CO2Calculator()
        fc = co2.forecast(datetime(2026, 1, 1), 48)
        assert len(fc) == 48

    def test_savings_positive(self):
        co2 = CO2Calculator()
        peak = datetime(2026, 1, 1, 19, 0)
        night = datetime(2026, 1, 1, 3, 0)
        co2_s, cost_s = co2.calculate_savings(100.0, peak, night)
        assert co2_s > 0

    def test_savings_same_time(self):
        co2 = CO2Calculator()
        t = datetime(2026, 1, 1, 12, 0)
        co2_s, _ = co2.calculate_savings(100.0, t, t)
        assert co2_s == 0.0

    def test_optimal_shift_window(self):
        co2 = CO2Calculator()
        start, intensity = co2.find_optimal_shift_window(
            datetime(2026, 1, 1, 18, 0), 2.0, 24
        )
        assert intensity < 0.45  # better than evening peak


class TestDemandResponseOrchestrator:
    def _make_optimizer_with_nodes(self):
        opt = OnionGridOptimizer()
        for i in range(5):
            opt.register_node(
                GridNode(
                    node_id=f"meter_{i}",
                    node_type=NodeType.METER,
                    capacity_kw=100,
                    current_demand_kw=80,
                    controllable=True,
                    priority=5 + i,
                )
            )
        return opt

    def test_init(self):
        opt = self._make_optimizer_with_nodes()
        dro = DemandResponseOrchestrator(opt)
        assert dro.beta_target > 0

    def test_identify_shiftable(self):
        opt = self._make_optimizer_with_nodes()
        dro = DemandResponseOrchestrator(opt)
        shiftable = dro.identify_shiftable_loads()
        assert len(shiftable) > 0

    def test_statistics_empty(self):
        opt = OnionGridOptimizer()
        dro = DemandResponseOrchestrator(opt)
        stats = dro.get_statistics()
        assert stats["total_events"] == 0

    def test_co2_report(self):
        opt = OnionGridOptimizer()
        dro = DemandResponseOrchestrator(opt)
        report = dro.get_co2_report(24)
        assert "total_co2_saved_kg" in report
        assert "equivalents" in report
