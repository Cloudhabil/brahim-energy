"""Tests for brahim_energy.grid.eu."""

from brahim_energy.grid.eu import (
    BrahimGridStressCalculator,
    CountryGridConfig,
    EU_GRID_CONFIGS,
    GridState,
    StressLevel,
    StressResult,
    classify_stress,
    visualize_24h,
)


class TestEUGridConfigs:
    def test_12_countries(self):
        assert len(EU_GRID_CONFIGS) == 12

    def test_germany_exists(self):
        assert "DE" in EU_GRID_CONFIGS
        assert EU_GRID_CONFIGS["DE"].name == "Germany"

    def test_spain_exists(self):
        assert "ES" in EU_GRID_CONFIGS

    def test_all_have_capacity(self):
        for code, cfg in EU_GRID_CONFIGS.items():
            assert cfg.installed_capacity_gw > 0, f"{code} has no capacity"

    def test_all_have_population(self):
        for code, cfg in EU_GRID_CONFIGS.items():
            assert cfg.population_millions > 0, f"{code} has no population"


class TestClassifyStress:
    def test_excellent(self):
        assert classify_stress(0.00005) == StressLevel.EXCELLENT

    def test_good(self):
        assert classify_stress(0.0005) == StressLevel.GOOD

    def test_moderate(self):
        assert classify_stress(0.005) == StressLevel.MODERATE

    def test_high(self):
        assert classify_stress(0.05) == StressLevel.HIGH

    def test_critical(self):
        assert classify_stress(0.5) == StressLevel.CRITICAL


class TestBrahimGridStressCalculator:
    def test_init_valid(self):
        calc = BrahimGridStressCalculator("DE")
        assert calc.config.name == "Germany"

    def test_init_invalid(self):
        import pytest
        with pytest.raises(ValueError):
            BrahimGridStressCalculator("XX")

    def test_calculate_stress(self):
        calc = BrahimGridStressCalculator("DE")
        stress = calc.calculate_stress(100_000, 50_000, 0.5)
        assert stress > 0
        assert stress < 1

    def test_stress_infinite_at_capacity(self):
        calc = BrahimGridStressCalculator("DE")
        stress = calc.calculate_stress(100, 100)
        assert stress == float("inf")

    def test_simulate_24h(self):
        calc = BrahimGridStressCalculator("DE")
        results = calc.simulate_24h(40_000)
        assert len(results) == 24
        for t, r in results:
            assert isinstance(r, StressResult)

    def test_simulate_24h_default(self):
        calc = BrahimGridStressCalculator("ES")
        results = calc.simulate_24h()
        assert len(results) == 24

    def test_find_optimal_windows(self):
        calc = BrahimGridStressCalculator("DE")
        windows = calc.find_optimal_windows(2.0, 0.01)
        # Should find at least some windows in a 24h period
        assert isinstance(windows, list)

    def test_co2_savings(self):
        calc = BrahimGridStressCalculator("DE")
        results = calc.simulate_24h(40_000)
        # Compare peak and off-peak
        if len(results) >= 5:
            savings = calc.calculate_co2_savings(
                1000, results[18][1], results[3][1]
            )
            assert isinstance(savings, float)

    def test_all_countries_simulate(self):
        for code in EU_GRID_CONFIGS:
            calc = BrahimGridStressCalculator(code)
            results = calc.simulate_24h()
            assert len(results) == 24, f"{code} sim failed"

    def test_visualize(self):
        calc = BrahimGridStressCalculator("DE")
        viz = visualize_24h(calc)
        assert "Germany" in viz
        assert len(viz) > 100
