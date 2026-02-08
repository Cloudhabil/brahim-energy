"""Tests for brahim_energy.grid.us."""

from brahim_energy.grid.us import (
    USGridStressCalculator,
    ISORegion,
    US_ISO_CONFIGS,
    EXTREME_WEATHER_SCENARIOS,
    StressLevel,
    calculate_duck_curve,
    classify_stress,
)


class TestUSISOConfigs:
    def test_8_isos(self):
        assert len(US_ISO_CONFIGS) == 8

    def test_caiso_exists(self):
        assert ISORegion.CAISO in US_ISO_CONFIGS
        assert US_ISO_CONFIGS[ISORegion.CAISO].name == "California ISO"

    def test_ercot_exists(self):
        assert ISORegion.ERCOT in US_ISO_CONFIGS

    def test_all_have_capacity(self):
        for iso, cfg in US_ISO_CONFIGS.items():
            assert cfg.installed_capacity_gw > 0, f"{iso} no capacity"


class TestExtremeWeather:
    def test_5_scenarios(self):
        assert len(EXTREME_WEATHER_SCENARIOS) == 5

    def test_polar_vortex(self):
        assert "polar_vortex" in EXTREME_WEATHER_SCENARIOS


class TestUSGridStressCalculator:
    def test_init(self):
        calc = USGridStressCalculator(ISORegion.CAISO)
        assert calc.config.name == "California ISO"

    def test_init_invalid(self):
        import pytest
        with pytest.raises((ValueError, KeyError)):
            USGridStressCalculator("FAKE")

    def test_simulate_24h(self):
        calc = USGridStressCalculator(ISORegion.CAISO)
        results = calc.simulate_24h()
        assert len(results) == 24

    def test_duck_curve_standalone(self):
        duck = calculate_duck_curve(12, 30000, 15000)
        assert hasattr(duck, 'net_load_mw')

    def test_all_isos_simulate(self):
        for iso in US_ISO_CONFIGS:
            calc = USGridStressCalculator(iso)
            results = calc.simulate_24h()
            assert len(results) == 24, f"{iso} sim failed"

    def test_find_optimal_windows(self):
        calc = USGridStressCalculator(ISORegion.ERCOT)
        windows = calc.find_optimal_windows(50000)
        assert isinstance(windows, list)

    def test_annual_co2(self):
        calc = USGridStressCalculator(ISORegion.PJM)
        savings = calc.calculate_annual_co2_savings(1000)
        assert isinstance(savings, dict)
        assert "total" in savings
