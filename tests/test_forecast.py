"""Tests for brahim_energy.forecast.demand."""

from brahim_energy.forecast.demand import (
    DemandForecaster,
    DemandTrend,
)


class TestDemandForecaster:
    def test_init(self):
        df = DemandForecaster(max_demand=10_000)
        assert df.max_demand == 10_000

    def test_forecast_no_history(self):
        df = DemandForecaster(max_demand=10_000, growth_rate=0.15)
        forecasts = df.forecast(12)
        assert len(forecasts) == 12
        for f in forecasts:
            assert f.predicted_demand > 0
            assert 0 <= f.confidence <= 1.0

    def test_forecast_with_history(self):
        df = DemandForecaster(max_demand=10_000, growth_rate=0.15)
        df.add_historical_data([
            (1, 1200), (2, 1450), (3, 1800), (4, 2100),
        ])
        forecasts = df.forecast(6)
        assert len(forecasts) == 6
        # All periods should be after last historical
        assert forecasts[0].period == 5

    def test_saturation_analysis(self):
        df = DemandForecaster(max_demand=10_000, growth_rate=0.15)
        analysis = df.get_saturation_analysis()
        assert "max_demand" in analysis
        assert "phi_ceiling" in analysis
        assert analysis["phi_ceiling"] > analysis["max_demand"]

    def test_normalize_to_214(self):
        df = DemandForecaster(max_demand=10_000)
        forecasts = df.forecast(10)
        normalized = df.normalize_to_214(forecasts)
        assert len(normalized) == 10
        assert abs(sum(normalized) - 214) < 0.01

    def test_trend_types(self):
        # Verify enum values
        assert DemandTrend.SATURATING.value == "saturating"
        assert DemandTrend.GROWING.value == "growing"
        assert DemandTrend.DECLINING.value == "declining"
        assert DemandTrend.STABLE.value == "stable"

    def test_high_growth_rate(self):
        df = DemandForecaster(max_demand=1000, growth_rate=1.0)
        forecasts = df.forecast(5)
        # With high growth, should approach saturation quickly
        assert forecasts[-1].saturation_level > 0.1

    def test_confidence_decreases(self):
        df = DemandForecaster(max_demand=10_000)
        forecasts = df.forecast(10)
        # Confidence should decrease over time
        assert forecasts[0].confidence > forecasts[-1].confidence
