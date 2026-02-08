"""
Demand Forecaster — PHI-saturation Demand Prediction
=====================================================

Predicts demand using PHI-saturation curves that model market saturation
following golden-ratio principles.  The saturation curve asymptotically
approaches the PHI-scaled maximum demand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from brahim_energy.constants import (
    BRAHIM_CENTER,
    BRAHIM_SEQUENCE,
    BRAHIM_SUM,
    LUCAS_NUMBERS,
    PHI,
)


class DemandTrend(Enum):
    """Demand trend classifications."""
    SATURATING = "saturating"
    GROWING = "growing"
    DECLINING = "declining"
    STABLE = "stable"


@dataclass
class DemandDataPoint:
    """A single demand observation."""
    period: int
    demand: float
    actual: Optional[float] = None


@dataclass
class DemandForecast:
    """Forecast result for a period."""
    period: int
    predicted_demand: float
    saturation_level: float
    confidence: float
    trend: DemandTrend


class DemandForecaster:
    """PHI-saturation demand forecaster.

    Uses golden-ratio principles to model demand saturation curves.
    The PHI-saturation model assumes demand approaches a maximum
    following the logistic curve scaled by PHI.
    """

    def __init__(self, max_demand: float, growth_rate: float = 0.1):
        self.max_demand = max_demand
        self.growth_rate = growth_rate
        self.historical_data: List[DemandDataPoint] = []
        self._phi_ceiling = max_demand * PHI
        self._forecasts: List[DemandForecast] = []

    def add_historical_data(self, data: List[Tuple[int, float]]) -> None:
        """Add historical demand data as ``(period, demand)`` tuples."""
        for period, demand in data:
            self.historical_data.append(DemandDataPoint(period, demand, demand))

    def _phi_saturation_curve(self, t: float, midpoint: float) -> float:
        """PHI-scaled logistic saturation at time *t*."""
        exponent = -PHI * self.growth_rate * (t - midpoint)
        exponent = max(min(exponent, 700), -700)
        return self.max_demand / (1 + math.exp(exponent))

    def _calculate_saturation_level(self, demand: float) -> float:
        return min(demand / self._phi_ceiling, 1.0)

    def _apply_brahim_correction(self, base_forecast: float, period: int) -> float:
        """Periodic modulation via Brahim sequence."""
        brahim_index = period % len(BRAHIM_SEQUENCE)
        brahim_factor = BRAHIM_SEQUENCE[brahim_index] / BRAHIM_CENTER
        correction = 1.0 + (brahim_factor - 1.0) * 0.1
        return base_forecast * correction

    def _apply_lucas_smoothing(self, values: List[float]) -> float:
        """Lucas-weighted moving average."""
        n = min(len(values), len(LUCAS_NUMBERS))
        if n == 0:
            return 0.0
        weights = LUCAS_NUMBERS[:n]
        total_weight = sum(weights)
        weighted_sum = sum(v * w for v, w in zip(values[-n:], weights))
        return weighted_sum / total_weight

    def _determine_trend(self, saturation: float, growth: float) -> DemandTrend:
        phi_threshold = 1 / PHI
        if saturation > phi_threshold:
            return DemandTrend.SATURATING
        elif growth > 0.05:
            return DemandTrend.GROWING
        elif growth < -0.05:
            return DemandTrend.DECLINING
        else:
            return DemandTrend.STABLE

    def forecast(self, periods: int) -> List[DemandForecast]:
        """Generate *periods* forecast steps."""
        self._forecasts = []
        if self.historical_data:
            last_period = max(dp.period for dp in self.historical_data)
            recent_demands = [
                dp.demand
                for dp in sorted(self.historical_data, key=lambda x: x.period)
            ][-5:]
            current_demand = self._apply_lucas_smoothing(recent_demands)
        else:
            last_period = 0
            current_demand = self.max_demand * 0.1

        midpoint = last_period + periods / PHI

        for i in range(1, periods + 1):
            t = last_period + i
            base_forecast = self._phi_saturation_curve(t, midpoint)
            adjusted_forecast = self._apply_brahim_correction(base_forecast, t)
            adjusted_forecast = min(adjusted_forecast, self._phi_ceiling)
            saturation = self._calculate_saturation_level(adjusted_forecast)
            confidence = 1.0 / (1.0 + 0.1 * i / PHI)
            if i == 1:
                growth = (adjusted_forecast - current_demand) / max(current_demand, 1)
            else:
                prev = self._forecasts[-1].predicted_demand
                growth = (adjusted_forecast - prev) / max(prev, 1)
            trend = self._determine_trend(saturation, growth)
            self._forecasts.append(
                DemandForecast(
                    period=t,
                    predicted_demand=adjusted_forecast,
                    saturation_level=saturation,
                    confidence=confidence,
                    trend=trend,
                )
            )
        return self._forecasts

    def get_saturation_analysis(self) -> Dict:
        """Detailed saturation metrics."""
        if not self._forecasts:
            self.forecast(12)
        final_forecast = self._forecasts[-1]
        phi_saturation = 1 / PHI
        periods_to_phi_saturation = None
        for f in self._forecasts:
            if f.saturation_level >= phi_saturation:
                periods_to_phi_saturation = f.period
                break
        return {
            "max_demand": self.max_demand,
            "phi_ceiling": self._phi_ceiling,
            "final_predicted_demand": final_forecast.predicted_demand,
            "final_saturation_level": final_forecast.saturation_level,
            "periods_to_phi_saturation": periods_to_phi_saturation,
            "phi_saturation_threshold": phi_saturation,
            "forecast_periods": len(self._forecasts),
            "sum_constant_normalized": (
                final_forecast.predicted_demand / self.max_demand * BRAHIM_SUM
            ),
        }

    def normalize_to_214(self, forecasts: List[DemandForecast]) -> List[float]:
        """Normalise forecast values so their sum equals 214."""
        total = sum(f.predicted_demand for f in forecasts)
        if total == 0:
            return [0.0] * len(forecasts)
        scale = BRAHIM_SUM / total
        return [f.predicted_demand * scale for f in forecasts]
