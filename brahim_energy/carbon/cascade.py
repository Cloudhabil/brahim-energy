"""
Photosynthesis Cascade & MOF Filter Analysis
=============================================

Standalone analysis engine for:

1. **cascade** — Photosynthesis efficiency cascade with sum-rule validation.
2. **mof_filter** — MOF material ranking for CO2 capture.
3. **full_system** — Combined cascade + MOF + quantum coherence model.

Returns plain dicts (no external base-class dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from brahim_energy.carbon.photosynthesis import (
    MOF_MATERIALS,
    NATURAL_STEPS,
    PHI_OPTIMAL_PORE_NM,
    co2_factor,
    mof_score,
    pore_selectivity,
    quantum_coherence_factor,
    temp_correction,
)
from brahim_energy.constants import D, x_from_D

# ===================================================================
# Lightweight result container (replaces phi_engine AnalysisResult)
# ===================================================================

@dataclass
class CarbonAnalysisResult:
    """Plain-dict-friendly result from cascade / MOF / full_system analysis."""
    success: bool
    data: Dict[str, Any]
    d_space_values: List[float]
    consistency_score: float
    hierarchy: List[Dict[str, Any]]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "d_space_values": self.d_space_values,
            "consistency_score": self.consistency_score,
            "hierarchy": self.hierarchy,
            "recommendations": self.recommendations,
        }


# ===================================================================
# Inline sum-rule validator (replaces phi_engine.analyzer)
# ===================================================================

def _validate_sum_rule(
    d_values: List[float],
    d_product: float,
    tolerance_ppm: float = 100,
) -> Dict[str, Any]:
    """Check ``Σ D_i ≈ D(Π x_i)``.

    Returns a dict with *valid*, *d_sum*, *d_product*, *deviation_ppm*.
    """
    d_sum = sum(d_values)
    if d_product == 0:
        dev_ppm = 0.0
    else:
        dev_ppm = abs(d_sum - d_product) / abs(d_product) * 1e6
    return {
        "valid": dev_ppm <= tolerance_ppm,
        "d_sum": d_sum,
        "d_product": d_product,
        "deviation_ppm": dev_ppm,
        "tolerance_ppm": tolerance_ppm,
    }


# ===================================================================
# Clamp helper
# ===================================================================

def _clamp_efficiency(eta: float) -> float:
    return max(1e-10, min(eta, 1.0))


# ===================================================================
# Cascade analyser
# ===================================================================

class PhotosynthesisCascadeAnalyzer:
    """Standalone photosynthesis / MOF / coherence analyser.

    Three modes:

    * ``analyze_cascade(steps, ...)`` — 7-step efficiency cascade.
    * ``analyze_mof(candidates, ...)`` — MOF material filter & ranking.
    * ``analyze_full_system(...)`` — combined system model.
    """

    # ------------------------------------------------------------------
    # cascade
    # ------------------------------------------------------------------

    def analyze_cascade(
        self,
        steps: Optional[List[Dict[str, Any]]] = None,
        temperature_c: float = 25.0,
        co2_ppm: float = 415.0,
        target_efficiency: float = 0.20,
    ) -> CarbonAnalysisResult:
        """Analyze a multi-step photosynthesis cascade in D-space."""
        if steps is None:
            steps = NATURAL_STEPS

        d_steps: List[Dict[str, Any]] = []
        product = 1.0
        for s in steps:
            eta = _clamp_efficiency(float(s.get("efficiency", 0.5)))
            d_val = D(eta)
            product *= eta
            d_steps.append({
                "name": s.get("name", "unnamed"),
                "efficiency": eta,
                "d_value": d_val,
                "catalyst": s.get("catalyst", ""),
            })

        d_overall = D(max(product, 1e-10))
        d_target = D(max(target_efficiency, 1e-10))
        temp_fac = temp_correction(temperature_c)
        co2_sat = co2_factor(co2_ppm)

        # Sum-rule
        d_values = [s["d_value"] for s in d_steps]
        d_sum = sum(d_values)
        sr = _validate_sum_rule(d_values, d_overall, tolerance_ppm=100)
        consistency = max(0.0, 1.0 - sr["deviation_ppm"] / 1e6)

        # Contributions
        for s in d_steps:
            s["contribution_pct"] = (s["d_value"] / d_sum * 100) if d_sum > 0 else 0

        bottleneck = max(d_steps, key=lambda s: s["d_value"])

        d_gap = d_target - d_overall
        improvements: List[str] = []
        if d_gap < 0:
            gap_abs = abs(d_gap)
            improvements.append(
                f"Reduce D({bottleneck['name']}) by {gap_abs:.3f} "
                f"(raise efficiency to "
                f"{x_from_D(bottleneck['d_value'] - gap_abs):.3f})"
            )

        recs = self._cascade_recommendations(
            d_steps, bottleneck, temp_fac, co2_sat,
            co2_ppm, target_efficiency, d_gap,
        )

        return CarbonAnalysisResult(
            success=True,
            data={
                "overall_efficiency": product,
                "d_overall": d_overall,
                "step_analysis": d_steps,
                "bottleneck": {
                    "step_name": bottleneck["name"],
                    "d_value": bottleneck["d_value"],
                    "efficiency": bottleneck["efficiency"],
                    "contribution_pct": bottleneck["contribution_pct"],
                    "catalyst": bottleneck.get("catalyst", ""),
                },
                "sum_rule": sr,
                "gap_to_target": {
                    "target": target_efficiency,
                    "d_target": d_target,
                    "d_gap": d_gap,
                    "improvements_needed": improvements,
                },
                "env_corrections": {
                    "temp_factor": temp_fac,
                    "co2_saturation": co2_sat,
                },
            },
            d_space_values=d_values,
            consistency_score=consistency,
            hierarchy=[
                {"step": s["name"], "d_value": s["d_value"], "rank": i + 1}
                for i, s in enumerate(
                    sorted(d_steps, key=lambda s: -s["d_value"])
                )
            ],
            recommendations=recs,
        )

    @staticmethod
    def _cascade_recommendations(
        steps: List[Dict],
        bottleneck: Dict,
        temp_fac: float,
        co2_sat: float,
        co2_ppm: float,
        target: float,
        d_gap: float,
    ) -> List[str]:
        recs: List[str] = []
        bn_name = bottleneck["name"]
        if bn_name == "carbon_fixation":
            recs.append(
                "Carbon fixation (RuBisCO) is the bottleneck. Consider "
                "engineered RuBisCO variants or C4/CAM mechanisms."
            )
        elif bn_name == "photorespiration":
            recs.append(
                "Photorespiration wastes energy. Encapsulate RuBisCO in "
                "carboxysomes to suppress this pathway."
            )
        elif bn_name == "water_splitting":
            recs.append(
                "Water splitting catalyst limits efficiency. Explore "
                "Co-Pi or Ir-oxide catalysts for artificial OEC."
            )
        else:
            recs.append(
                f"Step '{bn_name}' is the primary bottleneck "
                f"(D={bottleneck['d_value']:.3f}). Focus R&D here."
            )
        if d_gap < 0:
            recs.append(
                f"Target efficiency ({target:.1%}) requires D-reduction "
                f"of {abs(d_gap):.3f}."
            )
        if co2_sat < 0.8:
            recs.append(
                f"CO2 saturation is only {co2_sat:.1%} at {co2_ppm:.0f} ppm. "
                "Consider MOF pre-concentrator."
            )
        if temp_fac < 0.9:
            recs.append(
                f"Temperature factor {temp_fac:.3f} indicates suboptimal "
                "conditions. Target 20-30 C."
            )
        return recs

    # ------------------------------------------------------------------
    # mof_filter
    # ------------------------------------------------------------------

    def analyze_mof(
        self,
        candidates: Optional[List[str]] = None,
        abundant_only: bool = False,
        max_cost_relative: Optional[float] = None,
        min_selectivity: Optional[float] = None,
        self_healing: bool = False,
    ) -> CarbonAnalysisResult:
        """Rank MOF materials for CO2 capture."""
        if candidates is None:
            candidates = list(MOF_MATERIALS.keys())

        entries: List[Dict[str, Any]] = []
        for name in candidates:
            mat = MOF_MATERIALS.get(name)
            if mat is None:
                continue
            score = mof_score(mat)
            pore_phi = abs(mat["pore_nm"] - PHI_OPTIMAL_PORE_NM)
            geo_sel = pore_selectivity(mat["pore_nm"])
            entries.append({
                "name": name, **mat,
                "score": score,
                "d_score": D(max(score, 1e-10)),
                "pore_phi_match": pore_phi,
                "geometric_selectivity": geo_sel,
            })

        filtered = list(entries)
        if abundant_only:
            filtered = [c for c in filtered if c.get("abundant", False)]
        if max_cost_relative is not None:
            filtered = [
                c for c in filtered
                if c.get("cost_relative", 999) <= max_cost_relative
            ]
        if min_selectivity is not None:
            filtered = [
                c for c in filtered
                if c.get("co2_n2_selectivity", 0) >= min_selectivity
            ]
        if self_healing:
            filtered = [c for c in filtered if c.get("self_healing", False)]

        filtered.sort(key=lambda c: -c["score"])
        d_values = [c["d_score"] for c in filtered]
        consistency = 1.0 if filtered else 0.0

        recs = self._mof_recommendations(filtered, entries)

        return CarbonAnalysisResult(
            success=True,
            data={
                "mof_ranking": [
                    {
                        "name": c["name"],
                        "score": round(c["score"], 3),
                        "d_score": round(c["d_score"], 4),
                        "pore_phi_match": round(c["pore_phi_match"], 4),
                        "geometric_selectivity": round(c["geometric_selectivity"], 2),
                        "co2_capacity_mmol_g": c["co2_capacity_mmol_g"],
                        "co2_n2_selectivity": c["co2_n2_selectivity"],
                        "abundant": c.get("abundant", False),
                        "self_healing": c.get("self_healing", False),
                    }
                    for c in filtered
                ],
                "phi_optimal_pore_nm": round(PHI_OPTIMAL_PORE_NM, 4),
                "total_candidates": len(entries),
                "after_constraints": len(filtered),
            },
            d_space_values=d_values,
            consistency_score=consistency,
            hierarchy=[
                {"mof": c["name"], "score": c["score"], "rank": i + 1}
                for i, c in enumerate(filtered)
            ],
            recommendations=recs,
        )

    @staticmethod
    def _mof_recommendations(
        filtered: List[Dict], all_cands: List[Dict]
    ) -> List[str]:
        recs: List[str] = []
        if not filtered:
            recs.append(
                "No MOFs survive the applied constraints. "
                "Relax abundant_only or increase max_cost_relative."
            )
            return recs
        best = filtered[0]
        recs.append(
            f"Top candidate: {best['name']} "
            f"(score={best['score']:.1f}, "
            f"capacity={best['co2_capacity_mmol_g']} mmol/g, "
            f"selectivity={best['co2_n2_selectivity']}x)."
        )
        phi_close = [c for c in filtered if c["pore_phi_match"] < 0.05]
        if phi_close:
            names = ", ".join(c["name"] for c in phi_close)
            recs.append(
                f"PHI-optimal pore match (<0.05 nm): {names}. "
                f"Golden-ratio pore ({PHI_OPTIMAL_PORE_NM:.3f} nm)."
            )
        healers = [c for c in filtered if c.get("self_healing")]
        if healers:
            names = ", ".join(c["name"] for c in healers)
            recs.append(f"Self-healing MOFs: {names}.")
        if len(filtered) < len(all_cands):
            removed = len(all_cands) - len(filtered)
            recs.append(f"{removed} candidate(s) removed by constraints.")
        return recs

    # ------------------------------------------------------------------
    # full_system
    # ------------------------------------------------------------------

    def analyze_full_system(
        self,
        steps: Optional[List[Dict[str, Any]]] = None,
        mof: str = "Fe-BTC",
        coherence_coupling: float = 0.8,
        temperature_c: float = 25.0,
        co2_ppm: float = 415.0,
        solar_irradiance_w_m2: float = 1000.0,
        unit_area_m2: float = 1.0,
        target_efficiency: float = 0.20,
    ) -> CarbonAnalysisResult:
        """Combined cascade + MOF + coherence system model."""
        cascade_result = self.analyze_cascade(
            steps, temperature_c, co2_ppm, target_efficiency
        )
        cascade_eff: float = cascade_result.data["overall_efficiency"]

        temp_fac = temp_correction(temperature_c)
        co2_sat = co2_factor(co2_ppm)
        coherence = quantum_coherence_factor(temperature_c, coherence_coupling)

        mof_entry = MOF_MATERIALS.get(mof, {})
        mof_cap = mof_entry.get("co2_capacity_mmol_g", 0.0) if mof_entry else 0.0
        mof_eff = min(mof_cap / 10.0, 1.0)

        eta_system = max(
            cascade_eff * mof_eff * coherence * temp_fac * co2_sat, 1e-10
        )
        d_system = D(eta_system)

        sunshine_hours = 12.0
        energy_j = solar_irradiance_w_m2 * unit_area_m2 * sunshine_hours * 3600
        co2_mol_day = energy_j * eta_system / 2870e3 * 6.0
        co2_kg_day = co2_mol_day * 0.044

        subsystems = [
            {"name": "cascade", "efficiency": cascade_eff,
             "d_value": D(max(cascade_eff, 1e-10))},
            {"name": "mof_capture", "efficiency": mof_eff,
             "d_value": D(max(mof_eff, 1e-10))},
            {"name": "coherence", "efficiency": coherence,
             "d_value": D(max(coherence, 1e-10))},
            {"name": "temperature", "efficiency": temp_fac,
             "d_value": D(max(temp_fac, 1e-10))},
            {"name": "co2_saturation", "efficiency": co2_sat,
             "d_value": D(max(co2_sat, 1e-10))},
        ]
        system_bottleneck = max(
            subsystems, key=lambda s: float(s["d_value"])  # type: ignore[arg-type]
        )

        d_values: list[float] = [float(s["d_value"]) for s in subsystems]
        sr = _validate_sum_rule(d_values, d_system, tolerance_ppm=1000)
        consistency = max(0.0, 1.0 - sr["deviation_ppm"] / 1e6)

        recs = self._full_recommendations(
            subsystems, system_bottleneck, eta_system,
            co2_kg_day, mof, coherence_coupling, co2_ppm,
        )

        return CarbonAnalysisResult(
            success=True,
            data={
                "overall_efficiency": eta_system,
                "d_system": d_system,
                "co2_per_m2_day_kg": round(co2_kg_day, 6),
                "subsystems": [
                    {"name": s["name"],
                     "efficiency": round(float(s["efficiency"]), 6),
                     "d_value": round(float(s["d_value"]), 4)}
                    for s in subsystems
                ],
                "bottleneck": {
                    "subsystem": system_bottleneck["name"],
                    "d_value": system_bottleneck["d_value"],
                    "efficiency": system_bottleneck["efficiency"],
                },
                "mof": {
                    "name": mof,
                    "score": round(mof_score(mof_entry), 3) if mof_entry else 0.0,
                    "capture_efficiency": round(mof_eff, 4),
                },
                "coherence": {
                    "factor": round(coherence, 6),
                    "coupling": coherence_coupling,
                    "temp_stable": coherence > 0.5,
                },
                "cascade_detail": cascade_result.data,
                "sum_rule": sr,
            },
            d_space_values=d_values,
            consistency_score=consistency,
            hierarchy=[
                {"subsystem": s["name"], "d_value": s["d_value"], "rank": i + 1}
                for i, s in enumerate(
                    sorted(subsystems, key=lambda s: -float(s["d_value"]))
                )
            ],
            recommendations=recs,
        )

    @staticmethod
    def _full_recommendations(
        subsystems: List[Dict],
        bottleneck: Dict,
        eta_system: float,
        co2_kg: float,
        mof_name: str,
        coupling: float,
        co2_ppm: float,
    ) -> List[str]:
        recs: List[str] = []
        bn = bottleneck["name"]
        recs.append(
            f"System efficiency: {eta_system:.4%}. "
            f"CO2 capture: {co2_kg:.4f} kg/m2/day."
        )
        recs.append(
            f"Primary bottleneck: {bn} "
            f"(D={bottleneck['d_value']:.3f}, "
            f"eff={bottleneck['efficiency']:.3f})."
        )
        if bn == "cascade":
            recs.append(
                "Photosynthesis cascade limits system. "
                "See cascade analysis for per-step improvements."
            )
        elif bn == "mof_capture":
            recs.append(
                f"MOF '{mof_name}' capture rate is the bottleneck. "
                "Consider higher-capacity MOFs (MOF-74-Mg: 8.9 mmol/g)."
            )
        elif bn == "coherence":
            recs.append(
                f"Quantum coherence (coupling={coupling:.2f}) degrades "
                "efficiency. Increase coupling via structured scaffolds."
            )
        elif bn == "co2_saturation":
            recs.append(
                f"CO2 saturation at {co2_ppm:.0f} ppm is low. "
                "Use MOF pre-concentrator."
            )
        return recs
