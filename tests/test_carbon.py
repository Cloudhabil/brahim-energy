"""Tests for brahim_energy.carbon."""

import math

from brahim_energy.carbon.photosynthesis import (
    D_NATURAL_OVERALL,
    MOF_MATERIALS,
    NATURAL_OVERALL,
    NATURAL_STEPS,
    PHI_OPTIMAL_PORE_NM,
    co2_factor,
    mof_score,
    photon_energy_eV,
    pore_selectivity,
    quantum_coherence_factor,
    temp_correction,
)
from brahim_energy.carbon.cascade import (
    CarbonAnalysisResult,
    PhotosynthesisCascadeAnalyzer,
)


# ===================================================================
# photosynthesis constants
# ===================================================================

class TestPhotonEnergy:
    def test_680nm(self):
        e = photon_energy_eV(680)
        assert abs(e - 1239.842 / 680) < 1e-6

    def test_raises_zero(self):
        import pytest
        with pytest.raises(ValueError):
            photon_energy_eV(0)


class TestNaturalSteps:
    def test_7_steps(self):
        assert len(NATURAL_STEPS) == 7

    def test_overall_product(self):
        product = 1.0
        for s in NATURAL_STEPS:
            product *= s["efficiency"]
        assert abs(product - NATURAL_OVERALL) < 1e-10

    def test_d_natural_positive(self):
        assert D_NATURAL_OVERALL > 0


class TestMOFMaterials:
    def test_8_entries(self):
        assert len(MOF_MATERIALS) == 8

    def test_all_have_pore(self):
        for name, mat in MOF_MATERIALS.items():
            assert mat["pore_nm"] > 0, f"{name} pore=0"

    def test_phi_optimal_pore(self):
        assert 0.3 < PHI_OPTIMAL_PORE_NM < 0.4


class TestHelperFunctions:
    def test_temp_correction_optimal(self):
        f = temp_correction(25.0)
        assert abs(f - 1.0) < 1e-10

    def test_temp_correction_cold(self):
        assert temp_correction(0.0) < 1.0

    def test_co2_factor_ambient(self):
        f = co2_factor(415)
        assert 0.6 < f < 0.8

    def test_co2_factor_zero(self):
        assert co2_factor(0) == 0.0

    def test_co2_factor_negative(self):
        assert co2_factor(-100) == 0.0

    def test_mof_score_positive(self):
        for name, mat in MOF_MATERIALS.items():
            s = mof_score(mat)
            assert s > 0, f"{name} score={s}"

    def test_pore_selectivity_peak(self):
        peak = pore_selectivity(PHI_OPTIMAL_PORE_NM)
        off = pore_selectivity(PHI_OPTIMAL_PORE_NM + 0.1)
        assert peak > off

    def test_quantum_coherence_room_temp(self):
        f = quantum_coherence_factor(25.0, 1.0)
        assert 0.5 < f < 1.0

    def test_quantum_coherence_zero_coupling(self):
        assert quantum_coherence_factor(25.0, 0.0) == 0.0


# ===================================================================
# cascade analyser
# ===================================================================

class TestPhotosynthesisCascadeAnalyzer:
    def test_analyze_cascade_defaults(self):
        pca = PhotosynthesisCascadeAnalyzer()
        result = pca.analyze_cascade()
        assert result.success
        assert result.data["overall_efficiency"] > 0
        assert len(result.d_space_values) == 7
        assert result.consistency_score > 0.9

    def test_cascade_bottleneck(self):
        pca = PhotosynthesisCascadeAnalyzer()
        result = pca.analyze_cascade()
        bn = result.data["bottleneck"]
        assert bn["step_name"] == "carbon_fixation"

    def test_cascade_custom_steps(self):
        pca = PhotosynthesisCascadeAnalyzer()
        steps = [
            {"name": "step1", "efficiency": 0.9},
            {"name": "step2", "efficiency": 0.8},
        ]
        result = pca.analyze_cascade(steps=steps)
        assert result.success
        assert len(result.d_space_values) == 2

    def test_analyze_mof_defaults(self):
        pca = PhotosynthesisCascadeAnalyzer()
        result = pca.analyze_mof()
        assert result.success
        assert len(result.data["mof_ranking"]) == 8

    def test_analyze_mof_abundant_only(self):
        pca = PhotosynthesisCascadeAnalyzer()
        result = pca.analyze_mof(abundant_only=True)
        assert result.success
        for m in result.data["mof_ranking"]:
            assert m["abundant"] is True

    def test_analyze_mof_self_healing(self):
        pca = PhotosynthesisCascadeAnalyzer()
        result = pca.analyze_mof(self_healing=True)
        assert result.success
        for m in result.data["mof_ranking"]:
            assert m["self_healing"] is True

    def test_analyze_full_system(self):
        pca = PhotosynthesisCascadeAnalyzer()
        result = pca.analyze_full_system()
        assert result.success
        assert result.data["overall_efficiency"] > 0
        assert result.data["co2_per_m2_day_kg"] > 0
        assert "bottleneck" in result.data

    def test_full_system_custom_mof(self):
        pca = PhotosynthesisCascadeAnalyzer()
        result = pca.analyze_full_system(mof="MOF-74-Mg")
        assert result.success
        assert result.data["mof"]["name"] == "MOF-74-Mg"

    def test_result_to_dict(self):
        pca = PhotosynthesisCascadeAnalyzer()
        result = pca.analyze_cascade()
        d = result.to_dict()
        assert "success" in d
        assert "recommendations" in d
