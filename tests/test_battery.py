"""Tests for brahim_energy.battery.optimizer."""

from brahim_energy.battery.optimizer import (
    BATTERY_CHEMISTRIES,
    MATERIALS_DB,
    ApplicationScale,
    BrahimBatteryCalculator,
    DurationClass,
    StorageCategory,
)


class TestBatteryChemistries:
    def test_has_entries(self):
        assert len(BATTERY_CHEMISTRIES) >= 10

    def test_lfp_exists(self):
        assert "lfp" in BATTERY_CHEMISTRIES

    def test_all_have_cycles(self):
        for code, chem in BATTERY_CHEMISTRIES.items():
            assert chem.cycle_life > 0, f"{code} cycle_life=0"


class TestMaterialsDB:
    def test_has_entries(self):
        assert len(MATERIALS_DB) >= 8

    def test_all_have_supply_concentration(self):
        for name, mat in MATERIALS_DB.items():
            assert 0 <= mat.supply_concentration <= 1.0, f"{name}"


class TestBrahimBatteryCalculator:
    def test_init(self):
        bc = BrahimBatteryCalculator()
        assert len(bc.chemistries) > 0

    def test_calculate_battery_stress(self):
        bc = BrahimBatteryCalculator()
        result = bc.calculate_battery_stress("lfp", 1000, 0.8, 25.0)
        assert result.stress >= 0
        assert result.chemistry is not None

    def test_find_optimal_residential(self):
        bc = BrahimBatteryCalculator()
        results = bc.find_optimal_battery(
            ApplicationScale.RESIDENTIAL, DurationClass.SHORT,
        )
        assert len(results) > 0

    def test_find_optimal_utility(self):
        bc = BrahimBatteryCalculator()
        results = bc.find_optimal_battery(
            ApplicationScale.UTILITY, DurationClass.LONG,
        )
        assert len(results) > 0

    def test_find_optimal_grid(self):
        bc = BrahimBatteryCalculator()
        results = bc.find_optimal_battery(
            ApplicationScale.GRID, DurationClass.MEDIUM,
        )
        assert len(results) > 0

    def test_find_optimal_with_budget(self):
        bc = BrahimBatteryCalculator()
        results = bc.find_optimal_battery(
            ApplicationScale.GRID, DurationClass.MEDIUM, budget_per_kwh=200,
        )
        for r in results:
            assert r.chemistry.cost_per_kwh <= 200

    def test_material_integrity(self):
        bc = BrahimBatteryCalculator()
        score = bc.calculate_material_integrity("lithium")
        assert 0 <= score.overall <= 1.0

    def test_chemistry_integrity(self):
        bc = BrahimBatteryCalculator()
        score = bc.calculate_chemistry_integrity("lfp")
        assert 0 <= score <= 1.0

    def test_grid_resonance(self):
        bc = BrahimBatteryCalculator()
        resonance = bc.calculate_grid_resonance("lfp")
        assert isinstance(resonance, float)
        assert resonance >= 0
