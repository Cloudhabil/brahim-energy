"""Tests for brahim_energy.battery.materials."""

from brahim_energy.battery.materials import (
    AgentState,
    AgentTask,
    CostOptimizer,
    DegradationForecaster,
    IndustryStandard,
    IntegrityScorer,
    MaterialEngineAgent,
    MaterialPropertyPredictor,
    SafetyClassifier,
)


class TestMLModels:
    def test_property_predictor(self):
        model = MaterialPropertyPredictor()
        model.train(
            [{"composition": {"Li": 0.5, "Fe": 0.3, "P": 0.2}}],
            epochs=5,
        )
        pred = model.predict({"composition": {"Li": 0.5, "Fe": 0.3, "P": 0.2}})
        assert isinstance(pred, dict)
        assert "energy_density" in pred

    def test_degradation_forecaster(self):
        model = DegradationForecaster()
        pred = model.predict({"cycles": 1000, "temperature": 25, "dod": 0.8})
        assert isinstance(pred, dict)

    def test_safety_classifier(self):
        model = SafetyClassifier()
        pred = model.predict({"thermal_runaway_temp": 250})
        assert isinstance(pred, dict)

    def test_cost_optimizer(self):
        model = CostOptimizer()
        pred = model.predict({"materials": ["lithium", "iron"]})
        assert isinstance(pred, dict)

    def test_integrity_scorer(self):
        model = IntegrityScorer()
        pred = model.predict({"supply_risk": 0.3, "recyclability": 0.8})
        assert isinstance(pred, dict)


class TestMaterialEngineAgent:
    def test_init(self):
        agent = MaterialEngineAgent()
        assert agent.state == AgentState.IDLE

    def test_submit_task(self):
        agent = MaterialEngineAgent()
        task_id = agent.submit_task("test_task", {"material": "LFP"})
        assert isinstance(task_id, str)
        assert len(agent.task_queue) == 1

    def test_perceive(self):
        agent = MaterialEngineAgent()
        obs = agent.perceive({"material": "NMC"})
        assert isinstance(obs, dict)

    def test_reason(self):
        agent = MaterialEngineAgent()
        obs = agent.perceive({"material": "LFP"})
        plan = agent.reason(obs)
        assert isinstance(plan, dict)

    def test_discover_material(self):
        agent = MaterialEngineAgent()
        result = agent.discover_material(
            {"energy_density": 300, "safety": 0.9}
        )
        assert isinstance(result, dict)

    def test_industry_standards(self):
        assert len(IndustryStandard) >= 10
