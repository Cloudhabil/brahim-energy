"""Tests for brahim_energy.grid.optimizer."""

import math
from datetime import datetime

from brahim_energy.grid.optimizer import (
    BrahimSignalTiming,
    DemandResponsePhase,
    GridNode,
    GridSnapshot,
    GridStatus,
    GridStressCalculator,
    NodeType,
    OnionGridOptimizer,
)


def _make_node(node_id="n1", capacity=100.0, demand=50.0, **kw):
    return GridNode(
        node_id=node_id,
        node_type=kw.pop("node_type", NodeType.METER),
        capacity_kw=capacity,
        current_demand_kw=demand,
        **kw,
    )


class TestGridNode:
    def test_utilization(self):
        n = _make_node(capacity=100, demand=75)
        assert abs(n.utilization - 0.75) < 1e-10

    def test_headroom(self):
        n = _make_node(capacity=100, demand=60)
        assert abs(n.headroom_kw - 40.0) < 1e-10

    def test_overloaded(self):
        n = _make_node(capacity=100, demand=101)
        assert n.is_overloaded

    def test_not_overloaded(self):
        n = _make_node(capacity=100, demand=99)
        assert not n.is_overloaded

    def test_stress_contribution_positive(self):
        n = _make_node(capacity=100, demand=50)
        assert n.stress_contribution() > 0

    def test_stress_contribution_overloaded(self):
        n = _make_node(capacity=100, demand=200)
        assert math.isinf(n.stress_contribution(epsilon=0))

    def test_zero_capacity_utilization(self):
        n = _make_node(capacity=0, demand=0)
        assert n.utilization == 0.0


class TestGridStressCalculator:
    def test_empty_nodes(self):
        calc = GridStressCalculator()
        assert calc.compute_instantaneous_stress([]) == 0.0

    def test_low_stress(self):
        calc = GridStressCalculator()
        nodes = [_make_node(capacity=1000, demand=100)]
        stress = calc.compute_instantaneous_stress(nodes)
        assert stress < 0.001

    def test_high_stress(self):
        calc = GridStressCalculator()
        nodes = [_make_node(capacity=100, demand=99)]
        stress = calc.compute_instantaneous_stress(nodes)
        assert stress > 0.1

    def test_classify_optimal(self):
        calc = GridStressCalculator()
        assert calc.classify_status(0.0) == GridStatus.OPTIMAL

    def test_classify_critical(self):
        calc = GridStressCalculator()
        assert calc.classify_status(1.0) == GridStatus.CRITICAL

    def test_top_contributors(self):
        calc = GridStressCalculator()
        nodes = [
            _make_node("a", 100, 90),
            _make_node("b", 100, 50),
        ]
        top = calc.get_top_contributors(nodes, top_n=1)
        assert top[0][0] == "a"

    def test_gradient(self):
        calc = GridStressCalculator()
        nodes = [_make_node("a", 100, 50)]
        grad = calc.compute_gradient(nodes)
        assert "a" in grad
        assert grad["a"] > 0


class TestBrahimSignalTiming:
    def test_timing_constants(self):
        timing = BrahimSignalTiming()
        assert timing.CYCLE_SECONDS == 60
        assert timing.GREEN_SECONDS == 27
        # RED + AMBER = 33, total cycle = 60
        assert timing.GREEN_SECONDS + timing.RED_SECONDS + timing.AMBER_SECONDS == 60

    def test_get_current_phase(self):
        timing = BrahimSignalTiming()
        phase, remaining = timing.get_current_phase()
        assert isinstance(phase, DemandResponsePhase)
        assert remaining >= 0


class TestOnionGridOptimizer:
    def test_register_and_get(self):
        opt = OnionGridOptimizer()
        node = _make_node("x", 100, 50)
        opt.register_node(node)
        assert opt.get_node("x") is node

    def test_analyze_empty(self):
        opt = OnionGridOptimizer()
        snap = opt.analyze()
        assert snap.stress == 0
        assert snap.status == GridStatus.OPTIMAL

    def test_analyze_with_nodes(self):
        opt = OnionGridOptimizer()
        opt.register_node(_make_node("a", 1000, 100))
        opt.register_node(_make_node("b", 1000, 200))
        snap = opt.analyze()
        assert snap.total_capacity_kw == 2000
        assert snap.total_demand_kw == 300

    def test_update_node(self):
        opt = OnionGridOptimizer()
        opt.register_node(_make_node("a", 100, 50))
        opt.update_node("a", current_demand_kw=80)
        assert opt.get_node("a").current_demand_kw == 80

    def test_load_shift(self):
        opt = OnionGridOptimizer()
        opt.register_node(_make_node("a", 100, 80, controllable=True, priority=8))
        shifts = opt.compute_optimal_load_shift(20)
        assert len(shifts) >= 1
        assert shifts[0][0] == "a"

    def test_beta_target_reduction(self):
        opt = OnionGridOptimizer()
        opt.register_node(_make_node("a", 100, 100))
        reduction = opt.compute_beta_target_reduction()
        assert abs(reduction - 100 * 0.236) < 0.1
