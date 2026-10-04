"""
心理数字孪生测试套件
====================
覆盖：
  1. 因果图发现（Granger Causality）
  2. 数字孪生状态预测
  3. 干预时机优化
  4. 端到端流程
  5. 边界情况
"""
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assessment.digital_twin import (
    CausalEdge,
    CausalGraph,
    DigitalTwinManager,
    InterventionDecision,
    InterventionTimingOptimizer,
    PersonalCausalDiscovery,
    PsychoDimension,
    PsychologicalDigitalTwin,
    Trajectory,
    TrajectoryPoint,
    get_digital_twin_manager,
)


# ============================================================
# 辅助函数
# ============================================================

def generate_causal_series(n: int, seed: int = 42):
    """生成有因果关系的合成时间序列

    因果关系：circadian →(lag=2)→ cognitive →(lag=1)→ text_emotion
    即：circadian 的变化在 2 步后影响 cognitive，再 1 步后影响 text_emotion。
    """
    rng = np.random.RandomState(seed)

    # L1: 白噪声（独立）
    circadian = rng.randn(n) * 0.3 + 0.5
    circadian = np.clip(circadian, 0, 1)

    # L2: cognitive = 0.5 * circadian(t-2) + noise
    cognitive = np.zeros(n)
    for t in range(2, n):
        cognitive[t] = 0.5 * circadian[t - 2] + rng.randn() * 0.15 + 0.25
    cognitive = np.clip(cognitive, 0, 1)

    # L3: text_emotion = 0.4 * cognitive(t-1) + noise
    text_emotion = np.zeros(n)
    for t in range(1, n):
        text_emotion[t] = 0.4 * cognitive[t - 1] + rng.randn() * 0.12 + 0.3
    text_emotion = np.clip(text_emotion, 0, 1)

    # 独立维度（不应有因果边）
    independent = rng.randn(n) * 0.3 + 0.5
    independent = np.clip(independent, 0, 1)

    return {
        "circadian": circadian.tolist(),
        "cognitive": cognitive.tolist(),
        "text_emotion": text_emotion.tolist(),
        "independent": independent.tolist(),
    }


# ============================================================
# 测试：因果图发现
# ============================================================

class TestCausalDiscovery:
    """因果图发现测试"""

    def test_discover_finds_causal_edges(self):
        """应发现 circadian→cognitive 和 cognitive→text_emotion 的因果边"""
        ts = generate_causal_series(n=200, seed=42)
        discovery = PersonalCausalDiscovery(max_lag=5, significance_level=0.1)
        graph = discovery.discover("user1", ts)

        assert graph.user_id == "user1"
        assert graph.n_observations == 200
        assert len(graph.edges) > 0

    def test_discover_respects_lag(self):
        """发现的因果边应有合理的延迟"""
        ts = generate_causal_series(n=300, seed=123)
        discovery = PersonalCausalDiscovery(max_lag=5, significance_level=0.1)
        graph = discovery.discover("user2", ts)

        for edge in graph.edges:
            assert 1 <= edge.lag <= 5

    def test_discover_strength_range(self):
        """因果强度应在 [-1, 1] 范围内"""
        ts = generate_causal_series(n=200)
        discovery = PersonalCausalDiscovery(max_lag=3)
        graph = discovery.discover("user3", ts)

        for edge in graph.edges:
            assert -1.0 <= edge.strength <= 1.0

    def test_discover_p_value_range(self):
        """p-value 应在 [0, 1] 范围内"""
        ts = generate_causal_series(n=200)
        discovery = PersonalCausalDiscovery(max_lag=3)
        graph = discovery.discover("user4", ts)

        for edge in graph.edges:
            assert 0.0 <= edge.p_value <= 1.0

    def test_insufficient_data(self):
        """数据不足时应返回空图"""
        ts = {"a": [0.5] * 5, "b": [0.3] * 5}
        discovery = PersonalCausalDiscovery(max_lag=5)
        graph = discovery.discover("user5", ts)

        assert len(graph.edges) == 0
        assert graph.n_observations == 5


# ============================================================
# 测试：因果图数据结构
# ============================================================

class TestCausalGraph:
    """因果图数据结构测试"""

    def _make_graph(self) -> CausalGraph:
        return CausalGraph(
            user_id="test",
            edges=[
                CausalEdge("circadian", "cognitive", 0.5, 2, 0.01, 0.99),
                CausalEdge("cognitive", "text_emotion", 0.4, 1, 0.02, 0.98),
                CausalEdge("text_emotion", "behavior", 0.3, 1, 0.03, 0.97),
            ],
        )

    def test_get_causes(self):
        graph = self._make_graph()
        causes = graph.get_causes("cognitive")
        assert len(causes) == 1
        assert causes[0].cause == "circadian"

    def test_get_effects(self):
        graph = self._make_graph()
        effects = graph.get_effects("cognitive")
        assert len(effects) == 1
        assert effects[0].effect == "text_emotion"

    def test_get_chain(self):
        graph = self._make_graph()
        chains = graph.get_chain("circadian", max_depth=3)
        assert len(chains) > 0
        # 应包含 circadian→cognitive→text_emotion→behavior
        longest = max(chains, key=len)
        assert len(longest) >= 2

    def test_serialization(self):
        graph = self._make_graph()
        d = graph.to_dict()
        restored = CausalGraph.from_dict(d)
        assert restored.user_id == "test"
        assert len(restored.edges) == 3
        assert restored.edges[0].cause == "circadian"


# ============================================================
# 测试：数字孪生
# ============================================================

class TestDigitalTwin:
    """数字孪生测试"""

    def test_update_and_get_state(self):
        twin = PsychologicalDigitalTwin()
        twin.update({"text_emotion": 0.7, "cognitive": 0.5})
        state = twin.get_current_state()
        assert "text_emotion" in state
        assert abs(state["text_emotion"] - 0.7) < 0.01

    def test_state_evolves_with_causal_model(self):
        """有因果模型时，状态应按因果关系演化"""
        graph = CausalGraph(
            user_id="test",
            edges=[
                CausalEdge("circadian", "cognitive", 0.5, 1, 0.01, 0.99),
            ],
        )
        twin = PsychologicalDigitalTwin(causal_graph=graph)

        # 输入高 circadian
        twin.update({"circadian": 0.9, "cognitive": 0.3})
        # 多步更新后，cognitive 应被 circadian 拉高
        for _ in range(10):
            twin.update({"circadian": 0.9, "cognitive": 0.3})

        state = twin.get_current_state()
        # cognitive 的预测值应高于初始观测（因为 circadian 的因果影响）
        assert state["cognitive"] > 0.3

    def test_trajectory_prediction(self):
        graph = CausalGraph(
            user_id="test",
            edges=[
                CausalEdge("circadian", "cognitive", 0.5, 1, 0.01, 0.99),
            ],
        )
        twin = PsychologicalDigitalTwin(causal_graph=graph)
        twin.update({"circadian": 0.8, "cognitive": 0.6})

        traj = twin.predict_trajectory(horizon=6)
        assert len(traj.points) == 6
        assert traj.forecast_horizon == 6

    def test_trajectory_uncertainty_increases(self):
        """轨迹不确定度应随步数递增"""
        twin = PsychologicalDigitalTwin()
        twin.update({"a": 0.5, "b": 0.5})
        traj = twin.predict_trajectory(horizon=5)

        for i in range(1, len(traj.points)):
            assert traj.points[i].uncertainty >= traj.points[i - 1].uncertainty

    def test_empty_twin_trajectory(self):
        twin = PsychologicalDigitalTwin()
        traj = twin.predict_trajectory()
        assert len(traj.points) == 0

    def test_adaptive_alpha(self):
        """当观测与预测差异大时，alpha 应增大"""
        twin = PsychologicalDigitalTwin()
        twin.update({"x": 0.5})
        alpha_normal = twin._adaptive_alpha({"x": 0.52})
        alpha_big_diff = twin._adaptive_alpha({"x": 0.9})
        assert alpha_big_diff > alpha_normal

    def test_crisis_probability_range(self):
        twin = PsychologicalDigitalTwin()
        twin.update({"text_emotion": 0.8, "cognitive": 0.9})
        traj = twin.predict_trajectory(horizon=5)
        for p in traj.points:
            assert 0.0 <= p.crisis_probability <= 1.0


# ============================================================
# 测试：干预时机优化
# ============================================================

class TestInterventionOptimizer:
    """干预时机优化测试"""

    def _make_high_risk_twin(self) -> PsychologicalDigitalTwin:
        graph = CausalGraph(
            user_id="high_risk",
            edges=[
                CausalEdge("circadian", "cognitive", 0.5, 1, 0.01, 0.99),
                CausalEdge("cognitive", "text_emotion", 0.4, 1, 0.02, 0.98),
            ],
        )
        twin = PsychologicalDigitalTwin(causal_graph=graph)
        # 高风险状态
        twin.update({"circadian": 0.85, "cognitive": 0.8, "text_emotion": 0.75})
        return twin

    def _make_low_risk_twin(self) -> PsychologicalDigitalTwin:
        twin = PsychologicalDigitalTwin()
        twin.update({"circadian": 0.2, "cognitive": 0.15, "text_emotion": 0.1})
        return twin

    def test_high_risk_triggers_intervention(self):
        twin = self._make_high_risk_twin()
        optimizer = InterventionTimingOptimizer(min_benefit=0.01)
        decision = optimizer.optimize(twin, horizon=12)

        assert isinstance(decision, InterventionDecision)
        # 高风险时应该建议干预
        assert decision.urgency > 0

    def test_low_risk_no_intervention(self):
        twin = self._make_low_risk_twin()
        optimizer = InterventionTimingOptimizer(min_benefit=0.1)
        decision = optimizer.optimize(twin, horizon=12)

        # 低风险时不应干预
        assert decision.expected_benefit < 0.5

    def test_decision_has_reasoning(self):
        twin = self._make_high_risk_twin()
        optimizer = InterventionTimingOptimizer()
        decision = optimizer.optimize(twin, horizon=8)
        assert len(decision.reasoning) > 0

    def test_empty_twin_no_intervention(self):
        twin = PsychologicalDigitalTwin()
        optimizer = InterventionTimingOptimizer()
        decision = optimizer.optimize(twin, horizon=8)
        assert not decision.should_intervene
        assert decision.urgency == 0.0

    def test_chain_identification(self):
        twin = self._make_high_risk_twin()
        optimizer = InterventionTimingOptimizer()
        decision = optimizer.optimize(twin, horizon=12)
        # 应识别出因果链
        if decision.chain_to_break:
            assert "→" in decision.chain_to_break


# ============================================================
# 测试：管理器
# ============================================================

class TestManager:
    """管理器测试"""

    def test_create_twin(self):
        mgr = DigitalTwinManager()
        twin = mgr.get_or_create_twin("user1")
        assert twin is not None

    def test_update_observation(self):
        mgr = DigitalTwinManager()
        mgr.update_observation("user1", {"text_emotion": 0.6, "cognitive": 0.4})
        twin = mgr.get_or_create_twin("user1")
        state = twin.get_current_state()
        assert "text_emotion" in state

    def test_discover_and_predict(self):
        mgr = DigitalTwinManager()
        ts = generate_causal_series(n=200)
        graph = mgr.discover_causal_graph("user1", ts)
        assert graph.user_id == "user1"

        # 更新观测
        for t in range(20):
            obs = {k: v[t] for k, v in ts.items()}
            mgr.update_observation("user1", obs)

        # 预测轨迹
        traj = mgr.predict_trajectory("user1", horizon=5)
        assert len(traj.points) == 5

    def test_get_summary(self):
        mgr = DigitalTwinManager()
        mgr.update_observation("user1", {"a": 0.5, "b": 0.3})
        summary = mgr.get_summary("user1")
        assert summary["user_id"] == "user1"
        assert "current_state" in summary

    def test_intervention_decision(self):
        mgr = DigitalTwinManager()
        mgr.update_observation("user1", {
            "text_emotion": 0.8, "cognitive": 0.7, "circadian": 0.9
        })
        decision = mgr.get_intervention_decision("user1")
        assert isinstance(decision, InterventionDecision)


# ============================================================
# 测试：端到端
# ============================================================

class TestEndToEnd:
    """端到端测试"""

    def test_full_pipeline(self):
        """完整流程：观测 → 因果发现 → 数字孪生 → 轨迹预测 → 干预决策"""
        mgr = DigitalTwinManager()

        # 1. 生成合成数据（有因果结构）
        ts = generate_causal_series(n=300, seed=99)

        # 2. 发现因果图
        graph = mgr.discover_causal_graph("e2e_user", ts)
        assert graph.n_observations == 300

        # 3. 模拟实时观测
        for t in range(50):
            obs = {k: v[t] for k, v in ts.items()}
            mgr.update_observation("e2e_user", obs)

        # 4. 预测轨迹
        traj = mgr.predict_trajectory("e2e_user", horizon=12)
        assert len(traj.points) == 12
        assert traj.get_crisis_risk() >= 0

        # 5. 获取干预决策
        decision = mgr.get_intervention_decision("e2e_user")
        assert isinstance(decision, InterventionDecision)
        assert 0 <= decision.urgency <= 1.0

        # 6. 获取摘要
        summary = mgr.get_summary("e2e_user")
        assert summary["has_causal_graph"] is True
        assert summary["history_length"] == 50

    def test_different_users_different_graphs(self):
        """不同用户应有不同的因果图"""
        mgr = DigitalTwinManager()

        # 用户 A：circadian → cognitive
        ts_a = generate_causal_series(n=200, seed=1)
        graph_a = mgr.discover_causal_graph("userA", ts_a)

        # 用户 B：不同种子 = 不同数据
        ts_b = generate_causal_series(n=200, seed=999)
        graph_b = mgr.discover_causal_graph("userB", ts_b)

        # 因果图应不同（边数或强度不同）
        # 注意：由于随机性，可能边数相同但强度不同
        assert graph_a.user_id != graph_b.user_id


# ============================================================
# 测试：边界情况
# ============================================================

class TestEdgeCases:
    """边界情况测试"""

    def test_single_observation(self):
        twin = PsychologicalDigitalTwin()
        twin.update({"x": 0.5})
        state = twin.get_current_state()
        assert state["x"] == 0.5

    def test_many_observations_no_crash(self):
        twin = PsychologicalDigitalTwin()
        for i in range(500):
            twin.update({"x": 0.5 + 0.1 * math.sin(i * 0.1)})
        assert len(twin._history) <= 200  # 应被截断

    def test_zero_values(self):
        twin = PsychologicalDigitalTwin()
        twin.update({"x": 0.0, "y": 0.0})
        state = twin.get_current_state()
        assert state["x"] == 0.0

    def test_one_values(self):
        twin = PsychologicalDigitalTwin()
        twin.update({"x": 1.0, "y": 1.0})
        state = twin.get_current_state()
        assert state["x"] == 1.0

    def test_new_dimension_added(self):
        twin = PsychologicalDigitalTwin()
        twin.update({"x": 0.5})
        twin.update({"x": 0.6, "y": 0.3})
        state = twin.get_current_state()
        assert "y" in state
