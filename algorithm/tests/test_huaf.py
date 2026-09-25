"""
HUAF-TC 算法单元测试

覆盖场景：
1. 空输入降级
2. 单模态直通
3. 多模态融合 — 概率归一化、不确定性计算
4. 模态缺失时的层降级
5. 年龄自适应门控差异
6. 时序一致性平滑
7. risk_score_to_probs 转换正确性
8. 全 11 模态端到端融合
"""
import math
import sys
import os

import numpy as np
import pytest

# 确保 algorithm/ 在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.fusion.hierarchical_fusion import (
    HUAFConfig,
    HierarchicalFusionEngine,
    HierarchicalFusionResult,
    ModalityResult,
    risk_score_to_probs,
)


# ============================================================
# 辅助工具
# ============================================================

def _make_text_mod(probs=None, confidence=0.8, risk=0.3):
    """构造一个文本模态结果"""
    if probs is None:
        probs = [0.1, 0.15, 0.5, 0.05, 0.2]
    return ModalityResult(
        name="text", probs=probs, risk_score=risk,
        confidence=confidence, evidence=["文本分析完成"],
    )


def _make_mod(name, risk=0.3, confidence=0.7):
    """构造一个仅含 risk_score 的模态结果"""
    probs = risk_score_to_probs(risk)
    return ModalityResult(
        name=name, probs=probs, risk_score=risk,
        confidence=confidence, evidence=[f"{name}分析完成"],
    )


# ============================================================
# 测试用例
# ============================================================

class TestEmptyInput:
    """空输入应返回均匀分布降级"""

    def test_empty_returns_uniform(self):
        engine = HierarchicalFusionEngine()
        result = engine.fuse([])
        assert len(result.fused_probs) == 5
        assert all(abs(p - 0.2) < 0.01 for p in result.fused_probs)
        assert result.confidence == 0.0

    def test_empty_has_evidence(self):
        engine = HierarchicalFusionEngine()
        result = engine.fuse([])
        assert any("无可用模态" in e for e in result.evidence)


class TestSingleModality:
    """单模态应直通，不做额外融合"""

    def test_single_text_passthrough(self):
        engine = HierarchicalFusionEngine()
        mod = _make_text_mod(probs=[0.1, 0.1, 0.6, 0.05, 0.15])
        result = engine.fuse([mod])
        # 单模态时 L1 层应直接输出该模态的概率
        assert result.layer_results[0].modality_count == 1
        assert len(result.fused_probs) == 5
        assert sum(result.fused_probs) == pytest.approx(1.0, abs=1e-6)

    def test_single_modality_confidence(self):
        engine = HierarchicalFusionEngine()
        mod = _make_text_mod(confidence=0.9)
        result = engine.fuse([mod])
        assert result.confidence > 0


class TestMultiModality:
    """多模态融合核心测试"""

    def test_probs_normalized(self):
        """融合后概率必须归一化"""
        engine = HierarchicalFusionEngine()
        mods = [
            _make_text_mod(probs=[0.5, 0.1, 0.2, 0.1, 0.1]),
            _make_mod("voice", risk=0.6, confidence=0.7),
            _make_mod("face", risk=0.2, confidence=0.8),
        ]
        result = engine.fuse(mods)
        assert sum(result.fused_probs) == pytest.approx(1.0, abs=1e-6)

    def test_high_confidence_modality_dominates(self):
        """高置信度模态应获得更大权重"""
        engine = HierarchicalFusionEngine()
        # text: 高置信度、低熵（尖锐分布）
        text = _make_text_mod(
            probs=[0.02, 0.03, 0.90, 0.02, 0.03], confidence=0.95
        )
        # voice: 低置信度、高熵（接近均匀分布）
        voice = ModalityResult(
            name="voice",
            probs=[0.21, 0.19, 0.22, 0.19, 0.19],
            risk_score=0.5, confidence=0.3,
        )
        result = engine.fuse([text, voice])
        # 融合结果应更接近 text 的分布（高焦虑）
        assert result.fused_probs[2] > 0.5  # 焦虑维度应占主导

    def test_all_modalities_low_confidence(self):
        """所有模态低置信度时，输出应接近均匀"""
        engine = HierarchicalFusionEngine()
        mods = [
            ModalityResult(name="text", probs=[0.2]*5, confidence=0.1),
            ModalityResult(name="voice", probs=[0.2]*5, confidence=0.1),
        ]
        result = engine.fuse(mods)
        # 低置信度 → 高不确定度 → 融合结果接近均匀
        for p in result.fused_probs:
            assert 0.05 < p < 0.5  # 不会极端偏离

    def test_11_modality_full_fusion(self):
        """完整 11 模态端到端融合"""
        engine = HierarchicalFusionEngine()
        mods = [
            _make_text_mod(probs=[0.05, 0.15, 0.55, 0.05, 0.20]),
            _make_mod("voice", risk=0.5, confidence=0.7),
            _make_mod("face", risk=0.3, confidence=0.8),
            _make_mod("circadian", risk=0.4, confidence=0.6),
            _make_mod("cognitive", risk=0.6, confidence=0.7),
            _make_mod("behavior", risk=0.35, confidence=0.65),
            _make_mod("hrv", risk=0.45, confidence=0.6),
            _make_mod("breathing", risk=0.3, confidence=0.7),
            _make_mod("behavioral_act", risk=0.4, confidence=0.5),
            _make_mod("eye", risk=0.2, confidence=0.6),
            _make_mod("voice_semantics", risk=0.5, confidence=0.65),
        ]
        result = engine.fuse(mods, age_group="mid_adolescent")

        assert len(result.fused_probs) == 5
        assert sum(result.fused_probs) == pytest.approx(1.0, abs=1e-6)
        assert 0 < result.confidence <= 1.0
        assert result.uncertainty >= 0
        assert len(result.layer_results) == 3
        assert len(result.gating_weights) == 3
        assert sum(result.gating_weights) == pytest.approx(1.0, abs=1e-3)
        # 三层都应有模态
        for lr in result.layer_results:
            assert lr.modality_count > 0


class TestAgeAdaptiveGating:
    """年龄自适应门控测试"""

    def test_different_age_groups_different_weights(self):
        """不同年龄段应产生不同的门控权重"""
        engine = HierarchicalFusionEngine()
        mods = [
            _make_text_mod(),
            _make_mod("voice", risk=0.4),
            _make_mod("face", risk=0.3),
            _make_mod("cognitive", risk=0.5),
            _make_mod("behavior", risk=0.4),
        ]
        r_early = engine.fuse(mods, age_group="early_adolescent")
        # 重置时序状态
        engine2 = HierarchicalFusionEngine()
        r_late = engine2.fuse(mods, age_group="young_adult")

        # 初中生 L1 权重应高于大学生
        assert r_early.gating_weights[0] > r_late.gating_weights[0]
        # 大学生 L2 权重应高于初中生
        assert r_late.gating_weights[1] > r_early.gating_weights[1]

    def test_unknown_age_group_uses_default(self):
        """未知年龄段应使用默认先验"""
        engine = HierarchicalFusionEngine()
        mods = [_make_text_mod()]
        result = engine.fuse(mods, age_group="unknown_group")
        assert len(result.gating_weights) == 3
        assert sum(result.gating_weights) == pytest.approx(1.0, abs=1e-3)


class TestTemporalSmoothing:
    """时序一致性平滑测试"""

    def test_first_call_no_smoothing(self):
        """首次调用无历史，不应平滑"""
        engine = HierarchicalFusionEngine()
        result = engine.fuse([_make_text_mod()])
        assert result.temporal_alpha == 0.5  # 首次调用 alpha=0.5

    def test_stable_predictions_smoothed(self):
        """稳定预测应产生高 alpha（更信任历史）"""
        engine = HierarchicalFusionEngine()
        mod = _make_text_mod(probs=[0.1, 0.1, 0.6, 0.05, 0.15])
        # 连续调用 3 次相同输入
        for _ in range(3):
            result = engine.fuse([mod])
        # 稳定后 alpha 应 >= 0.5
        assert result.temporal_alpha >= 0.5

    def test_volatile_predictions_reduce_alpha(self):
        """剧烈波动的预测应降低 alpha（更信任当前）"""
        engine = HierarchicalFusionEngine()
        # 第一次：高焦虑
        mod1 = _make_text_mod(probs=[0.02, 0.03, 0.90, 0.02, 0.03])
        engine.fuse([mod1])
        # 第二次：突然变快乐
        mod2 = _make_text_mod(probs=[0.80, 0.02, 0.02, 0.02, 0.14])
        result = engine.fuse([mod2])
        # 波动大 → alpha 应较低
        assert result.temporal_alpha < 0.8


class TestRiskScoreConversion:
    """risk_score_to_probs 转换测试"""

    def test_zero_risk(self):
        probs = risk_score_to_probs(0.0)
        assert len(probs) == 5
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        # 零风险 → 中性/快乐为主
        assert probs[4] > 0.3 or probs[0] > 0.1  # 中性或快乐

    def test_high_risk(self):
        probs = risk_score_to_probs(0.9)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        # 高风险 → 焦虑/悲伤为主
        assert probs[2] > probs[0]  # 焦虑 > 快乐

    def test_monotonic_anxiety(self):
        """风险越高，焦虑概率应越大"""
        p_low = risk_score_to_probs(0.2)
        p_high = risk_score_to_probs(0.8)
        assert p_high[2] > p_low[2]


class TestEntropyComputation:
    """熵计算正确性测试"""

    def test_uniform_max_entropy(self):
        """均匀分布应产生最大熵 = 1.0（归一化后）"""
        engine = HierarchicalFusionEngine()
        ent = engine._entropy(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        assert ent == pytest.approx(1.0, abs=1e-6)

    def test_sharp_min_entropy(self):
        """尖锐分布应产生低熵"""
        engine = HierarchicalFusionEngine()
        ent = engine._entropy(np.array([0.01, 0.01, 0.95, 0.01, 0.02]))
        assert ent < 0.3

    def test_entropy_non_negative(self):
        """熵应非负"""
        engine = HierarchicalFusionEngine()
        ent = engine._entropy(np.array([1.0, 0, 0, 0, 0]))
        assert ent >= 0


class TestEvidenceChain:
    """证据链完整性测试"""

    def test_evidence_contains_layer_info(self):
        """证据应包含各层信息"""
        engine = HierarchicalFusionEngine()
        mods = [
            _make_text_mod(),
            _make_mod("voice", risk=0.4),
            _make_mod("cognitive", risk=0.5),
        ]
        result = engine.fuse(mods, age_group="early_adolescent")
        evidence_str = " ".join(result.evidence)
        assert "基础感知层" in evidence_str
        assert "HUAF-TC" in evidence_str

    def test_evidence_contains_age_info(self):
        """传入 age_group 时证据应包含年龄信息"""
        engine = HierarchicalFusionEngine()
        result = engine.fuse([_make_text_mod()], age_group="young_adult")
        assert any("young_adult" in e for e in result.evidence)


class TestEdgeCases:
    """边界情况测试"""

    def test_zero_probability_in_distribution(self):
        """概率分布含 0 时不应崩溃"""
        engine = HierarchicalFusionEngine()
        mod = _make_text_mod(probs=[0.0, 0.0, 1.0, 0.0, 0.0])
        result = engine.fuse([mod])
        assert sum(result.fused_probs) == pytest.approx(1.0, abs=1e-6)

    def test_very_low_confidence(self):
        """极低置信度模态应被降权但不崩溃"""
        engine = HierarchicalFusionEngine()
        mod = ModalityResult(name="text", probs=[0.2]*5, confidence=0.01)
        result = engine.fuse([mod])
        assert sum(result.fused_probs) == pytest.approx(1.0, abs=1e-6)

    def test_many_calls_no_crash(self):
        """连续多次调用不应崩溃（时序窗口管理）"""
        engine = HierarchicalFusionEngine()
        for i in range(20):
            risk = (i % 10) / 10.0
            mod = _make_text_mod(
                probs=risk_score_to_probs(risk), confidence=0.7
            )
            result = engine.fuse([mod])
            assert sum(result.fused_probs) == pytest.approx(1.0, abs=1e-6)
