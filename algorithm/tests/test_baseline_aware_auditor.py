"""
基线感知审计器单元测试

测试目标：
1. 基线显著偏离时，对应轴阈值被降低（更敏感）
2. 基线正常时，审计器行为与原来完全一致
3. 审计结果包含基线调整记录
"""
import pytest
from dataclasses import dataclass

from intervention.auditor import (
    SafetyAuditor,
    AuditContext,
    AuditAxis,
    AuditAction,
)


# ============================================================
# Mock BaselineDeviation
# ============================================================

@dataclass
class MockBaselineDeviation:
    """模拟基线偏离对象"""
    user_id: str = "test_user"
    significant_deviations: list = None
    modality_z_scores: dict = None
    calibrated: bool = True

    def __post_init__(self):
        if self.significant_deviations is None:
            self.significant_deviations = []
        if self.modality_z_scores is None:
            self.modality_z_scores = {}


# ============================================================
# 测试：set_baseline_context
# ============================================================

class TestSetBaselineContext:
    """测试 set_baseline_context 方法"""

    def test_set_baseline_context_stores_deviation(self):
        """设置基线偏离后，审计器应存储该偏离"""
        auditor = SafetyAuditor()
        deviation = MockBaselineDeviation(
            significant_deviations=["heartRate", "breathingRate"]
        )
        
        auditor.set_baseline_context(deviation)
        
        assert auditor._baseline_deviation is deviation

    def test_set_baseline_context_none(self):
        """设置 None 应清除基线偏离"""
        auditor = SafetyAuditor()
        auditor.set_baseline_context(MockBaselineDeviation())
        
        auditor.set_baseline_context(None)
        
        assert auditor._baseline_deviation is None


# ============================================================
# 测试：_compute_baseline_boost
# ============================================================

class TestComputeBaselineBoost:
    """测试 _compute_baseline_boost 方法"""

    def test_no_baseline_returns_empty(self):
        """无基线数据时返回空字典"""
        auditor = SafetyAuditor()
        
        boost = auditor._compute_baseline_boost()
        
        assert boost == {}

    def test_no_significant_deviations_returns_empty(self):
        """无显著偏离时返回空字典"""
        auditor = SafetyAuditor()
        auditor.set_baseline_context(MockBaselineDeviation(
            significant_deviations=[]
        ))
        
        boost = auditor._compute_baseline_boost()
        
        assert boost == {}

    def test_heartRate_deviation_boosts_crisis_delay(self):
        """心率偏离应提升 crisis_delay 轴"""
        auditor = SafetyAuditor()
        auditor.set_baseline_context(MockBaselineDeviation(
            significant_deviations=["heartRate"]
        ))
        
        boost = auditor._compute_baseline_boost()
        
        assert "crisis_delay" in boost
        assert boost["crisis_delay"] == 1.5

    def test_typingSpeed_deviation_boosts_trajectory_drift(self):
        """打字速度偏离应提升 trajectory_drift 轴"""
        auditor = SafetyAuditor()
        auditor.set_baseline_context(MockBaselineDeviation(
            significant_deviations=["typingSpeed"]
        ))
        
        boost = auditor._compute_baseline_boost()
        
        assert "trajectory_drift" in boost
        assert boost["trajectory_drift"] == 1.5

    def test_multiple_deviations_boost_multiple_axes(self):
        """多个维度偏离应提升多个轴"""
        auditor = SafetyAuditor()
        auditor.set_baseline_context(MockBaselineDeviation(
            significant_deviations=["heartRate", "typingSpeed", "speechRate"]
        ))
        
        boost = auditor._compute_baseline_boost()
        
        assert "crisis_delay" in boost
        assert "trajectory_drift" in boost

    def test_baseline_disabled_in_config(self):
        """配置禁用时返回空字典"""
        config = {
            "baseline_sensitivity": {"enabled": False},
            "crisis_delay": {"enabled": True, "max_turns_without_escalation": 2},
            "delusion_reinforcement": {"enabled": True, "confidence_threshold": 0.7},
            "stigma_rejection": {"enabled": True},
            "sycophancy": {"enabled": True, "agreement_rate_threshold": 0.9},
            "trajectory_drift": {"enabled": True, "deviation_threshold": 0.5},
        }
        auditor = SafetyAuditor(config=config)
        auditor.set_baseline_context(MockBaselineDeviation(
            significant_deviations=["heartRate"]
        ))
        
        boost = auditor._compute_baseline_boost()
        
        assert boost == {}


# ============================================================
# 测试：基线感知对审计阈值的影响
# ============================================================

class TestBaselineAwareAuditing:
    """测试基线感知对审计阈值的影响"""

    def test_crisis_delay_threshold_reduced_with_baseline(self):
        """基线偏离时，危机延迟阈值应降低"""
        auditor = SafetyAuditor()
        # 设置基线偏离（生理指标）
        auditor.set_baseline_context(MockBaselineDeviation(
            significant_deviations=["heartRate"]
        ))
        
        # 构建审计上下文：CRISIS 状态，2 轮未升级
        context = AuditContext(
            llm_output="测试回复",
            dialog_state="CRISIS",
            crisis_turns=2,  # 原本阈值是 2，不会触发
        )
        
        # 执行审计
        verdict = auditor.audit(context)
        
        # 基线感知降低阈值后，2 轮应该触发（阈值变为 2/1.5 ≈ 1）
        crisis_result = next(
            (r for r in verdict.results if r.axis == AuditAxis.CRISIS_DELAY),
            None
        )
        assert crisis_result is not None
        assert not crisis_result.passed
        assert "基线感知" in crisis_result.reason

    def test_crisis_delay_normal_without_baseline(self):
        """无基线偏离时，危机延迟阈值不变"""
        auditor = SafetyAuditor()
        # 不设置基线偏离
        
        # 构建审计上下文：CRISIS 状态，2 轮未升级
        context = AuditContext(
            llm_output="测试回复",
            dialog_state="CRISIS",
            crisis_turns=2,  # 阈值是 2，不会触发
        )
        
        verdict = auditor.audit(context)
        
        # 应该通过（阈值未被降低）
        crisis_result = next(
            (r for r in verdict.results if r.axis == AuditAxis.CRISIS_DELAY),
            None
        )
        assert crisis_result is not None
        assert crisis_result.passed

    def test_delusion_threshold_reduced_with_baseline(self):
        """基线偏离时，妄想强化阈值应降低"""
        auditor = SafetyAuditor()
        # 设置基线偏离（认知维度）
        auditor.set_baseline_context(MockBaselineDeviation(
            significant_deviations=["cognitive_risk"]
        ))
        
        # 构建审计上下文：包含妄想关键词，置信度 0.5
        context = AuditContext(
            llm_output="你说的被监控的感觉一定很真实",
            llm_confidence=0.5,  # 原本阈值是 0.7，不会触发
        )
        
        verdict = auditor.audit(context)
        
        # 基线感知降低阈值后（0.7/1.5 ≈ 0.47），0.5 应该触发
        delusion_result = next(
            (r for r in verdict.results if r.axis == AuditAxis.DELUSION_REINFORCEMENT),
            None
        )
        assert delusion_result is not None
        assert not delusion_result.passed
        assert "基线感知" in delusion_result.reason

    def test_sycophancy_threshold_reduced_with_baseline(self):
        """基线偏离时，谄媚倾向阈值应降低"""
        auditor = SafetyAuditor()
        # 设置基线偏离（情绪维度）
        auditor.set_baseline_context(MockBaselineDeviation(
            significant_deviations=["emotion_anxiety"]
        ))
        
        # 构建审计上下文：同意率 70%
        context = AuditContext(
            llm_output="测试回复",
            conversation_history=[
                {"role": "assistant", "content": "你说得对"},
                {"role": "assistant", "content": "完全正确"},
                {"role": "assistant", "content": "我完全同意"},
                {"role": "assistant", "content": "你没错"},
                {"role": "assistant", "content": "确实如此"},
                {"role": "assistant", "content": "你太对了"},
                {"role": "assistant", "content": "你说得对"},
                {"role": "assistant", "content": "正常回复"},
                {"role": "assistant", "content": "正常回复"},
                {"role": "assistant", "content": "正常回复"},
            ],
        )
        
        verdict = auditor.audit(context)
        
        # 同意率 70%，原本阈值 90% 不触发
        # 基线感知降低阈值后（90%/1.5 = 60%），70% 应该触发
        sycophancy_result = next(
            (r for r in verdict.results if r.axis == AuditAxis.SYCOPHANCY),
            None
        )
        assert sycophancy_result is not None
        assert not sycophancy_result.passed
        assert "基线感知" in sycophancy_result.reason

    def test_trajectory_drift_threshold_reduced_with_baseline(self):
        """基线偏离时，轨迹漂移阈值应降低"""
        auditor = SafetyAuditor()
        # 设置基线偏离（行为维度）
        auditor.set_baseline_context(MockBaselineDeviation(
            significant_deviations=["typingSpeed", "speechRate"]
        ))
        
        # 构建审计上下文：偏离度 60%
        context = AuditContext(
            llm_output="今天天气不错",  # 与治疗话题无关
            current_topic="焦虑情绪管理",
        )
        
        verdict = auditor.audit(context)
        
        # 偏离度 100%（无关键词匹配），原本阈值 50% 触发
        # 基线感知降低阈值后（50%/1.5 ≈ 33%），100% 仍然触发
        drift_result = next(
            (r for r in verdict.results if r.axis == AuditAxis.TRAJECTORY_DRIFT),
            None
        )
        assert drift_result is not None
        assert not drift_result.passed
        assert "基线感知" in drift_result.reason


# ============================================================
# 测试：get_baseline_adjustment_info
# ============================================================

class TestGetBaselineAdjustmentInfo:
    """测试 get_baseline_adjustment_info 方法"""

    def test_no_baseline_returns_not_adjusted(self):
        """无基线数据时返回未调整"""
        auditor = SafetyAuditor()
        
        info = auditor.get_baseline_adjustment_info()
        
        assert info["baseline_adjusted"] is False
        assert info["adjusted_axes"] == []
        assert info["baseline_evidence"] == []

    def test_with_baseline_returns_adjusted(self):
        """有基线偏离时返回调整信息"""
        # 使用自定义配置确保映射正确
        config = {
            "baseline_sensitivity": {
                "enabled": True,
                "axis_boost": {"crisis_delay": 1.5},
                "modality_to_axis_mapping": {
                    "heartRate": "crisis_delay",
                    "breathingRate": "crisis_delay",
                },
            },
            "crisis_delay": {"enabled": True, "max_turns_without_escalation": 2},
            "delusion_reinforcement": {"enabled": True, "confidence_threshold": 0.7},
            "stigma_rejection": {"enabled": True},
            "sycophancy": {"enabled": True, "agreement_rate_threshold": 0.9},
            "trajectory_drift": {"enabled": True, "deviation_threshold": 0.5},
        }
        auditor = SafetyAuditor(config=config)
        deviation = MockBaselineDeviation(
            significant_deviations=["heartRate", "breathingRate"]
        )
        auditor.set_baseline_context(deviation)
        
        # 调用 audit() 触发 boost 计算（_baseline_axis_boost 在 audit() 中设置）
        context = AuditContext(llm_output="测试")
        auditor.audit(context)
        
        info = auditor.get_baseline_adjustment_info()
        
        assert info["baseline_adjusted"] is True
        assert "crisis_delay" in info["adjusted_axes"]
        assert "heartRate" in info["baseline_evidence"]
        assert "breathingRate" in info["baseline_evidence"]


# ============================================================
# 测试：审计器行为一致性
# ============================================================

class TestAuditorConsistency:
    """测试审计器在无基线数据时行为与原来完全一致"""

    def test_audit_without_baseline_same_as_before(self):
        """无基线数据时，审计结果应与原来完全一致"""
        auditor = SafetyAuditor()
        
        # 构建一个正常的审计上下文（不设置 current_topic 以跳过轨迹漂移检测）
        context = AuditContext(
            llm_output="我理解你的感受，这确实很困难",
            dialog_state="ENGAGED",
            turn_count=5,
            crisis_turns=0,
            risk_level="low",
            conversation_history=[],
            llm_confidence=0.5,
            current_topic="",  # 不设置话题，跳过轨迹漂移检测
        )
        
        verdict = auditor.audit(context)
        
        # 所有轴应该通过
        assert verdict.passed is True
        assert all(r.passed for r in verdict.results)

    def test_audit_with_empty_baseline_same_as_before(self):
        """空基线偏离时，审计结果应与原来完全一致"""
        auditor = SafetyAuditor()
        auditor.set_baseline_context(MockBaselineDeviation(
            significant_deviations=[]  # 无显著偏离
        ))
        
        context = AuditContext(
            llm_output="我理解你的感受",
            dialog_state="ENGAGED",
        )
        
        verdict = auditor.auditor.audit(context) if hasattr(auditor, 'auditor') else auditor.audit(context)
        
        assert verdict.passed is True


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
