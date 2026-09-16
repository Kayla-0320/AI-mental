"""
五轴安全审计器单元测试

覆盖：
- 五轴审计逻辑
- 配置加载
- 审计结果数据结构
- 中间件包装
- 重写器逻辑
"""
from __future__ import annotations

import pytest

from intervention.auditor import (
    AuditContext,
    SafetyAuditor,
    audit_llm_output,
    generate_crisis_resource_card,
    load_audit_config,
)
from intervention.middleware import (
    AuditedLLMMiddleware,
    DefaultRewriter,
    LLMResponse,
    MiddlewareConfig,
    create_audited_middleware,
)
from shared.dataclasses import (
    AuditAction,
    AuditAxis,
    AuditResult,
    AuditVerdict,
)


# ============================================================
# 数据类测试
# ============================================================

class TestAuditDataClasses:
    def test_audit_axis_values(self):
        """五个审计轴全部定义"""
        assert len(AuditAxis) == 5
        expected = {"crisis_delay", "delusion_reinforcement", "stigma_rejection",
                    "sycophancy", "trajectory_drift"}
        assert {a.value for a in AuditAxis} == expected

    def test_audit_action_values(self):
        """审计动作定义完整"""
        assert len(AuditAction) == 4

    def test_audit_result_structure(self):
        """AuditResult 结构正确"""
        result = AuditResult(
            axis=AuditAxis.CRISIS_DELAY,
            passed=False,
            reason="测试理由",
            suggested_action=AuditAction.ESCALATE_HUMAN,
        )
        assert result.axis == AuditAxis.CRISIS_DELAY
        assert result.passed is False
        assert result.reason == "测试理由"
        assert result.suggested_action == AuditAction.ESCALATE_HUMAN

    def test_audit_verdict_structure(self):
        """AuditVerdict 结构正确"""
        verdict = AuditVerdict(
            passed=False,
            results=[
                AuditResult(axis=AuditAxis.STIGMA_REJECTION, passed=False),
            ],
            final_action=AuditAction.REWRITE,
            rewritten_content="重写内容",
        )
        assert verdict.passed is False
        assert len(verdict.results) == 1
        assert verdict.final_action == AuditAction.REWRITE


# ============================================================
# 配置加载测试
# ============================================================

class TestConfigLoading:
    def test_load_config_returns_dict(self):
        """加载配置返回字典"""
        config = load_audit_config()
        assert isinstance(config, dict)
        assert "crisis_delay" in config
        assert "delusion_reinforcement" in config

    def test_config_has_thresholds(self):
        """配置包含阈值"""
        config = load_audit_config()
        assert "max_turns_without_escalation" in config["crisis_delay"]
        assert "confidence_threshold" in config["delusion_reinforcement"]
        assert "agreement_rate_threshold" in config["sycophancy"]
        assert "deviation_threshold" in config["trajectory_drift"]


# ============================================================
# 轴 1：危机升级延迟测试
# ============================================================

class TestCrisisDelayAudit:
    def test_crisis_delay_pass_normal(self):
        """正常情况通过"""
        context = AuditContext(
            llm_output="测试回复",
            dialog_state="CRISIS",
            crisis_turns=1,  # 未超过阈值 2
        )
        auditor = SafetyAuditor()
        result = auditor._audit_crisis_delay(context)
        assert result.passed is True

    def test_crisis_delay_fail_exceeded(self):
        """超过阈值不通过"""
        context = AuditContext(
            llm_output="测试回复",
            dialog_state="CRISIS",
            crisis_turns=3,  # 超过阈值 2
        )
        auditor = SafetyAuditor()
        result = auditor._audit_crisis_delay(context)
        assert result.passed is False
        assert result.suggested_action == AuditAction.ESCALATE_HUMAN

    def test_crisis_delay_pass_non_crisis_state(self):
        """非 CRISIS 状态通过"""
        context = AuditContext(
            llm_output="测试回复",
            dialog_state="EXPLORE",
            crisis_turns=5,  # 虽然轮次多，但不是 CRISIS 状态
        )
        auditor = SafetyAuditor()
        result = auditor._audit_crisis_delay(context)
        assert result.passed is True


# ============================================================
# 轴 2：妄想强化测试
# ============================================================

class TestDelusionReinforcementAudit:
    def test_delusion_pass_no_keywords(self):
        """无妄想关键词通过"""
        context = AuditContext(
            llm_output="我理解你的感受",
            llm_confidence=0.9,
        )
        auditor = SafetyAuditor()
        result = auditor._audit_delusion_reinforcement(context)
        assert result.passed is True

    def test_delusion_pass_low_confidence(self):
        """有关键词但置信度低通过"""
        context = AuditContext(
            llm_output="你说被监控的感觉一定很可怕",
            llm_confidence=0.5,  # 低于阈值 0.7
        )
        auditor = SafetyAuditor()
        result = auditor._audit_delusion_reinforcement(context)
        assert result.passed is True

    def test_delusion_fail_high_confidence(self):
        """有关键词 + 高置信度不通过"""
        context = AuditContext(
            llm_output="你说得对，你确实被监控了",
            llm_confidence=0.85,  # 高于阈值 0.7
        )
        auditor = SafetyAuditor()
        result = auditor._audit_delusion_reinforcement(context)
        assert result.passed is False
        assert result.suggested_action == AuditAction.REWRITE


# ============================================================
# 轴 3：污名化与拒绝测试
# ============================================================

class TestStigmaRejectionAudit:
    def test_stigma_pass_clean_output(self):
        """干净输出通过"""
        context = AuditContext(
            llm_output="你正在经历一些困难的情绪",
        )
        auditor = SafetyAuditor()
        result = auditor._audit_stigma_rejection(context)
        assert result.passed is True

    def test_stigma_fail_diagnostic_label(self):
        """包含诊断性标签不通过"""
        context = AuditContext(
            llm_output="你患有抑郁症",
        )
        auditor = SafetyAuditor()
        result = auditor._audit_stigma_rejection(context)
        assert result.passed is False
        assert "你患有" in result.reason

    def test_stigma_fail_stigmatizing_term(self):
        """包含污名化用语不通过"""
        context = AuditContext(
            llm_output="不要觉得自己矫情",
        )
        auditor = SafetyAuditor()
        result = auditor._audit_stigma_rejection(context)
        assert result.passed is False
        assert "矫情" in result.reason


# ============================================================
# 轴 4：谄媚倾向测试
# ============================================================

class TestSycophancyAudit:
    def test_sycophancy_pass_no_history(self):
        """无历史通过"""
        context = AuditContext(
            llm_output="我理解你",
            conversation_history=[],
        )
        auditor = SafetyAuditor()
        result = auditor._audit_sycophancy(context)
        assert result.passed is True

    def test_sycophancy_pass_low_agreement(self):
        """低同意率通过"""
        context = AuditContext(
            llm_output="我理解你",
            conversation_history=[
                {"role": "assistant", "content": "你说得对"},
                {"role": "assistant", "content": "这确实很难"},
                {"role": "assistant", "content": "我理解你的感受"},
            ],
        )
        auditor = SafetyAuditor()
        result = auditor._audit_sycophancy(context)
        assert result.passed is True

    def test_sycophancy_fail_high_agreement(self):
        """高同意率不通过"""
        context = AuditContext(
            llm_output="我完全同意",
            conversation_history=[
                {"role": "assistant", "content": "你说得对"},
                {"role": "assistant", "content": "完全正确"},
                {"role": "assistant", "content": "我完全同意"},
                {"role": "assistant", "content": "你没错"},
                {"role": "assistant", "content": "确实如此"},
            ],
        )
        auditor = SafetyAuditor()
        result = auditor._audit_sycophancy(context)
        assert result.passed is False
        assert "同意率" in result.reason


# ============================================================
# 轴 5：轨迹漂移测试
# ============================================================

class TestTrajectoryDriftAudit:
    def test_drift_pass_no_topic(self):
        """无话题信息跳过"""
        context = AuditContext(
            llm_output="测试回复",
            current_topic="",
        )
        auditor = SafetyAuditor()
        result = auditor._audit_trajectory_drift(context)
        assert result.passed is True

    def test_drift_pass_relevant_output(self):
        """相关输出通过"""
        context = AuditContext(
            llm_output="你的情绪和感受很重要，焦虑和压力是可以应对的，想法可以调整",  # 包含 6/9 关键词
            current_topic="情绪调节",
        )
        auditor = SafetyAuditor()
        result = auditor._audit_trajectory_drift(context)
        assert result.passed is True

    def test_drift_fail_irrelevant_output(self):
        """无关输出不通过"""
        context = AuditContext(
            llm_output="今天天气不错，我们去玩吧",
            current_topic="焦虑管理",
        )
        auditor = SafetyAuditor()
        result = auditor._audit_trajectory_drift(context)
        assert result.passed is False
        assert "偏离度" in result.reason


# ============================================================
# 综合审计测试
# ============================================================

class TestFullAudit:
    def test_audit_all_pass(self):
        """全部通过"""
        verdict = audit_llm_output(
            llm_output="我理解你的感受",
            dialog_state="EXPLORE",
            turn_count=3,
        )
        assert verdict.passed is True
        assert verdict.final_action == AuditAction.PASS

    def test_audit_crisis_delay_fail(self):
        """危机延迟失败"""
        verdict = audit_llm_output(
            llm_output="测试",
            dialog_state="CRISIS",
            crisis_turns=5,
        )
        assert verdict.passed is False
        assert verdict.final_action == AuditAction.ESCALATE_HUMAN

    def test_audit_stigma_fail(self):
        """污名化失败"""
        verdict = audit_llm_output(
            llm_output="你患有抑郁症",
        )
        assert verdict.passed is False
        assert verdict.final_action == AuditAction.REWRITE


# ============================================================
# 危机资源卡测试
# ============================================================

class TestCrisisResourceCard:
    def test_generate_card(self):
        """生成资源卡"""
        card = generate_crisis_resource_card()
        assert "心理援助" in card or "热线" in card

    def test_card_contains_hotlines(self):
        """资源卡包含热线"""
        card = generate_crisis_resource_card()
        assert "400" in card or "010" in card


# ============================================================
# 中间件测试
# ============================================================

class MockLLMProvider:
    """模拟 LLM 提供者"""

    def __init__(self, response: str = "测试回复", confidence: float = 0.5):
        self.response = response
        self.confidence = confidence

    def generate(self, prompt: str, context: dict = None) -> LLMResponse:
        return LLMResponse(
            content=self.response,
            confidence=self.confidence,
        )


class TestMiddleware:
    def test_middleware_pass_clean_output(self):
        """干净输出通过"""
        middleware = create_audited_middleware(
            llm_provider=MockLLMProvider("我理解你的感受"),
        )
        response = middleware.generate(
            prompt="测试",
            dialog_state="EXPLORE",
        )
        assert response.content == "我理解你的感受"
        assert response.metadata["audit_passed"] is True

    def test_middleware_rewrite_stigma(self):
        """污名化内容被重写"""
        middleware = create_audited_middleware(
            llm_provider=MockLLMProvider("你患有抑郁症"),
        )
        response = middleware.generate(prompt="测试")
        # 应该被重写（"你患有" → "你可能正在经历"）
        assert "你患有" not in response.content
        assert response.metadata["rewrite_attempts"] >= 1

    def test_middleware_escalate_crisis(self):
        """危机延迟触发升级"""
        middleware = create_audited_middleware(
            llm_provider=MockLLMProvider("测试"),
        )
        response = middleware.generate(
            prompt="测试",
            dialog_state="CRISIS",
            crisis_turns=5,
        )
        assert "专业" in response.content or "热线" in response.content
        assert response.metadata["audit_action"] == "escalate_human"

    def test_middleware_disable_audit(self):
        """禁用审计"""
        middleware = create_audited_middleware(
            llm_provider=MockLLMProvider("你患有抑郁症"),
            enable_audit=False,
        )
        response = middleware.generate(prompt="测试")
        # 不审计，原样输出
        assert response.content == "你患有抑郁症"


# ============================================================
# 重写器测试
# ============================================================

class TestDefaultRewriter:
    def test_rewriter_removes_diagnostic_labels(self):
        """移除诊断性标签"""
        rewriter = DefaultRewriter()
        result = rewriter.rewrite(
            "你患有抑郁症",
            AuditVerdict(passed=False, final_action=AuditAction.REWRITE),
            AuditContext(llm_output="你患有抑郁症"),
        )
        assert "你患有" not in result
        assert "你可能正在经历" in result

    def test_rewriter_removes_stigma(self):
        """移除污名化用语"""
        rewriter = DefaultRewriter()
        result = rewriter.rewrite(
            "不要觉得自己矫情",
            AuditVerdict(passed=False, final_action=AuditAction.REWRITE),
            AuditContext(llm_output="不要觉得自己矫情"),
        )
        assert "矫情" not in result
        assert "敏感" in result
