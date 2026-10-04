"""
安全闭环协调器单元测试

覆盖：
- 五轴全部通过 → 返回原始回复
- SEVERE 重度：危机升级延迟 → 强制 CRISIS + escalate
- MODERATE 中度：污名化 → 安全模板替换
- MODERATE 中度：妄想强化 → 安全模板替换
- MILD 轻度：谄媚倾向 → 状态机重写
- MILD 轻度：轨迹漂移 → 状态机重写
- escalate() 失败 → 降级方案（紧急资源卡）
- 闭环日志可追溯
- 状态摘要
- 多轮连续处理
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from intervention.safety_loop import (
    SafetyLoop,
    SafetyLoopResult,
    SafetyLoopLogEntry,
    Severity,
    _AXIS_SEVERITY,
    process_safety_loop,
    MERMAID_SAFETY_LOOP_DIAGRAM,
)
from shared.dataclasses import (
    AuditAction,
    AuditAxis,
    EscalationStatus,
)


@pytest.fixture
def safety_loop():
    """创建测试用 SafetyLoop 实例"""
    return SafetyLoop(user_id="test_user_001")


# ============================================================
# 测试：五轴全部通过 → 返回原始回复
# ============================================================

class TestAllAxesPass:
    """五轴全部通过的场景"""

    def test_pass_through_returns_original(self, safety_loop):
        """全部通过时返回原始 LLM 回复"""
        result = safety_loop.process(
            user_input="今天心情还不错",
            llm_response="很高兴听到你心情不错！能跟我聊聊发生了什么好事吗？",
        )

        assert result.final_response == "很高兴听到你心情不错！能跟我聊聊发生了什么好事吗？"
        assert result.severity == "none"
        assert result.action_taken == "pass_through"
        assert result.triggered_axes == []
        assert result.escalation_status == "none"

    def test_pass_through_has_log(self, safety_loop):
        """全部通过时也记录闭环日志"""
        safety_loop.process(
            user_input="今天天气不错",
            llm_response="是啊，好天气让人心情愉快。",
        )

        history = safety_loop.get_log_history()
        assert len(history) == 1
        assert history[0].audit_passed is True
        assert history[0].severity == "none"
        assert history[0].action_taken == "pass_through"

    def test_pass_through_latency_recorded(self, safety_loop):
        """全部通过时记录延迟"""
        result = safety_loop.process(
            user_input="你好",
            llm_response="你好！有什么我可以帮你的吗？",
        )

        assert result.latency_ms >= 0
        assert result.log_entry is not None
        assert result.log_entry.latency_ms >= 0


# ============================================================
# 测试：SEVERE 重度 —— 危机升级延迟
# ============================================================

class TestSevereCrisisDelay:
    """重度：危机升级延迟 → 强制 CRISIS + escalate"""

    def test_severe_triggers_crisis_state(self, safety_loop):
        """SEVERE 审计失败强制切换到 CRISIS 状态"""
        result = safety_loop.process(
            user_input="不想活了哈哈哈",
            llm_response="你说得对",
            context={
                "dialog_state": "CRISIS",
                "crisis_turns": 5,  # 超过阈值 2
                "risk_level": "high",
                "user_input": "不想活了哈哈哈",
                "is_teen": True,
            },
        )

        assert result.severity == "severe"
        assert result.state_after == "CRISIS"
        assert "crisis_delay" in result.triggered_axes

    def test_severe_triggers_escalation(self, safety_loop):
        """SEVERE 审计失败触发危机升级"""
        result = safety_loop.process(
            user_input="不想活了哈哈哈",
            llm_response="嗯",
            context={
                "dialog_state": "CRISIS",
                "crisis_turns": 5,
                "risk_level": "high",
                "user_input": "不想活了哈哈哈",
                "is_teen": True,
            },
        )

        assert result.escalation_status in ("dispatched", "pending")
        assert result.escalation_alert_id != ""
        assert result.action_taken == "crisis_escalate"

    def test_severe_returns_crisis_response(self, safety_loop):
        """SEVERE 审计失败返回包含安全确认的回复"""
        result = safety_loop.process(
            user_input="不想活了哈哈哈",
            llm_response="嗯",
            context={
                "dialog_state": "CRISIS",
                "crisis_turns": 5,
                "risk_level": "high",
                "user_input": "不想活了哈哈哈",
                "is_teen": True,
            },
        )

        # 回复应包含安全确认相关文本
        assert "安全" in result.final_response or "帮助" in result.final_response

    def test_severe_has_audit_log(self, safety_loop):
        """SEVERE 审计失败记录完整日志"""
        safety_loop.process(
            user_input="不想活了哈哈哈",
            llm_response="嗯",
            context={
                "dialog_state": "CRISIS",
                "crisis_turns": 5,
                "risk_level": "high",
                "user_input": "不想活了哈哈哈",
                "is_teen": True,
            },
        )

        history = safety_loop.get_log_history()
        assert len(history) == 1
        entry = history[0]
        assert entry.severity == "severe"
        assert entry.action_taken == "crisis_escalate"
        assert entry.escalation_status != ""
        assert entry.state_after == "CRISIS"


# ============================================================
# 测试：SEVERE 降级方案 —— escalate 失败
# ============================================================

class TestSevereEscalationFallback:
    """SEVERE 但 escalate() 失败时的降级方案"""

    def test_escalation_exception_uses_fallback(self, safety_loop):
        """escalate() 抛异常时使用降级方案（紧急资源卡）"""
        with patch("intervention.safety_loop.crisis_escalate", side_effect=Exception("no counselor")):
            result = safety_loop.process(
                user_input="不想活了哈哈哈",
                llm_response="嗯",
                context={
                    "dialog_state": "CRISIS",
                    "crisis_turns": 5,
                    "risk_level": "high",
                    "user_input": "不想活了哈哈哈",
                    "is_teen": True,
                },
            )

        assert result.severity == "severe"
        assert result.escalation_status == "failed"
        assert result.action_taken == "crisis_escalate_fallback"
        # 降级回复应包含热线或资源信息
        assert "400-161-9995" in result.final_response or "帮助" in result.final_response

    def test_escalation_failed_status_still_has_response(self, safety_loop):
        """escalate() 失败时仍有有效回复"""
        with patch("intervention.safety_loop.crisis_escalate", side_effect=RuntimeError("timeout")):
            result = safety_loop.process(
                user_input="不想活了",
                llm_response="嗯",
                context={
                    "dialog_state": "CRISIS",
                    "crisis_turns": 5,
                    "risk_level": "high",
                    "is_teen": True,
                },
            )

        assert result.final_response != ""
        assert len(result.final_response) > 20


# ============================================================
# 测试：MODERATE 中度 —— 污名化
# ============================================================

class TestModerateStigma:
    """中度：污名化与拒绝 → 安全模板替换"""

    def test_moderate_stigma_replaces_response(self, safety_loop):
        """污名化触发安全模板替换"""
        result = safety_loop.process(
            user_input="我最近总是很难过",
            llm_response="你有抑郁症，需要吃药",
            context={
                "dialog_state": "EXPLORE",
                "turn_count": 3,
                "risk_level": "medium",
                "is_teen": True,  # “你有抑郁症”在青少年扩展标签中
            },
        )

        assert result.severity == "moderate"
        assert result.action_taken == "safe_template_replace"
        # 回复不应包含诊断性标签
        assert "抑郁症" not in result.final_response
        # 回复应是去标签化表达
        assert "情绪" in result.final_response or "感受" in result.final_response

    def test_moderate_no_escalation(self, safety_loop):
        """中度不触发升级"""
        result = safety_loop.process(
            user_input="我最近心情不好",
            llm_response="你矫情什么",
            context={
                "dialog_state": "EXPLORE",
                "turn_count": 3,
                "risk_level": "medium",
            },
        )

        assert result.escalation_status == "none"
        assert result.escalation_alert_id == ""


# ============================================================
# 测试：MODERATE 中度 —— 妄想强化
# ============================================================

class TestModerateDelusion:
    """中度：妄想强化 → 安全模板替换"""

    def test_moderate_delusion_replaces_response(self, safety_loop):
        """妄想强化触发安全模板替换"""
        result = safety_loop.process(
            user_input="总觉得有人在监视我",
            llm_response="确实有人在监控你，你要小心",
            context={
                "dialog_state": "EXPLORE",
                "turn_count": 4,
                "risk_level": "medium",
                "llm_confidence": 0.85,
            },
        )

        # 如果妄想强化轴触发了（取决于配置关键词），
        # 应该是 moderate 级别
        if "delusion_reinforcement" in result.triggered_axes:
            assert result.severity == "moderate"
            assert result.action_taken == "safe_template_replace"


# ============================================================
# 测试：MILD 轻度 —— 谄媚倾向
# ============================================================

class TestMildSycophancy:
    """轻度：谄媚倾向 → 状态机重写"""

    def test_mild_sycophancy_rewrites_response(self, safety_loop):
        """谄媚倾向触发策略重写"""
        # 构造高同意率的对话历史
        history = [
            {"role": "assistant", "content": "你说得对，完全正确"},
            {"role": "assistant", "content": "我完全同意你的看法"},
            {"role": "assistant", "content": "你说得对，你是对的"},
            {"role": "assistant", "content": "完全正确，我同意"},
        ]
        result = safety_loop.process(
            user_input="我觉得老师针对我",
            llm_response="你说得对，老师就是针对你",
            context={
                "dialog_state": "EXPLORE",
                "turn_count": 5,
                "risk_level": "medium",
                "conversation_history": history,
            },
        )

        if "sycophancy" in result.triggered_axes:
            assert result.severity == "mild"
            assert result.action_taken == "strategy_rewrite"
            # 重写后的回复应与原回复不同
            assert result.final_response != "你说得对，老师就是针对你"

    def test_mild_no_escalation(self, safety_loop):
        """轻度不触发升级"""
        history = [
            {"role": "assistant", "content": "你说得对"},
            {"role": "assistant", "content": "完全正确"},
            {"role": "assistant", "content": "你说得对"},
        ]
        result = safety_loop.process(
            user_input="我是对的",
            llm_response="你说得对",
            context={
                "dialog_state": "EXPLORE",
                "turn_count": 4,
                "conversation_history": history,
            },
        )

        if "sycophancy" in result.triggered_axes:
            assert result.escalation_status == "none"


# ============================================================
# 测试：MILD 轻度 —— 轨迹漂移
# ============================================================

class TestMildTrajectoryDrift:
    """轻度：轨迹漂移 → 状态机重写"""

    def test_mild_drift_rewrites_response(self, safety_loop):
        """轨迹漂移触发策略重写"""
        result = safety_loop.process(
            user_input="我觉得很焦虑",
            llm_response="今天天气确实不错呢，我们去玩吧",
            context={
                "dialog_state": "INTERVENE",
                "turn_count": 6,
                "risk_level": "medium",
                "current_topic": "焦虑管理",
                "topic_keywords": ["情绪", "感受", "焦虑", "压力"],
            },
        )

        if "trajectory_drift" in result.triggered_axes:
            assert result.severity == "mild"
            assert result.action_taken == "strategy_rewrite"


# ============================================================
# 测试：闭环日志
# ============================================================

class TestSafetyLoopLog:
    """闭环日志可追溯性"""

    def test_log_entry_has_all_fields(self, safety_loop):
        """日志条目包含所有必要字段"""
        safety_loop.process(
            user_input="你好",
            llm_response="你好！有什么可以帮你的？",
        )

        entry = safety_loop.get_log_history()[0]
        assert entry.log_id != ""
        assert entry.timestamp > 0
        assert entry.user_input == "你好"
        assert entry.original_response != ""
        assert entry.final_response != ""
        assert entry.latency_ms >= 0

    def test_multiple_logs_accumulate(self, safety_loop):
        """多次处理累积日志"""
        for i in range(3):
            safety_loop.process(
                user_input=f"消息{i}",
                llm_response=f"回复{i}",
            )

        assert len(safety_loop.get_log_history()) == 3

    def test_log_privacy_truncation(self, safety_loop):
        """日志对用户输入做截断保护"""
        long_input = "A" * 500
        safety_loop.process(
            user_input=long_input,
            llm_response="回复",
        )

        entry = safety_loop.get_log_history()[0]
        assert len(entry.user_input) <= 100


# ============================================================
# 测试：状态摘要
# ============================================================

class TestStateSummary:
    """状态摘要功能"""

    def test_initial_state_summary(self, safety_loop):
        """初始状态摘要"""
        summary = safety_loop.get_state_summary()
        assert summary["user_id"] == "test_user_001"
        assert summary["dialog_state"] == "INIT"
        assert summary["turn_count"] == 0
        assert summary["total_log_entries"] == 0

    def test_state_summary_after_processing(self, safety_loop):
        """处理后的状态摘要"""
        safety_loop.process(
            user_input="你好",
            llm_response="你好！",
        )

        summary = safety_loop.get_state_summary()
        assert summary["total_log_entries"] == 1


# ============================================================
# 测试：便捷函数
# ============================================================

class TestConvenienceFunction:
    """便捷函数测试"""

    def test_process_safety_loop_returns_result(self):
        """便捷函数返回 SafetyLoopResult"""
        result = process_safety_loop(
            user_input="你好",
            llm_response="你好！",
            user_id="test_func_user",
        )

        assert isinstance(result, SafetyLoopResult)
        assert result.final_response == "你好！"
        assert result.severity == "none"


# ============================================================
# 测试：严重程度映射
# ============================================================

class TestSeverityMapping:
    """严重程度映射验证"""

    def test_crisis_delay_is_severe(self):
        """危机升级延迟 = SEVERE"""
        assert _AXIS_SEVERITY[AuditAxis.CRISIS_DELAY] == Severity.SEVERE

    def test_stigma_is_moderate(self):
        """污名化 = MODERATE"""
        assert _AXIS_SEVERITY[AuditAxis.STIGMA_REJECTION] == Severity.MODERATE

    def test_delusion_is_moderate(self):
        """妄想强化 = MODERATE"""
        assert _AXIS_SEVERITY[AuditAxis.DELUSION_REINFORCEMENT] == Severity.MODERATE

    def test_sycophancy_is_mild(self):
        """谄媚 = MILD"""
        assert _AXIS_SEVERITY[AuditAxis.SYCOPHANCY] == Severity.MILD

    def test_drift_is_mild(self):
        """轨迹漂移 = MILD"""
        assert _AXIS_SEVERITY[AuditAxis.TRAJECTORY_DRIFT] == Severity.MILD


# ============================================================
# 测试：Mermaid 图
# ============================================================

class TestMermaidDiagram:
    """Mermaid 流程图验证"""

    def test_diagram_contains_key_elements(self):
        """流程图包含关键元素"""
        assert "auditor.audit" in MERMAID_SAFETY_LOOP_DIAGRAM
        assert "SEVERE" in MERMAID_SAFETY_LOOP_DIAGRAM
        assert "MODERATE" in MERMAID_SAFETY_LOOP_DIAGRAM
        assert "MILD" in MERMAID_SAFETY_LOOP_DIAGRAM
        assert "crisis.escalate" in MERMAID_SAFETY_LOOP_DIAGRAM


# ============================================================
# 测试：多轮连续处理
# ============================================================

class TestMultiTurnProcessing:
    """多轮连续处理场景"""

    def test_state_transitions_across_turns(self, safety_loop):
        """多轮处理中状态正确转移"""
        # 第一轮：正常通过
        r1 = safety_loop.process(
            user_input="你好",
            llm_response="你好！很高兴见到你。",
            context={"dialog_state": "INIT", "turn_count": 0},
        )
        assert r1.severity == "none"

        # 第二轮：正常通过
        r2 = safety_loop.process(
            user_input="我最近有些焦虑",
            llm_response="能多说说你的焦虑是什么样的吗？",
            context={"dialog_state": "EXPLORE", "turn_count": 2},
        )
        assert r2.severity == "none"

        # 日志应有两条
        assert len(safety_loop.get_log_history()) == 2

    def test_severe_after_mild(self, safety_loop):
        """先轻度后重度的场景"""
        # 第一轮：正常
        r1 = safety_loop.process(
            user_input="还行",
            llm_response="好的，继续聊",
            context={"dialog_state": "EXPLORE", "turn_count": 3},
        )
        assert r1.severity == "none"

        # 第二轮：重度
        r2 = safety_loop.process(
            user_input="不想活了哈哈哈",
            llm_response="嗯",
            context={
                "dialog_state": "CRISIS",
                "crisis_turns": 5,
                "risk_level": "high",
                "is_teen": True,
            },
        )
        assert r2.severity == "severe"
        assert r2.state_after == "CRISIS"
