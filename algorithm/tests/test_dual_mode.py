"""
双模态对话引擎单元测试 —— 苏格拉底反问状态感知

测试覆盖：
    1. 情绪未平复时不切换到 SOCRATIC 模式
    2. 危机状态时不切换到 SOCRATIC 模式
    3. 基线显著偏离时不切换到 SOCRATIC 模式
    4. 用户连续回避 2 次后自动退回共情模式
    5. 正常探索意愿时正确切换到 SOCRATIC 模式
    6. 第六轴审计不通过时强制退回共情模式
    7. SOCRATIC_READY 状态转移正确性
    8. DialogEngine 双模态属性追踪
"""
import pytest
import sys
import os

# 添加算法目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from intervention.state_machine import (
    DialogState,
    DialogueMode,
    DialogEngine,
    evaluate_mode,
    has_exploration_intent,
    is_avoidance,
    transition,
)
from shared.dataclasses import RiskLevel


# ============================================================
# 测试 1：情绪未平复时不切换
# ============================================================

class TestEmotionNotCalm:
    """情绪未平复时不应切换到 SOCRATIC 模式"""

    def test_evaluate_mode_emotion_not_calm(self):
        """连续 calm turns < 2 时应返回 EMPATHY"""
        mode = evaluate_mode(
            user_input="让我想想为什么",
            risk_level=RiskLevel.LOW,
            consecutive_calm_turns=0,  # 情绪未平复
            max_abs_z_score=0.5,
        )
        assert mode == DialogueMode.EMPATHY

    def test_evaluate_mode_one_calm_turn(self):
        """只有 1 轮 calm 时不应切换"""
        mode = evaluate_mode(
            user_input="帮我分析一下",
            risk_level=RiskLevel.LOW,
            consecutive_calm_turns=1,  # 只有 1 轮
            max_abs_z_score=0.5,
        )
        assert mode == DialogueMode.EMPATHY

    def test_evaluate_mode_two_calm_turns_with_intent(self):
        """连续 2 轮 calm + 探索意愿 → 应切换"""
        mode = evaluate_mode(
            user_input="让我想想为什么",
            risk_level=RiskLevel.LOW,
            consecutive_calm_turns=2,  # 刚好 2 轮
            max_abs_z_score=0.5,
        )
        assert mode == DialogueMode.SOCRATIC


# ============================================================
# 测试 2：危机状态时不切换
# ============================================================

class TestCrisisState:
    """危机状态时不应切换到 SOCRATIC 模式"""

    def test_evaluate_mode_crisis_risk(self):
        """CRISIS 风险等级时不应切换"""
        mode = evaluate_mode(
            user_input="让我想想",
            risk_level=RiskLevel.CRISIS,
            consecutive_calm_turns=5,
            max_abs_z_score=0.5,
        )
        assert mode == DialogueMode.EMPATHY

    def test_evaluate_mode_high_risk(self):
        """HIGH 风险等级时不应切换"""
        mode = evaluate_mode(
            user_input="帮我分析一下",
            risk_level=RiskLevel.HIGH,
            consecutive_calm_turns=5,
            max_abs_z_score=0.5,
        )
        assert mode == DialogueMode.EMPATHY

    def test_transition_crisis_from_socratic_ready(self):
        """SOCRATIC_READY 状态下检测到危机 → 转入 CRISIS"""
        next_state = transition(
            DialogState.SOCRATIC_READY,
            RiskLevel.CRISIS,
            turn_count=5,
        )
        assert next_state == DialogState.CRISIS


# ============================================================
# 测试 3：基线显著偏离时不切换
# ============================================================

class TestBaselineDeviation:
    """基线显著偏离时不应切换到 SOCRATIC 模式"""

    def test_evaluate_mode_high_z_score(self):
        """|Z| > 2.0 时不应切换"""
        mode = evaluate_mode(
            user_input="让我想想为什么",
            risk_level=RiskLevel.LOW,
            consecutive_calm_turns=5,
            max_abs_z_score=2.5,  # 显著偏离
        )
        assert mode == DialogueMode.EMPATHY

    def test_evaluate_mode_boundary_z_score(self):
        """|Z| = 2.0 时恰好可以切换（阈值是 ≤ 2.0）"""
        mode = evaluate_mode(
            user_input="让我想想为什么",
            risk_level=RiskLevel.LOW,
            consecutive_calm_turns=5,
            max_abs_z_score=2.0,  # 恰好在阈值
        )
        assert mode == DialogueMode.SOCRATIC

    def test_evaluate_mode_negative_z_score(self):
        """负 Z-score 的绝对值也应被检测"""
        mode = evaluate_mode(
            user_input="帮我分析",
            risk_level=RiskLevel.LOW,
            consecutive_calm_turns=5,
            max_abs_z_score=-3.0,  # 负值但绝对值大
        )
        assert mode == DialogueMode.EMPATHY


# ============================================================
# 测试 4：用户连续回避 2 次后自动退回共情
# ============================================================

class TestAvoidanceExit:
    """用户连续回避 2 次后应自动退回共情模式"""

    def test_engine_avoidance_counting(self):
        """引擎应正确计数回避次数"""
        engine = DialogEngine("test_user")
        engine.start()

        engine.record_avoidance(True)
        assert engine.consecutive_avoidance_count == 1
        assert not engine.should_exit_socratic()

        engine.record_avoidance(True)
        assert engine.consecutive_avoidance_count == 2
        assert engine.should_exit_socratic()

    def test_engine_avoidance_reset(self):
        """非回避输入应重置计数"""
        engine = DialogEngine("test_user")
        engine.start()

        engine.record_avoidance(True)
        engine.record_avoidance(True)
        assert engine.consecutive_avoidance_count == 2

        engine.record_avoidance(False)  # 非回避
        assert engine.consecutive_avoidance_count == 0

    def test_engine_exit_socratic_on_avoidance(self):
        """连续 2 次回避后 evaluate_and_update_mode 应退回 EMPATHY"""
        engine = DialogEngine("test_user")
        engine.start(risk_level=RiskLevel.LOW)
        # 手动设置到 SOCRATIC_READY 状态
        engine.current_state = DialogState.SOCRATIC_READY
        engine.current_mode = DialogueMode.SOCRATIC
        engine.consecutive_avoidance_count = 2

        mode = engine.evaluate_and_update_mode("不想说", RiskLevel.LOW)
        assert mode == DialogueMode.EMPATHY
        assert engine.current_state == DialogState.EXPLORE

    def test_is_avoidance_keywords(self):
        """回避关键词检测"""
        assert is_avoidance("不想说")
        assert is_avoidance("别问了")
        assert is_avoidance("算了")
        assert is_avoidance("换个话题")
        assert not is_avoidance("让我想想")
        assert not is_avoidance("")


# ============================================================
# 测试 5：正常探索意愿时正确切换
# ============================================================

class TestExplorationIntent:
    """正常探索意愿时应正确切换到 SOCRATIC 模式"""

    def test_has_exploration_intent_positive(self):
        """包含探索意愿关键词"""
        assert has_exploration_intent("让我想想为什么")
        assert has_exploration_intent("帮我分析一下")
        assert has_exploration_intent("是怎么回事")
        assert has_exploration_intent("然后呢")

    def test_has_exploration_intent_negative(self):
        """不包含探索意愿关键词"""
        assert not has_exploration_intent("我好烦")
        assert not has_exploration_intent("不想说")
        assert not has_exploration_intent("")

    def test_full_mode_switch(self):
        """完整模式切换流程"""
        engine = DialogEngine("test_user")
        engine.start(risk_level=RiskLevel.LOW)

        # 先推进到 EXPLORE 状态（需要 2 轮 INIT）
        engine.current_state = DialogState.EXPLORE

        # 模拟情绪平复
        engine.update_emotion_state(False)  # calm turn 1
        engine.update_emotion_state(False)  # calm turn 2
        assert engine.consecutive_calm_turns == 2

        # 模拟正常基线
        engine.update_baseline_z_score(0.5)

        # 用户表达探索意愿
        mode = engine.evaluate_and_update_mode("让我想想为什么", RiskLevel.LOW)
        assert mode == DialogueMode.SOCRATIC
        assert engine.current_state == DialogState.SOCRATIC_READY


# ============================================================
# 测试 6：第六轴审计不通过时强制退回
# ============================================================

class TestSocraticTimingAudit:
    """第六轴审计不通过时应强制退回共情模式"""

    def test_audit_crisis_state_blocks_socratic(self):
        """危机状态下第六轴应不通过"""
        from intervention.auditor import SafetyAuditor, AuditContext

        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么会这样呢？",
            dialog_state="SOCRATIC_READY",
            dialogue_mode="SOCRATIC",
            risk_level="crisis",
            consecutive_calm_turns=5,
            consecutive_avoidance_count=0,
            max_abs_z_score=0.5,
        )

        result = auditor._audit_socratic_timing(context)
        assert not result.passed
        assert result.axis.value == "socratic_timing"

    def test_audit_emotion_not_calm_blocks_socratic(self):
        """情绪未平复时第六轴应不通过"""
        from intervention.auditor import SafetyAuditor, AuditContext

        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么呢？",
            dialog_state="SOCRATIC_READY",
            dialogue_mode="SOCRATIC",
            risk_level="low",
            consecutive_calm_turns=0,  # 情绪未平复
            consecutive_avoidance_count=0,
            max_abs_z_score=0.5,
        )

        result = auditor._audit_socratic_timing(context)
        assert not result.passed

    def test_audit_avoidance_blocks_socratic(self):
        """用户连续回避后第六轴应不通过"""
        from intervention.auditor import SafetyAuditor, AuditContext

        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你怎么看这个问题？",
            dialog_state="SOCRATIC_READY",
            dialogue_mode="SOCRATIC",
            risk_level="low",
            consecutive_calm_turns=5,
            consecutive_avoidance_count=2,  # 连续回避 2 次
            max_abs_z_score=0.5,
        )

        result = auditor._audit_socratic_timing(context)
        assert not result.passed

    def test_audit_baseline_deviation_blocks_socratic(self):
        """基线显著偏离时第六轴应不通过"""
        from intervention.auditor import SafetyAuditor, AuditContext

        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么呢？",
            dialog_state="SOCRATIC_READY",
            dialogue_mode="SOCRATIC",
            risk_level="low",
            consecutive_calm_turns=5,
            consecutive_avoidance_count=0,
            max_abs_z_score=3.0,  # 显著偏离
        )

        result = auditor._audit_socratic_timing(context)
        assert not result.passed

    def test_audit_passes_when_all_conditions_met(self):
        """所有条件满足时第六轴应通过"""
        from intervention.auditor import SafetyAuditor, AuditContext

        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么呢？",
            dialog_state="SOCRATIC_READY",
            dialogue_mode="SOCRATIC",
            risk_level="low",
            consecutive_calm_turns=3,
            consecutive_avoidance_count=0,
            max_abs_z_score=1.0,
        )

        result = auditor._audit_socratic_timing(context)
        assert result.passed

    def test_audit_skipped_in_empathy_mode(self):
        """EMPATHY 模式下第六轴应自动通过"""
        from intervention.auditor import SafetyAuditor, AuditContext

        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="我理解你的感受",
            dialog_state="EXPLORE",
            dialogue_mode="EMPATHY",  # 共情模式
            risk_level="crisis",  # 即使是危机
            consecutive_calm_turns=0,
            consecutive_avoidance_count=5,
            max_abs_z_score=5.0,
        )

        result = auditor._audit_socratic_timing(context)
        assert result.passed  # EMPATHY 模式下不审计


# ============================================================
# 测试 7：SOCRATIC_READY 状态转移正确性
# ============================================================

class TestSocraticReadyTransition:
    """SOCRATIC_READY 状态转移规则"""

    def test_socratic_ready_stays_on_low_risk(self):
        """SOCRATIC_READY + LOW 风险 → 保持 SOCRATIC_READY"""
        next_state = transition(
            DialogState.SOCRATIC_READY,
            RiskLevel.LOW,
            turn_count=5,
        )
        assert next_state == DialogState.SOCRATIC_READY

    def test_socratic_ready_stays_on_medium_risk(self):
        """SOCRATIC_READY + MEDIUM 风险 → 保持 SOCRATIC_READY"""
        next_state = transition(
            DialogState.SOCRATIC_READY,
            RiskLevel.MEDIUM,
            turn_count=5,
        )
        assert next_state == DialogState.SOCRATIC_READY

    def test_socratic_ready_to_explore_on_high_risk(self):
        """SOCRATIC_READY + HIGH 风险 → 退回 EXPLORE"""
        next_state = transition(
            DialogState.SOCRATIC_READY,
            RiskLevel.HIGH,
            turn_count=5,
        )
        assert next_state == DialogState.EXPLORE

    def test_socratic_ready_to_crisis(self):
        """SOCRATIC_READY + CRISIS → 进入 CRISIS"""
        next_state = transition(
            DialogState.SOCRATIC_READY,
            RiskLevel.CRISIS,
            turn_count=5,
        )
        assert next_state == DialogState.CRISIS


# ============================================================
# 测试 8：DialogEngine 双模态属性追踪
# ============================================================

class TestDialogEngineDualMode:
    """DialogEngine 双模态属性追踪"""

    def test_engine_initial_mode(self):
        """引擎初始模式应为 EMPATHY"""
        engine = DialogEngine("test_user")
        engine.start()
        assert engine.current_mode == DialogueMode.EMPATHY
        assert engine.consecutive_calm_turns == 0
        assert engine.consecutive_avoidance_count == 0
        assert engine.max_abs_z_score == 0.0

    def test_engine_update_emotion_state(self):
        """情绪状态更新"""
        engine = DialogEngine("test_user")
        engine.start()

        engine.update_emotion_state(False)  # calm
        assert engine.consecutive_calm_turns == 1

        engine.update_emotion_state(False)  # calm
        assert engine.consecutive_calm_turns == 2

        engine.update_emotion_state(True)  # negative
        assert engine.consecutive_calm_turns == 0

    def test_engine_state_summary_includes_mode(self):
        """状态摘要应包含模式信息"""
        engine = DialogEngine("test_user")
        engine.start()
        engine.current_mode = DialogueMode.SOCRATIC

        summary = engine.get_state_summary()
        assert "current_mode" in summary
        assert summary["current_mode"] == "SOCRATIC"
        assert "consecutive_calm_turns" in summary
        assert "consecutive_avoidance_count" in summary
        assert "max_abs_z_score" in summary

    def test_engine_reset_on_start(self):
        """start() 应重置所有双模态属性"""
        engine = DialogEngine("test_user")
        engine.start()
        engine.current_mode = DialogueMode.SOCRATIC
        engine.consecutive_calm_turns = 5
        engine.consecutive_avoidance_count = 2
        engine.max_abs_z_score = 3.0

        engine.start()  # 重新启动
        assert engine.current_mode == DialogueMode.EMPATHY
        assert engine.consecutive_calm_turns == 0
        assert engine.consecutive_avoidance_count == 0
        assert engine.max_abs_z_score == 0.0
