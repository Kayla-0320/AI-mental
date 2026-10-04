"""
双模态对话引擎深度分级 + CBT 记录 + 分级降级 单元测试

测试覆盖：
    1. 三级深度触发（SHALLOW/MEDIUM/DEEP）
    2. 三级降级（depth=0/1/2）
    3. 连续回避冻结（3 轮）
    4. CBT 记录生成（COMPLETE/PARTIAL/NOT_STARTED）
    5. 前端融合验证（无模式指示器）
"""
import pytest
import sys
from pathlib import Path

# 添加算法路径
algorithm_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(algorithm_path))

from intervention.state_machine import (
    DialogEngine,
    DialogState,
    DialogueMode,
    SocraticDepth,
    evaluate_depth,
    has_exploration_intent,
    is_avoidance,
)
from shared.dataclasses import RiskLevel, AuditAxis, AuditResult, AuditAction
from intervention.auditor import SafetyAuditor, AuditContext
from intervention.cbt.cbt_record import (
    CBTRecordTracker,
    CBTRestructuringRecord,
    CBTRecordStatus,
    store_cbt_record,
    get_cbt_record,
    get_user_cbt_records,
    clear_cbt_records,
)


# ============================================================
# 1. 三级深度触发测试
# ============================================================

class TestSocraticDepthTriggering:
    """测试三级深度触发逻辑"""

    def test_shallow_depth_emotion_just_calm(self):
        """情绪刚平复（calm_turns=2）→ SHALLOW"""
        depth = evaluate_depth(
            consecutive_calm_turns=2,
            max_abs_z_score=1.0,
            consecutive_exploration_turns=0,
            risk_level=RiskLevel.LOW,
        )
        assert depth == SocraticDepth.SHALLOW

    def test_shallow_depth_baseline_edge(self):
        """基线边缘偏离（|Z| ∈ [1.5, 2.0]）+ 不满足 DEEP → SHALLOW"""
        # 当不满足 DEEP 条件（如 exploration < 2）时，|Z| >= 1.5 降级到 SHALLOW
        depth = evaluate_depth(
            consecutive_calm_turns=4,
            max_abs_z_score=1.8,
            consecutive_exploration_turns=1,  # 不满足 DEEP 的 exploration >= 2
            risk_level=RiskLevel.LOW,
        )
        # |Z| >= 1.5 不满足 MEDIUM 条件，降级到 SHALLOW
        assert depth == SocraticDepth.SHALLOW

    def test_medium_depth_stable_emotion(self):
        """情绪稳定 + 基线正常 + 用户主动探索 → MEDIUM"""
        depth = evaluate_depth(
            consecutive_calm_turns=3,
            max_abs_z_score=1.0,
            consecutive_exploration_turns=1,
            risk_level=RiskLevel.LOW,
        )
        assert depth == SocraticDepth.MEDIUM

    def test_medium_depth_baseline_normal(self):
        """基线正常（|Z| < 1.5）→ MEDIUM"""
        depth = evaluate_depth(
            consecutive_calm_turns=4,
            max_abs_z_score=1.2,
            consecutive_exploration_turns=1,
            risk_level=RiskLevel.MEDIUM,
        )
        assert depth == SocraticDepth.MEDIUM

    def test_deep_depth_consecutive_exploration(self):
        """情绪稳定 + 非危机 + 连续 2 轮主动探索 → DEEP"""
        depth = evaluate_depth(
            consecutive_calm_turns=4,
            max_abs_z_score=0.5,
            consecutive_exploration_turns=2,
            risk_level=RiskLevel.LOW,
        )
        assert depth == SocraticDepth.DEEP

    def test_deep_depth_requires_calm_4_turns(self):
        """DEEP 需要 calm_turns >= 4"""
        depth = evaluate_depth(
            consecutive_calm_turns=3,  # 不够
            max_abs_z_score=0.5,
            consecutive_exploration_turns=3,
            risk_level=RiskLevel.LOW,
        )
        # calm_turns < 4，不满足 DEEP 条件
        assert depth == SocraticDepth.MEDIUM

    def test_deep_depth_requires_non_crisis(self):
        """DEEP 需要非危机状态"""
        depth = evaluate_depth(
            consecutive_calm_turns=5,
            max_abs_z_score=0.5,
            consecutive_exploration_turns=3,
            risk_level=RiskLevel.HIGH,  # 高风险
        )
        # risk_level=HIGH，不满足 DEEP 条件
        assert depth == SocraticDepth.MEDIUM


# ============================================================
# 2. 三级降级测试
# ============================================================

class TestSocraticDepthDowngrade:
    """测试三级降级逻辑"""

    def test_downgrade_to_empathy_crisis(self):
        """危机状态 → depth=0（退回共情）"""
        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么会这样呢？",
            dialogue_mode="SOCRATIC",
            dialog_state="SOCRATIC_READY",
            risk_level="crisis",
            consecutive_calm_turns=5,
        )
        result = auditor._audit_socratic_timing(context)
        assert result.passed is False
        assert result.recommended_depth == 0

    def test_downgrade_to_empathy_uncalm(self):
        """情绪未平复 → depth=0（退回共情）"""
        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么会这样呢？",
            dialogue_mode="SOCRATIC",
            dialog_state="SOCRATIC_READY",
            risk_level="low",
            consecutive_calm_turns=1,  # < min_calm_turns=2
        )
        result = auditor._audit_socratic_timing(context)
        assert result.passed is False
        assert result.recommended_depth == 0

    def test_downgrade_to_empathy_avoidance(self):
        """连续回避 → depth=0（退回共情，冻结 3 轮）"""
        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么会这样呢？",
            dialogue_mode="SOCRATIC",
            dialog_state="SOCRATIC_READY",
            risk_level="low",
            consecutive_calm_turns=5,
            consecutive_avoidance_count=2,  # >= max_avoidance=2
        )
        result = auditor._audit_socratic_timing(context)
        assert result.passed is False
        assert result.recommended_depth == 0

    def test_downgrade_to_shallow_baseline_deviation(self):
        """基线显著偏离 → depth=1（降级到 SHALLOW）"""
        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么会这样呢？",
            dialogue_mode="SOCRATIC",
            dialog_state="SOCRATIC_READY",
            risk_level="low",
            consecutive_calm_turns=5,
            max_abs_z_score=2.5,  # > max_abs_z_score=2.0
        )
        result = auditor._audit_socratic_timing(context)
        assert result.passed is False
        assert result.recommended_depth == 1

    def test_pass_with_depth_3_deep(self):
        """通过审计 + 情绪稳定 → depth=3（DEEP）"""
        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么会这样呢？",
            dialogue_mode="SOCRATIC",
            dialog_state="SOCRATIC_READY",
            risk_level="low",
            consecutive_calm_turns=5,  # >= 4
            max_abs_z_score=0.5,
        )
        result = auditor._audit_socratic_timing(context)
        assert result.passed is True
        assert result.recommended_depth == 3

    def test_pass_with_depth_2_medium(self):
        """通过审计 + 情绪稳定 + 基线正常 → depth=2（MEDIUM）"""
        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么会这样呢？",
            dialogue_mode="SOCRATIC",
            dialog_state="SOCRATIC_READY",
            risk_level="low",
            consecutive_calm_turns=3,  # >= 3
            max_abs_z_score=1.0,  # < 1.5
        )
        result = auditor._audit_socratic_timing(context)
        assert result.passed is True
        assert result.recommended_depth == 2

    def test_pass_with_depth_1_shallow(self):
        """通过审计 + 情绪刚平复 → depth=1（SHALLOW）"""
        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="你觉得为什么会这样呢？",
            dialogue_mode="SOCRATIC",
            dialog_state="SOCRATIC_READY",
            risk_level="low",
            consecutive_calm_turns=2,  # 刚达到阈值
            max_abs_z_score=1.0,
        )
        result = auditor._audit_socratic_timing(context)
        assert result.passed is True
        assert result.recommended_depth == 1


# ============================================================
# 3. 连续回避冻结测试
# ============================================================

class TestAvoidanceFreeze:
    """测试连续回避后深度冻结逻辑"""

    def test_avoidance_triggers_freeze(self):
        """连续回避 2 次 → 深度冻结 3 轮"""
        engine = DialogEngine(user_id="test_user")
        engine.start(risk_level=RiskLevel.LOW)
        engine.current_state = DialogState.EXPLORE
        engine.consecutive_calm_turns = 5

        # 模拟用户探索意愿
        engine.evaluate_and_update_mode("我想想想", RiskLevel.LOW)
        assert engine.current_mode == DialogueMode.SOCRATIC

        # 模拟连续回避 2 次
        engine.record_avoidance(True)
        engine.record_avoidance(True)

        # 触发退出
        engine.evaluate_and_update_mode("不想说", RiskLevel.LOW)
        assert engine.current_mode == DialogueMode.EMPATHY
        assert engine.depth_frozen_turns == 3

    def test_freeze_decrements_each_turn(self):
        """冻结轮次每轮递减"""
        engine = DialogEngine(user_id="test_user")
        engine.start(risk_level=RiskLevel.LOW)
        engine.current_state = DialogState.EXPLORE
        engine.consecutive_calm_turns = 5
        engine.depth_frozen_turns = 3

        # 模拟进入 SOCRATIC 模式（冻结期间）
        engine.evaluate_and_update_mode("我想想想", RiskLevel.LOW)
        # 冻结期间应限制为 SHALLOW
        assert engine.current_depth == SocraticDepth.SHALLOW
        assert engine.depth_frozen_turns == 2

    def test_freeze_prevents_deep(self):
        """冻结期间不能进入 DEEP"""
        engine = DialogEngine(user_id="test_user")
        engine.start(risk_level=RiskLevel.LOW)
        engine.current_state = DialogState.EXPLORE
        engine.consecutive_calm_turns = 10
        engine.consecutive_exploration_turns = 5
        engine.depth_frozen_turns = 2

        engine.evaluate_and_update_mode("我想想想", RiskLevel.LOW)
        # 冻结期间不能进入 DEEP
        assert engine.current_depth == SocraticDepth.SHALLOW


# ============================================================
# 4. CBT 记录生成测试
# ============================================================

class TestCBTRecordGeneration:
    """测试 CBT 认知重构记录生成"""

    def setup_method(self):
        """每个测试前清空记录"""
        clear_cbt_records()

    def test_complete_record(self):
        """完整走过 DEEP 流程 → COMPLETE"""
        tracker = CBTRecordTracker(user_id="user_123", conversation_id="conv_456")

        # 模拟完整流程
        tracker.update_depth_reached("SHALLOW")
        tracker.increment_socratic_turns()
        tracker.record_automatic_thought("我觉得没人喜欢我")
        tracker.record_distortion_type("读心术")

        tracker.update_depth_reached("MEDIUM")
        tracker.increment_socratic_turns()
        tracker.record_evidence(["朋友没回消息"], ["朋友可能在忙"])
        tracker.record_alternative_thought("朋友可能只是没看到")

        tracker.update_depth_reached("DEEP")
        tracker.increment_socratic_turns()
        tracker.record_emotion_intensity(initial=8, final=4)
        tracker.record_action_plan("下次主动问朋友")

        record = tracker.generate_record()
        assert record.status == CBTRecordStatus.COMPLETE
        assert record.depth_reached == "DEEP"
        assert record.turns_in_socratic == 3
        assert record.automatic_thought == "我觉得没人喜欢我"
        assert "改善" in record.summary

    def test_partial_record(self):
        """部分完成（未到 DEEP）→ PARTIAL"""
        tracker = CBTRecordTracker(user_id="user_123", conversation_id="conv_789")

        tracker.update_depth_reached("SHALLOW")
        tracker.increment_socratic_turns()
        tracker.record_automatic_thought("我肯定会失败")

        tracker.update_depth_reached("MEDIUM")
        tracker.increment_socratic_turns()

        record = tracker.generate_record()
        assert record.status == CBTRecordStatus.PARTIAL
        assert record.depth_reached == "MEDIUM"

    def test_not_started_record(self):
        """未进入苏格拉底模式 → NOT_STARTED"""
        tracker = CBTRecordTracker(user_id="user_123", conversation_id="conv_000")
        record = tracker.generate_record()
        assert record.status == CBTRecordStatus.NOT_STARTED
        assert "未进入" in record.summary

    def test_store_and_retrieve_record(self):
        """存储和获取 CBT 记录"""
        tracker = CBTRecordTracker(user_id="user_123", conversation_id="conv_456")
        tracker.update_depth_reached("DEEP")
        tracker.increment_socratic_turns()
        tracker.increment_socratic_turns()
        tracker.increment_socratic_turns()

        record = tracker.generate_record()
        store_cbt_record(record)

        # 获取记录
        retrieved = get_cbt_record("conv_456")
        assert retrieved is not None
        assert retrieved.status == CBTRecordStatus.COMPLETE

        # 获取用户所有记录
        user_records = get_user_cbt_records("user_123")
        assert len(user_records) == 1


# ============================================================
# 5. DialogEngine 深度追踪测试
# ============================================================

class TestDialogEngineDepthTracking:
    """测试 DialogEngine 深度追踪"""

    def test_engine_tracks_depth(self):
        """引擎追踪当前深度"""
        engine = DialogEngine(user_id="test_user")
        engine.start(risk_level=RiskLevel.LOW)
        assert engine.current_depth == SocraticDepth.SHALLOW

    def test_engine_tracks_exploration_turns(self):
        """引擎追踪连续探索轮次"""
        engine = DialogEngine(user_id="test_user")
        engine.start(risk_level=RiskLevel.LOW)
        engine.current_state = DialogState.EXPLORE
        engine.consecutive_calm_turns = 5

        # 第一次探索
        engine.evaluate_and_update_mode("我想想想", RiskLevel.LOW)
        assert engine.consecutive_exploration_turns == 1

        # 第二次探索
        engine.evaluate_and_update_mode("为什么", RiskLevel.LOW)
        assert engine.consecutive_exploration_turns == 2

    def test_engine_resets_exploration_on_non_explore(self):
        """非探索输入重置连续探索轮次"""
        engine = DialogEngine(user_id="test_user")
        engine.start(risk_level=RiskLevel.LOW)
        engine.current_state = DialogState.EXPLORE
        engine.consecutive_calm_turns = 5

        engine.evaluate_and_update_mode("我想想想", RiskLevel.LOW)
        assert engine.consecutive_exploration_turns == 1

        # 非探索输入
        engine.evaluate_and_update_mode("今天天气不错", RiskLevel.LOW)
        assert engine.consecutive_exploration_turns == 0

    def test_state_summary_includes_depth(self):
        """状态摘要包含深度信息"""
        engine = DialogEngine(user_id="test_user")
        engine.start(risk_level=RiskLevel.LOW)
        summary = engine.get_state_summary()

        assert "current_depth" in summary
        assert "consecutive_exploration_turns" in summary
        assert "depth_frozen_turns" in summary


# ============================================================
# 6. 辅助函数测试
# ============================================================

class TestHelperFunctions:
    """测试辅助函数"""

    def test_has_exploration_intent(self):
        """检测探索意愿"""
        assert has_exploration_intent("我想想想") is True
        assert has_exploration_intent("为什么") is True
        assert has_exploration_intent("帮我分析") is True
        assert has_exploration_intent("不想说") is False
        assert has_exploration_intent("") is False

    def test_is_avoidance(self):
        """检测回避"""
        assert is_avoidance("不想说") is True
        assert is_avoidance("不知道") is True
        assert is_avoidance("别问了") is True
        assert is_avoidance("我想想想") is False
        assert is_avoidance("") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
