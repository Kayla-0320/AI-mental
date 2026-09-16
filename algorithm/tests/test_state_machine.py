"""
对话状态机单元测试

覆盖：
- 状态定义完整性
- 状态转移规则（7 条规则）
- 动作选择逻辑
- Action 结构化对象
- CRISIS 状态升级触发
- 禁止诊断性语言
- DialogEngine 端到端流程
"""
from __future__ import annotations

import pytest

from intervention.state_machine import (
    Action,
    ActionType,
    DialogEngine,
    DialogState,
    MERMAID_STATE_DIAGRAM,
    STATE_ALLOWED_ACTIONS,
    create_crisis_alert,
    select_action,
    transition,
)
from shared.dataclasses import (
    CrisisAlert,
    EvidenceItem,
    RiskAssessment,
    RiskLevel,
)


# ============================================================
# 状态定义测试
# ============================================================

class TestDialogState:
    def test_five_states_defined(self):
        """五个状态全部定义"""
        assert len(DialogState) == 5

    def test_state_names(self):
        """状态名称正确"""
        names = {s.value for s in DialogState}
        assert names == {"INIT", "EXPLORE", "INTERVENE", "CRISIS", "CLOSE"}


class TestActionType:
    def test_action_types_defined(self):
        """动作类型完整"""
        expected = {
            "OPEN_QUESTION", "EMOTION_REFLECTION", "CBT_GUIDE",
            "MINDFULNESS_GUIDE", "SAFETY_CHECK", "RESOURCE_PROVIDE", "SUMMARY",
        }
        actual = {a.value for a in ActionType}
        assert actual == expected


class TestAction:
    def test_action_is_dataclass(self):
        """Action 是结构化对象"""
        action = Action(
            action_type=ActionType.OPEN_QUESTION,
            params={"topic": "test"},
            state=DialogState.INIT,
        )
        assert action.action_type == ActionType.OPEN_QUESTION
        assert action.params == {"topic": "test"}
        assert action.state == DialogState.INIT

    def test_action_no_dialogue_text(self):
        """Action 不包含具体话术"""
        action = Action(action_type=ActionType.CBT_GUIDE)
        # 确保没有 text/dialogue/response 等字段
        assert not hasattr(action, "text")
        assert not hasattr(action, "dialogue")
        assert not hasattr(action, "response")

    def test_forbidden_patterns_present(self):
        """Action 包含禁止模式"""
        action = Action(action_type=ActionType.OPEN_QUESTION)
        assert len(action.forbidden_patterns) > 0
        assert "诊断" in action.forbidden_patterns
        assert "患有" in action.forbidden_patterns


# ============================================================
# 状态转移测试
# ============================================================

class TestTransition:
    # 规则 1：任何状态 + CRISIS 风险 → CRISIS 状态
    @pytest.mark.parametrize("state", list(DialogState))
    def test_crisis_risk_always_to_crisis(self, state):
        """规则 1：任何状态下 CRISIS 风险 → CRISIS 状态"""
        if state == DialogState.CLOSE:
            pytest.skip("CLOSE 是终态")
        result = transition(state, RiskLevel.CRISIS, turn_count=5)
        assert result == DialogState.CRISIS

    # 规则 2：HIGH + worsening → CRISIS
    @pytest.mark.parametrize("state", [DialogState.INIT, DialogState.EXPLORE, DialogState.INTERVENE])
    def test_high_worsening_to_crisis(self, state):
        """规则 2：HIGH 风险 + 恶化趋势 → CRISIS"""
        result = transition(state, RiskLevel.HIGH, turn_count=5, emotion_trend="worsening")
        assert result == DialogState.CRISIS

    # 规则 3：CRISIS + LOW → CLOSE
    def test_crisis_low_to_close(self):
        """规则 3：CRISIS + 风险降至 LOW → CLOSE"""
        result = transition(DialogState.CRISIS, RiskLevel.LOW, turn_count=10)
        assert result == DialogState.CLOSE

    def test_crisis_medium_stays(self):
        """规则 3：CRISIS + MEDIUM → 保持 CRISIS"""
        result = transition(DialogState.CRISIS, RiskLevel.MEDIUM, turn_count=10)
        assert result == DialogState.CRISIS

    # 规则 4：INIT → EXPLORE
    def test_init_to_explore_after_2_turns(self):
        """规则 4：INIT + turn >= 2 → EXPLORE"""
        result = transition(DialogState.INIT, RiskLevel.LOW, turn_count=2)
        assert result == DialogState.EXPLORE

    def test_init_stays_before_2_turns(self):
        """规则 4：INIT + turn < 2 → 保持 INIT"""
        result = transition(DialogState.INIT, RiskLevel.LOW, turn_count=1)
        assert result == DialogState.INIT

    # 规则 5：EXPLORE → INTERVENE
    def test_explore_to_intervene_low_risk(self):
        """规则 5：EXPLORE + LOW + 足够轮次 → INTERVENE"""
        result = transition(DialogState.EXPLORE, RiskLevel.LOW, turn_count=3)
        assert result == DialogState.INTERVENE

    def test_explore_to_intervene_high_risk(self):
        """规则 5：EXPLORE + HIGH + 足够轮次 → INTERVENE"""
        result = transition(DialogState.EXPLORE, RiskLevel.HIGH, turn_count=3)
        assert result == DialogState.INTERVENE

    def test_explore_forced_intervene_after_8(self):
        """规则 5：EXPLORE + turn >= 8 → 强制 INTERVENE"""
        result = transition(DialogState.EXPLORE, RiskLevel.MEDIUM, turn_count=8)
        assert result == DialogState.INTERVENE

    def test_explore_stays_insufficient_turns(self):
        """规则 5：EXPLORE + 轮次不足 → 保持 EXPLORE"""
        result = transition(DialogState.EXPLORE, RiskLevel.LOW, turn_count=1)
        assert result == DialogState.EXPLORE

    # 规则 6：INTERVENE → CLOSE / EXPLORE
    def test_intervene_to_close_low_improving(self):
        """规则 6：INTERVENE + LOW + improving → CLOSE"""
        result = transition(DialogState.INTERVENE, RiskLevel.LOW, turn_count=10, emotion_trend="improving")
        assert result == DialogState.CLOSE

    def test_intervene_to_explore_high_risk(self):
        """规则 6：INTERVENE + HIGH → EXPLORE（风险回升）"""
        result = transition(DialogState.INTERVENE, RiskLevel.HIGH, turn_count=10)
        assert result == DialogState.EXPLORE

    def test_intervene_forced_close_after_15(self):
        """规则 6：INTERVENE + turn >= 15 → 强制 CLOSE"""
        result = transition(DialogState.INTERVENE, RiskLevel.MEDIUM, turn_count=15)
        assert result == DialogState.CLOSE

    # 规则 7：CLOSE 是终态
    def test_close_is_terminal(self):
        """规则 7：CLOSE → CLOSE（终态）"""
        result = transition(DialogState.CLOSE, RiskLevel.LOW, turn_count=20)
        assert result == DialogState.CLOSE


# ============================================================
# 动作选择测试
# ============================================================

class TestSelectAction:
    def test_init_first_turn_open_question(self):
        """INIT 首轮 → OPEN_QUESTION"""
        action = select_action(DialogState.INIT, turn_count=0)
        assert action.action_type == ActionType.OPEN_QUESTION
        assert action.state == DialogState.INIT

    def test_init_later_turn_emotion_reflection(self):
        """INIT 后续轮 → EMOTION_REFLECTION"""
        action = select_action(DialogState.INIT, turn_count=1)
        assert action.action_type == ActionType.EMOTION_REFLECTION

    def test_explore_alternates(self):
        """EXPLORE 交替提问和反映"""
        action_even = select_action(DialogState.EXPLORE, turn_count=2)
        action_odd = select_action(DialogState.EXPLORE, turn_count=3)
        assert action_even.action_type == ActionType.OPEN_QUESTION
        assert action_odd.action_type == ActionType.EMOTION_REFLECTION

    def test_intervene_high_risk_cbt(self):
        """INTERVENE + HIGH → CBT_GUIDE"""
        action = select_action(DialogState.INTERVENE, risk_level=RiskLevel.HIGH)
        assert action.action_type == ActionType.CBT_GUIDE

    def test_intervene_low_late_mindfulness(self):
        """INTERVENE + LOW + 后期 → MINDFULNESS_GUIDE"""
        action = select_action(DialogState.INTERVENE, turn_count=12, risk_level=RiskLevel.LOW)
        assert action.action_type == ActionType.MINDFULNESS_GUIDE

    def test_crisis_safety_check(self):
        """CRISIS → SAFETY_CHECK"""
        action = select_action(DialogState.CRISIS)
        assert action.action_type == ActionType.SAFETY_CHECK

    def test_crisis_requires_escalation(self):
        """CRISIS 动作必须 requires_escalation=True"""
        action = select_action(DialogState.CRISIS)
        assert action.requires_escalation is True

    def test_close_first_summary(self):
        """CLOSE 首轮 → SUMMARY"""
        action = select_action(DialogState.CLOSE, turn_count=0)
        assert action.action_type == ActionType.SUMMARY

    def test_close_later_resource(self):
        """CLOSE 后续 → RESOURCE_PROVIDE"""
        action = select_action(DialogState.CLOSE, turn_count=1)
        assert action.action_type == ActionType.RESOURCE_PROVIDE


# ============================================================
# 升级层接口测试
# ============================================================

class TestEscalation:
    def test_create_crisis_alert(self):
        """创建危机警报"""
        assessment = RiskAssessment(
            phq9_estimated=(15.0, 19.0),
            gad7_estimated=(12.0, 16.0),
            risk_level=RiskLevel.CRISIS,
            confidence=0.9,
            evidence=[
                EvidenceItem(source="test", description="测试证据", weight=0.9),
            ],
        )
        alert = create_crisis_alert("user_123", assessment)
        assert isinstance(alert, CrisisAlert)
        assert alert.user_id == "user_123"
        assert alert.risk_score == 1.0  # CRISIS → 1.0
        assert len(alert.trigger_evidence) >= 2


# ============================================================
# DialogEngine 端到端测试
# ============================================================

class TestDialogEngine:
    def test_start_init_state(self):
        """启动后处于 INIT 状态"""
        engine = DialogEngine("user_1")
        action = engine.start()
        assert engine.current_state == DialogState.INIT
        assert action.action_type == ActionType.OPEN_QUESTION

    def test_start_crisis_risk(self):
        """启动时就是 CRISIS 风险 → 直接进入 CRISIS"""
        engine = DialogEngine("user_1")
        action = engine.start(risk_level=RiskLevel.CRISIS)
        assert engine.current_state == DialogState.CRISIS
        assert action.requires_escalation is True

    def test_full_flow_normal(self):
        """正常流程：INIT → EXPLORE → INTERVENE → CLOSE"""
        engine = DialogEngine("user_1")
        engine.start(risk_level=RiskLevel.LOW)

        # 轮次 0：INIT（turn_count=0 → 1）
        engine.next_turn(RiskLevel.LOW)
        assert engine.current_state == DialogState.INIT

        # 轮次 1：INIT（turn_count=1 → 2）
        engine.next_turn(RiskLevel.LOW)
        assert engine.current_state == DialogState.INIT

        # 轮次 2：INIT → EXPLORE（turn_count=2 满足转移条件）
        engine.next_turn(RiskLevel.LOW)
        assert engine.current_state == DialogState.EXPLORE

        # 轮次 3：EXPLORE → INTERVENE（探索充分）
        engine.next_turn(RiskLevel.LOW)
        assert engine.current_state == DialogState.INTERVENE

        # 干预 + 改善 → CLOSE
        engine.next_turn(RiskLevel.LOW, emotion_trend="improving")
        assert engine.current_state == DialogState.CLOSE
        assert engine.is_terminal()

    def test_crisis_escalation_flow(self):
        """危机流程：触发升级层"""
        engine = DialogEngine("user_1")
        engine.start(risk_level=RiskLevel.LOW)

        # 风险突然升至 CRISIS
        action = engine.next_turn(RiskLevel.CRISIS)
        assert engine.current_state == DialogState.CRISIS
        assert action.requires_escalation is True

        # 触发升级层
        assessment = RiskAssessment(
            phq9_estimated=(20.0, 27.0),
            gad7_estimated=(15.0, 21.0),
            risk_level=RiskLevel.CRISIS,
            confidence=0.95,
            evidence=[],
        )
        alert = engine.trigger_escalation(assessment)
        assert isinstance(alert, CrisisAlert)
        assert alert.risk_score == 1.0

    def test_state_history_tracked(self):
        """状态历史被记录"""
        engine = DialogEngine("user_1")
        engine.start(risk_level=RiskLevel.LOW)
        engine.next_turn(RiskLevel.LOW)  # turn=0, INIT
        engine.next_turn(RiskLevel.LOW)  # turn=1, INIT
        engine.next_turn(RiskLevel.LOW)  # turn=2, INIT → EXPLORE
        engine.next_turn(RiskLevel.LOW)  # turn=3, EXPLORE → INTERVENE

        summary = engine.get_state_summary()
        # 至少 INIT → EXPLORE → INTERVENE 三个状态
        assert len(summary["state_history"]) >= 3


# ============================================================
# Mermaid 图测试
# ============================================================

class TestMermaidDiagram:
    def test_mermaid_diagram_exists(self):
        """Mermaid 图存在"""
        assert MERMAID_STATE_DIAGRAM is not None
        assert "stateDiagram-v2" in MERMAID_STATE_DIAGRAM

    def test_mermaid_contains_all_states(self):
        """Mermaid 图包含所有状态"""
        for state in DialogState:
            assert state.value in MERMAID_STATE_DIAGRAM
