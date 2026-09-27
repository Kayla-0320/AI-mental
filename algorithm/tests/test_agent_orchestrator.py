"""Agent 编排器单元测试"""
import time
from intervention.agent_orchestrator import (
    AgentOrchestrator,
    SessionManager,
    SelfAssessment,
    MoodDiary,
    MoodEntry,
    AssessmentResult,
)
from shared.dataclasses import EmotionResult, RiskLevel


def _make_emotion(probs: list[float]) -> EmotionResult:
    return EmotionResult(
        text_emotion_probs=probs,
        audio_risk_prob=None,
        confidence=0.8,
        timestamp=time.time(),
        evidence=["test"],
    )


def test_session_manager_init():
    sm = SessionManager(user_id="test_user")
    assert sm.user_id == "test_user"
    assert sm.turn_count == 0
    assert len(sm.emotion_history) == 0


def test_session_manager_record():
    sm = SessionManager(user_id="test_user")
    sm.record_emotion(_make_emotion([0.1, 0.5, 0.2, 0.1, 0.1]))
    sm.record_risk(RiskLevel.MEDIUM)
    sm.increment_turn()
    assert sm.turn_count == 1
    assert len(sm.emotion_history) == 1
    assert sm.risk_history[0] == RiskLevel.MEDIUM


def test_session_summary():
    sm = SessionManager(user_id="test_user")
    for probs, risk in [
        ([0.6, 0.1, 0.1, 0.1, 0.1], RiskLevel.LOW),
        ([0.1, 0.5, 0.2, 0.1, 0.1], RiskLevel.MEDIUM),
        ([0.1, 0.6, 0.1, 0.1, 0.1], RiskLevel.MEDIUM),
    ]:
        sm.record_emotion(_make_emotion(probs))
        sm.record_risk(risk)
        sm.increment_turn()
    summary = sm.get_session_summary()
    assert "test_user" in summary
    assert "3" in summary


def test_mood_diary_trend():
    diary = MoodDiary()
    diary.record([0.6, 0.1, 0.1, 0.1, 0.1], turn_index=0)
    diary.record([0.5, 0.1, 0.1, 0.1, 0.2], turn_index=1)
    diary.record([0.7, 0.05, 0.05, 0.05, 0.15], turn_index=2)
    diary.record([0.8, 0.05, 0.05, 0.05, 0.05], turn_index=3)
    trend = diary.get_trend()
    assert trend in ("improving", "stable", "worsening")


def test_mood_diary_arc_data():
    diary = MoodDiary()
    diary.record([0.5, 0.2, 0.1, 0.1, 0.1], turn_index=0)
    diary.record([0.3, 0.3, 0.2, 0.1, 0.1], turn_index=1)
    arc = diary.get_arc_data()
    assert len(arc) == 2
    assert "turn" in arc[0]
    assert "mood_score" in arc[0]
    assert "dominant_emotion" in arc[0]


def test_self_assessment_phq2():
    result = SelfAssessment.quick_phq2([2, 3])
    assert result.score == 5
    assert result.severity == "筛查阳性"
    assert result.scale_name == "PHQ-2"


def test_self_assessment_phq2_low():
    result = SelfAssessment.quick_phq2([0, 1])
    assert result.score == 1
    assert result.severity == "筛查阴性"


def test_self_assessment_gad2():
    result = SelfAssessment.quick_gad2([3, 2])
    assert result.score == 5
    assert result.severity == "筛查阳性"
    assert result.scale_name == "GAD-2"


def test_self_assessment_mood_checkin_low():
    result = SelfAssessment.mood_checkin(2)
    assert result.score == 2
    assert result.severity == "偏低"


def test_self_assessment_mood_checkin_high():
    result = SelfAssessment.mood_checkin(5)
    assert result.severity == "良好"


def test_orchestrator_init():
    orch = AgentOrchestrator()
    assert len(orch._sessions) == 0


def test_orchestrator_process_turn():
    orch = AgentOrchestrator()
    result = orch.process_turn(
        user_id="test_user",
        message="最近压力好大，感觉喘不过气来",
        conversation_history=[],
    )
    assert result.user_id == "test_user"
    assert result.turn_id != ""
    assert len(result.emotion_probs) == 5
    assert result.risk_level in ("low", "medium", "high", "crisis")
    assert result.dialog_state != ""
    assert result.dialog_mode in ("EMPATHY", "SOCRATIC")


def test_orchestrator_multi_turn():
    orch = AgentOrchestrator()
    r1 = orch.process_turn("u1", "你好", [])
    r2 = orch.process_turn("u1", "最近学习压力很大", [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": r1.reply},
    ])
    assert r2.turn_id != r1.turn_id
    sm = orch._sessions["u1"]
    assert sm.turn_count == 2


def test_orchestrator_session_summary():
    orch = AgentOrchestrator()
    orch.process_turn("u1", "最近心情不太好", [])
    summary = orch.get_session_summary("u1")
    assert "u1" in summary


def test_orchestrator_mood_diary():
    orch = AgentOrchestrator()
    orch.process_turn("u1", "今天很不开心", [])
    diary = orch.get_mood_diary("u1")
    assert isinstance(diary, dict)
    assert "entries_count" in diary or "total_entries" in diary or "avg_mood" in diary


def test_orchestrator_self_assessment():
    orch = AgentOrchestrator()
    result = orch.run_self_assessment("phq2", answers=[2, 2])
    assert result.score == 4
    assert result.severity == "筛查阳性"
