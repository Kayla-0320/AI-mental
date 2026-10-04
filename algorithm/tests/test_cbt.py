"""
CBT 认知重构模块测试
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intervention.cbt.cognitive_restructuring import (
    CBTGuide,
    CBTStep,
    CBTSession,
    CognitiveDistortion,
    DISTORTION_DESCRIPTIONS,
)


class TestCBTStep:
    """测试 CBT 步骤枚举"""

    def test_all_steps_exist(self):
        assert CBTStep.IDENTIFY_THOUGHT == "identify_thought"
        assert CBTStep.IDENTIFY_DISTORTION == "identify_distortion"
        assert CBTStep.EXAMINE_EVIDENCE == "examine_evidence"
        assert CBTStep.ALTERNATIVE_THOUGHT == "alternative_thought"
        assert CBTStep.EVALUATE_AND_PLAN == "evaluate_and_plan"
        assert CBTStep.COMPLETED == "completed"
        assert CBTStep.EXITED == "exited"

    def test_step_count(self):
        assert len(CBTStep) == 7


class TestCognitiveDistortion:
    """测试认知扭曲类型"""

    def test_all_distortions_have_descriptions(self):
        for dist in CognitiveDistortion:
            assert dist in DISTORTION_DESCRIPTIONS
            info = DISTORTION_DESCRIPTIONS[dist]
            assert "name" in info
            assert "description" in info
            assert "challenge" in info

    def test_distortion_count(self):
        assert len(CognitiveDistortion) == 10


class TestCBTSession:
    """测试 CBT 会话"""

    def test_initial_state(self):
        guide = CBTGuide()
        session = guide.start("user_001")
        assert session.user_id == "user_001"
        assert session.current_step == CBTStep.IDENTIFY_THOUGHT
        assert session.automatic_thought == ""
        assert session.exited is False

    def test_next_step_progression(self):
        guide = CBTGuide()
        session = guide.start("user_001")

        # 步骤 1 → 2
        session = guide.next_step(session)
        assert session.current_step == CBTStep.IDENTIFY_DISTORTION

        # 步骤 2 → 3
        session = guide.next_step(session)
        assert session.current_step == CBTStep.EXAMINE_EVIDENCE

        # 步骤 3 → 4
        session = guide.next_step(session)
        assert session.current_step == CBTStep.ALTERNATIVE_THOUGHT

        # 步骤 4 → 5
        session = guide.next_step(session)
        assert session.current_step == CBTStep.EVALUATE_AND_PLAN

        # 步骤 5 → 完成
        session = guide.next_step(session)
        assert session.current_step == CBTStep.COMPLETED

    def test_exit_session(self):
        guide = CBTGuide()
        session = guide.start("user_001")
        session = guide.exit_session(session, "用户不想继续")
        assert session.exited is True
        assert session.exit_reason == "用户不想继续"
        assert session.current_step == CBTStep.EXITED

    def test_exit_prevents_progression(self):
        guide = CBTGuide()
        session = guide.start("user_001")
        session = guide.exit_session(session)
        # 退出后不应再推进
        session = guide.next_step(session)
        assert session.current_step == CBTStep.EXITED


class TestCBTDataRecording:
    """测试数据记录"""

    def test_record_thought(self):
        guide = CBTGuide()
        session = guide.start("user_001")
        guide.record_thought(session, "我觉得没人喜欢我")
        assert session.automatic_thought == "我觉得没人喜欢我"

    def test_record_distortion(self):
        guide = CBTGuide()
        session = guide.start("user_001")
        guide.record_distortion(session, CognitiveDistortion.MIND_READING)
        assert session.identified_distortion == CognitiveDistortion.MIND_READING

    def test_record_evidence(self):
        guide = CBTGuide()
        session = guide.start("user_001")
        guide.record_evidence(
            session,
            supporting=["上次确实被忽视了"],
            contradicting=["但昨天朋友还约我吃饭"],
        )
        assert len(session.supporting_evidence) == 1
        assert len(session.contradicting_evidence) == 1

    def test_record_emotion_intensity_clamped(self):
        guide = CBTGuide()
        session = guide.start("user_001")
        guide.record_emotion_intensity(session, initial=15, final=-3)
        assert session.initial_emotion_intensity == 10
        assert session.final_emotion_intensity == 0


class TestCBTGuidance:
    """测试引导话术生成"""

    def test_guidance_for_each_step(self):
        guide = CBTGuide()
        session = guide.start("user_001")

        steps = [
            CBTStep.IDENTIFY_THOUGHT,
            CBTStep.IDENTIFY_DISTORTION,
            CBTStep.EXAMINE_EVIDENCE,
            CBTStep.ALTERNATIVE_THOUGHT,
            CBTStep.EVALUATE_AND_PLAN,
            CBTStep.COMPLETED,
            CBTStep.EXITED,
        ]

        for step in steps:
            session.current_step = step
            guidance = guide.get_guidance(session)
            assert guidance.step == step
            assert len(guidance.prompt) > 0

    def test_identify_thought_guidance_has_examples(self):
        guide = CBTGuide()
        session = guide.start("user_001")
        guidance = guide.get_guidance(session)
        assert len(guidance.examples) > 0
        assert len(guidance.follow_up_questions) > 0

    def test_completed_guidance_is_terminal(self):
        guide = CBTGuide()
        session = guide.start("user_001")
        session.current_step = CBTStep.COMPLETED
        guidance = guide.get_guidance(session)
        assert guidance.is_terminal is True

    def test_exited_guidance_is_positive(self):
        guide = CBTGuide()
        session = guide.start("user_001")
        session.current_step = CBTStep.EXITED
        guidance = guide.get_guidance(session)
        assert "没关系" in guidance.prompt or "勇敢" in guidance.prompt


class TestFullCBTSession:
    """测试完整 CBT 会话流程"""

    def test_run_full_session(self):
        guide = CBTGuide()
        session = guide.run_full_session(
            user_id="user_test",
            automatic_thought="我肯定考不好",
            distortion=CognitiveDistortion.FORTUNE_TELLING,
            supporting=["上次没复习好"],
            contradicting=["但这次复习了两周"],
            alternative="我不确定结果，但我可以尽力",
            initial_emotion=8,
            final_emotion=4,
            action_plan="明天再做一套模拟题",
        )

        assert session.current_step == CBTStep.COMPLETED
        assert session.automatic_thought == "我肯定考不好"
        assert session.identified_distortion == CognitiveDistortion.FORTUNE_TELLING
        assert session.alternative_thought == "我不确定结果，但我可以尽力"
        assert session.initial_emotion_intensity == 8
        assert session.final_emotion_intensity == 4
        assert session.action_plan == "明天再做一套模拟题"
        assert session.exited is False

    def test_full_session_emotion_improved(self):
        guide = CBTGuide()
        session = guide.run_full_session(
            user_id="user_test",
            automatic_thought="没人喜欢我",
            initial_emotion=9,
            final_emotion=5,
        )
        assert session.final_emotion_intensity < session.initial_emotion_intensity
