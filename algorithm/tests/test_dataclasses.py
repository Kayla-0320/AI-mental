"""
dataclasses 模块单元测试

验证所有数据类的实例化、字段类型和约束。
"""
import time
import unittest
from dataclasses import fields

from shared.dataclasses import (
    CrisisAlert,
    EmotionResult,
    EvidenceItem,
    FeatureSummary,
    RiskAssessment,
    RiskLevel,
)


class TestRiskLevel(unittest.TestCase):
    """RiskLevel 枚举测试"""

    def test_enum_values(self):
        """枚举值必须为 low / medium / high / crisis"""
        self.assertEqual(RiskLevel.LOW, "low")
        self.assertEqual(RiskLevel.MEDIUM, "medium")
        self.assertEqual(RiskLevel.HIGH, "high")
        self.assertEqual(RiskLevel.CRISIS, "crisis")

    def test_enum_count(self):
        """枚举成员数量必须为 4"""
        self.assertEqual(len(RiskLevel), 4)

    def test_enum_is_str(self):
        """枚举值必须是 str 子类"""
        for member in RiskLevel:
            self.assertIsInstance(member, str)


class TestEvidenceItem(unittest.TestCase):
    """EvidenceItem 数据类测试"""

    def setUp(self):
        self.item = EvidenceItem(
            source="perception.text",
            description="检测到焦虑关键词",
            weight=0.85,
        )

    def test_field_types(self):
        """字段类型必须为 str, str, float"""
        self.assertIsInstance(self.item.source, str)
        self.assertIsInstance(self.item.description, str)
        self.assertIsInstance(self.item.weight, float)

    def test_field_count(self):
        """字段数量必须为 3"""
        self.assertEqual(len(fields(EvidenceItem)), 3)


class TestEmotionResult(unittest.TestCase):
    """EmotionResult 数据类测试"""

    def setUp(self):
        self.result = EmotionResult(
            text_emotion_probs=[0.1, 0.3, 0.4, 0.1, 0.1],
            audio_risk_prob=0.6,
            confidence=0.88,
            timestamp=time.time(),
            evidence=["文本含焦虑关键词", "语速偏快"],
        )

    def test_text_emotion_probs_length(self):
        """text_emotion_probs 必须为 5 维"""
        self.assertEqual(len(self.result.text_emotion_probs), 5)

    def test_text_emotion_probs_type(self):
        """text_emotion_probs 元素必须为 float"""
        for p in self.result.text_emotion_probs:
            self.assertIsInstance(p, float)

    def test_audio_risk_prob_optional(self):
        """audio_risk_prob 允许为 None"""
        result_no_audio = EmotionResult(
            text_emotion_probs=[0.2, 0.2, 0.2, 0.2, 0.2],
            audio_risk_prob=None,
            confidence=0.7,
            timestamp=time.time(),
            evidence=[],
        )
        self.assertIsNone(result_no_audio.audio_risk_prob)

    def test_audio_risk_prob_type(self):
        """audio_risk_prob 必须为 float 或 None"""
        self.assertIsInstance(self.result.audio_risk_prob, (float, type(None)))

    def test_confidence_type(self):
        """confidence 必须为 float"""
        self.assertIsInstance(self.result.confidence, float)

    def test_timestamp_type(self):
        """timestamp 必须为 float"""
        self.assertIsInstance(self.result.timestamp, float)

    def test_evidence_type(self):
        """evidence 必须为 list[str]"""
        self.assertIsInstance(self.result.evidence, list)
        for e in self.result.evidence:
            self.assertIsInstance(e, str)

    def test_field_count(self):
        """字段数量必须为 5"""
        self.assertEqual(len(fields(EmotionResult)), 5)


class TestRiskAssessment(unittest.TestCase):
    """RiskAssessment 数据类测试"""

    def setUp(self):
        self.assessment = RiskAssessment(
            phq9_estimated=(10.0, 15.0),
            gad7_estimated=(8.0, 12.0),
            risk_level=RiskLevel.MEDIUM,
            confidence=0.82,
            evidence=[
                EvidenceItem("assessment.scale", "PHQ-9 预估中高分", 0.9),
                EvidenceItem("perception.behavior", "近期活跃度下降", 0.6),
            ],
        )

    def test_phq9_estimated_type(self):
        """phq9_estimated 必须为 tuple[float, float]"""
        self.assertIsInstance(self.assessment.phq9_estimated, tuple)
        self.assertEqual(len(self.assessment.phq9_estimated), 2)
        for v in self.assessment.phq9_estimated:
            self.assertIsInstance(v, float)

    def test_gad7_estimated_type(self):
        """gad7_estimated 必须为 tuple[float, float]"""
        self.assertIsInstance(self.assessment.gad7_estimated, tuple)
        self.assertEqual(len(self.assessment.gad7_estimated), 2)

    def test_risk_level_enum(self):
        """risk_level 必须为 RiskLevel 枚举"""
        self.assertIsInstance(self.assessment.risk_level, RiskLevel)

    def test_risk_level_all_values(self):
        """risk_level 支持所有枚举值"""
        for level in RiskLevel:
            a = RiskAssessment(
                phq9_estimated=(0.0, 5.0),
                gad7_estimated=(0.0, 5.0),
                risk_level=level,
                confidence=0.5,
                evidence=[],
            )
            self.assertEqual(a.risk_level, level)

    def test_confidence_type(self):
        """confidence 必须为 float"""
        self.assertIsInstance(self.assessment.confidence, float)

    def test_evidence_type(self):
        """evidence 必须为 list[EvidenceItem]"""
        self.assertIsInstance(self.assessment.evidence, list)
        for e in self.assessment.evidence:
            self.assertIsInstance(e, EvidenceItem)

    def test_field_count(self):
        """字段数量必须为 5"""
        self.assertEqual(len(fields(RiskAssessment)), 5)


class TestCrisisAlert(unittest.TestCase):
    """CrisisAlert 数据类测试"""

    def setUp(self):
        self.alert = CrisisAlert(
            user_id="user_abc123",
            risk_score=0.92,
            trigger_evidence=["检测到自伤关键词", "情绪急剧恶化"],
            timestamp=time.time(),
            recommended_action="立即通知监护人并推送危机热线",
        )

    def test_user_id_type(self):
        """user_id 必须为 str"""
        self.assertIsInstance(self.alert.user_id, str)

    def test_risk_score_type(self):
        """risk_score 必须为 float"""
        self.assertIsInstance(self.alert.risk_score, float)

    def test_trigger_evidence_type(self):
        """trigger_evidence 必须为 list[str]"""
        self.assertIsInstance(self.alert.trigger_evidence, list)
        for e in self.alert.trigger_evidence:
            self.assertIsInstance(e, str)

    def test_timestamp_type(self):
        """timestamp 必须为 float"""
        self.assertIsInstance(self.alert.timestamp, float)

    def test_recommended_action_type(self):
        """recommended_action 必须为 str"""
        self.assertIsInstance(self.alert.recommended_action, str)

    def test_field_count(self):
        """字段数量必须为 5"""
        self.assertEqual(len(fields(CrisisAlert)), 5)


class TestFeatureSummary(unittest.TestCase):
    """FeatureSummary 数据类测试

    注意：此数据类确保不包含原始文本或可识别信息。
    """

    def setUp(self):
        self.summary = FeatureSummary(
            text_emotion_probs=[0.15, 0.25, 0.35, 0.10, 0.15],
            audio_risk_prob=None,
            confidence=0.91,
            session_id="session_xyz789",
            timestamp=time.time(),
        )

    def test_text_emotion_probs_type(self):
        """text_emotion_probs 必须为 list[float]"""
        self.assertIsInstance(self.summary.text_emotion_probs, list)
        for p in self.summary.text_emotion_probs:
            self.assertIsInstance(p, float)

    def test_audio_risk_prob_optional(self):
        """audio_risk_prob 允许为 None"""
        self.assertIsNone(self.summary.audio_risk_prob)

    def test_audio_risk_prob_with_value(self):
        """audio_risk_prob 也可为 float"""
        s = FeatureSummary(
            text_emotion_probs=[0.2, 0.2, 0.2, 0.2, 0.2],
            audio_risk_prob=0.45,
            confidence=0.8,
            session_id="s1",
            timestamp=time.time(),
        )
        self.assertIsInstance(s.audio_risk_prob, float)

    def test_confidence_type(self):
        """confidence 必须为 float"""
        self.assertIsInstance(self.summary.confidence, float)

    def test_session_id_type(self):
        """session_id 必须为 str"""
        self.assertIsInstance(self.summary.session_id, str)

    def test_timestamp_type(self):
        """timestamp 必须为 float"""
        self.assertIsInstance(self.summary.timestamp, float)

    def test_no_raw_text_fields(self):
        """确保数据类不包含原始文本或可识别信息字段"""
        field_names = {f.name for f in fields(FeatureSummary)}
        # 不应包含任何可能携带原始信息的字段
        sensitive_names = {"text", "content", "message", "user_name", "nickname"}
        self.assertTrue(field_names.isdisjoint(sensitive_names))

    def test_field_count(self):
        """字段数量必须为 5"""
        self.assertEqual(len(fields(FeatureSummary)), 5)


if __name__ == "__main__":
    unittest.main()
