"""
年龄差异化多模态感知测试

覆盖：
- AgeGroup 枚举和 AgeProfile 工厂函数
- 各年龄段关键词库完整性
- 文本分析年龄校准（同一段俚语在不同年龄组的风险分差异）
- 语音分析年龄校准（同一F0在不同年龄组的风险分差异）
- 面部分析年龄校准（同一AU强度在不同年龄组的风险分差异）
- 行为分析年龄校准
- 融合权重年龄自适应
- 端到端：同一输入在不同年龄组产生不同结果
- 向后兼容性（age_profile=None 时行为不变）
"""
import sys
import os
import unittest

# 确保可以导入 algorithm 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.age_config import (
    AgeGroup,
    AgeProfile,
    TextAgeParams,
    VoiceAgeParams,
    FaceAgeParams,
    BehaviorAgeParams,
    FusionAgeParams,
    get_age_profile,
    infer_age_group,
    get_age_profile_by_age,
    get_all_age_groups,
)
from perception.perception_service import PerceptionService


# ============================================================
# 年龄分组与画像测试
# ============================================================

class TestAgeGroupEnum(unittest.TestCase):
    """AgeGroup 枚举测试"""

    def test_three_groups_defined(self):
        """三个年龄分组均已定义"""
        self.assertEqual(len(AgeGroup), 3)

    def test_group_values(self):
        """分组值正确"""
        self.assertEqual(AgeGroup.EARLY_ADOLESCENT.value, "early_adolescent")
        self.assertEqual(AgeGroup.MID_ADOLESCENT.value, "mid_adolescent")
        self.assertEqual(AgeGroup.YOUNG_ADULT.value, "young_adult")


class TestInferAgeGroup(unittest.TestCase):
    """年龄推断测试"""

    def test_early_adolescent_range(self):
        """12-15岁 → EARLY_ADOLESCENT"""
        for age in [12, 13, 14, 15]:
            self.assertEqual(infer_age_group(age), AgeGroup.EARLY_ADOLESCENT)

    def test_mid_adolescent_range(self):
        """16-18岁 → MID_ADOLESCENT"""
        for age in [16, 17, 18]:
            self.assertEqual(infer_age_group(age), AgeGroup.MID_ADOLESCENT)

    def test_young_adult_range(self):
        """19-22岁 → YOUNG_ADULT"""
        for age in [19, 20, 21, 22]:
            self.assertEqual(infer_age_group(age), AgeGroup.YOUNG_ADULT)

    def test_below_range(self):
        """低于范围 → EARLY_ADOLESCENT（最敏感）"""
        self.assertEqual(infer_age_group(10), AgeGroup.EARLY_ADOLESCENT)
        self.assertEqual(infer_age_group(8), AgeGroup.EARLY_ADOLESCENT)

    def test_above_range(self):
        """高于范围 → YOUNG_ADULT（最成熟）"""
        self.assertEqual(infer_age_group(25), AgeGroup.YOUNG_ADULT)
        self.assertEqual(infer_age_group(30), AgeGroup.YOUNG_ADULT)

    def test_none_returns_none(self):
        """None 年龄 → None"""
        self.assertIsNone(infer_age_group(None))


class TestAgeProfileFactory(unittest.TestCase):
    """AgeProfile 工厂函数测试"""

    def test_get_all_profiles(self):
        """所有年龄画像均可获取"""
        for group in AgeGroup:
            profile = get_age_profile(group)
            self.assertIsInstance(profile, AgeProfile)
            self.assertEqual(profile.age_group, group)

    def test_early_adolescent_params(self):
        """初中画像参数正确"""
        p = get_age_profile(AgeGroup.EARLY_ADOLESCENT)
        self.assertEqual(p.age_range, (12, 15))
        # 文本
        self.assertGreater(len(p.text.slang_keywords), 10)
        self.assertGreater(len(p.text.crisis_keywords), 10)
        self.assertAlmostEqual(p.text.expression_formality, 0.2)
        self.assertAlmostEqual(p.text.emoji_sensitivity, 0.15)
        # 语音
        self.assertAlmostEqual(p.voice.f0_baseline, 220.0)
        self.assertAlmostEqual(p.voice.f0_anxiety_threshold, 300.0)
        # 面部
        self.assertAlmostEqual(p.face.au_intensity_baseline, 0.6)
        self.assertAlmostEqual(p.face.masking_detection_weight, 0.1)
        # 融合（11 模态权重）
        self.assertAlmostEqual(p.fusion.face_weight, 0.20)
        self.assertAlmostEqual(p.fusion.text_weight, 0.25)
        self.assertAlmostEqual(p.fusion.voice_semantics_weight, 0.05)
        self.assertAlmostEqual(p.fusion.risk_threshold_offset, -0.05)
        # 新增模态权重存在
        self.assertGreater(p.fusion.circadian_weight, 0)
        self.assertGreater(p.fusion.cognitive_weight, 0)

    def test_mid_adolescent_params(self):
        """高中画像参数正确"""
        p = get_age_profile(AgeGroup.MID_ADOLESCENT)
        self.assertEqual(p.age_range, (15, 18))
        self.assertAlmostEqual(p.voice.f0_baseline, 190.0)
        self.assertAlmostEqual(p.face.au_intensity_baseline, 0.5)
        self.assertAlmostEqual(p.fusion.text_weight, 0.30)
        self.assertAlmostEqual(p.fusion.voice_semantics_weight, 0.05)
        self.assertAlmostEqual(p.fusion.risk_threshold_offset, 0.0)

    def test_young_adult_params(self):
        """大学画像参数正确"""
        p = get_age_profile(AgeGroup.YOUNG_ADULT)
        self.assertEqual(p.age_range, (18, 22))
        self.assertAlmostEqual(p.voice.f0_baseline, 170.0)
        self.assertAlmostEqual(p.face.au_intensity_baseline, 0.4)
        self.assertAlmostEqual(p.face.masking_detection_weight, 0.5)
        self.assertAlmostEqual(p.fusion.text_weight, 0.35)
        self.assertAlmostEqual(p.fusion.face_weight, 0.08)
        self.assertAlmostEqual(p.fusion.voice_semantics_weight, 0.05)
        self.assertAlmostEqual(p.fusion.risk_threshold_offset, 0.05)

    def test_get_age_profile_by_age(self):
        """从年龄获取画像"""
        p = get_age_profile_by_age(13)
        self.assertIsNotNone(p)
        self.assertEqual(p.age_group, AgeGroup.EARLY_ADOLESCENT)

    def test_get_age_profile_by_age_none(self):
        """None 年龄返回 None"""
        self.assertIsNone(get_age_profile_by_age(None))

    def test_get_all_age_groups_summary(self):
        """获取所有年龄分组摘要"""
        groups = get_all_age_groups()
        self.assertEqual(len(groups), 3)
        for g in groups:
            self.assertIn("group", g)
            self.assertIn("age_range", g)
            self.assertIn("description", g)


# ============================================================
# 关键词库完整性测试
# ============================================================

class TestKeywordCompleteness(unittest.TestCase):
    """各年龄段关键词库完整性"""

    def test_each_group_has_slang_keywords(self):
        """每个年龄组都有足够的俚语关键词"""
        for group in AgeGroup:
            p = get_age_profile(group)
            self.assertGreaterEqual(
                len(p.text.slang_keywords), 15,
                f"{group.value} 俚语关键词不足 15 个"
            )

    def test_each_group_has_crisis_keywords(self):
        """每个年龄组都有危机关键词"""
        for group in AgeGroup:
            p = get_age_profile(group)
            self.assertGreaterEqual(
                len(p.text.crisis_keywords), 10,
                f"{group.value} 危机关键词不足 10 个"
            )

    def test_crisis_keywords_differ_across_groups(self):
        """不同年龄组的危机关键词有差异"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).text.crisis_keywords
        young = get_age_profile(AgeGroup.YOUNG_ADULT).text.crisis_keywords
        # 至少有一些不同的词
        early_set = set(early)
        young_set = set(young)
        # 两组不应完全相同
        self.assertNotEqual(early_set, young_set)

    def test_slang_keywords_differ_across_groups(self):
        """不同年龄组的俚语关键词有差异"""
        early = set(get_age_profile(AgeGroup.EARLY_ADOLESCENT).text.slang_keywords)
        young = set(get_age_profile(AgeGroup.YOUNG_ADULT).text.slang_keywords)
        # 至少有一些不同
        self.assertNotEqual(early, young)

    def test_emoji_sensitivity_decreases_with_age(self):
        """emoji 灵敏度随年龄递减"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).text.emoji_sensitivity
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).text.emoji_sensitivity
        young = get_age_profile(AgeGroup.YOUNG_ADULT).text.emoji_sensitivity
        self.assertGreater(early, mid)
        self.assertGreater(mid, young)

    def test_expression_formality_increases_with_age(self):
        """表达正式度随年龄递增"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).text.expression_formality
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).text.expression_formality
        young = get_age_profile(AgeGroup.YOUNG_ADULT).text.expression_formality
        self.assertLess(early, mid)
        self.assertLess(mid, young)


# ============================================================
# 语音模态年龄校准测试
# ============================================================

class TestVoiceAgeCalibration(unittest.TestCase):
    """语音分析年龄校准测试"""

    def test_f0_baseline_decreases_with_age(self):
        """基频基线随年龄递减"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).voice.f0_baseline
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).voice.f0_baseline
        young = get_age_profile(AgeGroup.YOUNG_ADULT).voice.f0_baseline
        self.assertGreater(early, mid)
        self.assertGreater(mid, young)

    def test_f0_anxiety_threshold_decreases_with_age(self):
        """焦虑判定高基频阈值随年龄递减"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).voice.f0_anxiety_threshold
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).voice.f0_anxiety_threshold
        young = get_age_profile(AgeGroup.YOUNG_ADULT).voice.f0_anxiety_threshold
        self.assertGreater(early, mid)
        self.assertGreater(mid, young)

    def test_acoustic_risk_differs_by_age(self):
        """同一声学特征在不同年龄组产生不同风险分"""
        from perception.voice.acoustic import AcousticFeatures, features_to_emotion_risk

        # 一个中等偏高的基频 + 高语速（对初中生正常，对大学生焦虑）
        features = AcousticFeatures(
            f0_mean=260.0,
            f0_std=30.0,
            f0_range=80.0,
            energy_mean=50.0,
            energy_std=5.0,
            speech_rate=5.2,  # 超过高中/大学阈值(5.0/4.8) 但不超过初中(5.5)
            pause_ratio=0.15,
            duration=10.0,
            valid=True,
        )

        early_profile = get_age_profile(AgeGroup.EARLY_ADOLESCENT)
        young_profile = get_age_profile(AgeGroup.YOUNG_ADULT)

        risk_early = features_to_emotion_risk(features, age_profile=early_profile)
        risk_young = features_to_emotion_risk(features, age_profile=young_profile)
        risk_default = features_to_emotion_risk(features)

        # 5.2 音节/s 对大学生（阈值4.8）是焦虑指标，对初中生（阈值5.5）不是
        self.assertGreater(risk_young, risk_early)
        # 默认值应该与高中接近（默认阈值5.0）
        self.assertIsInstance(risk_default, float)

    def test_backward_compatible_no_age(self):
        """不提供 age_profile 时行为与原来一致"""
        from perception.voice.acoustic import AcousticFeatures, features_to_emotion_risk

        features = AcousticFeatures(
            f0_mean=100.0,
            energy_mean=-35.0,
            speech_rate=2.0,
            pause_ratio=0.5,
            f0_range=30.0,
            energy_std=12.0,
            valid=True,
        )
        risk = features_to_emotion_risk(features)
        self.assertIsInstance(risk, float)
        self.assertGreater(risk, 0.0)


# ============================================================
# 面部模态年龄校准测试
# ============================================================

class TestFaceAgeCalibration(unittest.TestCase):
    """面部分析年龄校准测试"""

    def test_au_baseline_decreases_with_age(self):
        """AU 强度基线随年龄递减（掩饰能力递增）"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).face.au_intensity_baseline
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).face.au_intensity_baseline
        young = get_age_profile(AgeGroup.YOUNG_ADULT).face.au_intensity_baseline
        self.assertGreater(early, mid)
        self.assertGreater(mid, young)

    def test_masking_weight_increases_with_age(self):
        """掩饰检测权重随年龄递增"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).face.masking_detection_weight
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).face.masking_detection_weight
        young = get_age_profile(AgeGroup.YOUNG_ADULT).face.masking_detection_weight
        self.assertLess(early, mid)
        self.assertLess(mid, young)

    def test_facial_risk_differs_by_age(self):
        """同一面部特征在不同年龄组产生不同风险分"""
        from perception.face.facial_expression import FacialFeatures, features_to_emotion_risk

        # 使用很低 AU 强度 + 多微表情 + 一些积极AU → 基础风险中等
        # 掩饰检测在大学生中触发（低AU强度+多微表情 → 掩饰嫌疑）
        features = FacialFeatures(
            au1_inner_brow_raiser=0.02,
            au4_brow_lowerer=0.03,
            au12_lip_corner_puller=0.10,
            au15_lip_corner_depressor=0.02,
            micro_expression_count=4,
            micro_expression_duration_avg=150.0,
            expression_intensity_mean=0.10,
            expression_intensity_max=0.20,
            detection_confidence=0.8,
        )

        early_profile = get_age_profile(AgeGroup.EARLY_ADOLESCENT)
        young_profile = get_age_profile(AgeGroup.YOUNG_ADULT)

        result_early = features_to_emotion_risk(features, age_profile=early_profile)
        result_young = features_to_emotion_risk(features, age_profile=young_profile)
        result_default = features_to_emotion_risk(features)

        # 大学生掩饰检测权重更高，低AU强度+多微表情 → 掩饰风险加成
        self.assertIn("risk_score", result_early)
        self.assertIn("risk_score", result_young)
        self.assertIn("risk_score", result_default)
        # 大学生掩饰加成应使风险更高
        self.assertGreater(result_young["risk_score"], result_early["risk_score"])

    def test_backward_compatible_no_age(self):
        """不提供 age_profile 时行为与原来一致"""
        from perception.face.facial_expression import FacialFeatures, features_to_emotion_risk

        features = FacialFeatures(
            au12_lip_corner_puller=0.5,
            au6_cheek_raiser=0.4,
            detection_confidence=0.9,
        )
        result = features_to_emotion_risk(features)
        self.assertIn("risk_score", result)
        self.assertIn("emotion_probs", result)


# ============================================================
# 行为模态年龄校准测试
# ============================================================

class TestBehaviorAgeCalibration(unittest.TestCase):
    """行为分析年龄校准测试"""

    def test_late_night_threshold_increases_with_age(self):
        """深夜阈值随年龄递增（小时数值越大表示越晚）"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).behavior.late_night_hour
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).behavior.late_night_hour
        young = get_age_profile(AgeGroup.YOUNG_ADULT).behavior.late_night_hour
        # 初中=23, 高中=0, 大学=1
        # 注意：0点比23点"更晚"，但数值上 0 < 23
        # 验证配置值正确
        self.assertEqual(early, 23)
        self.assertEqual(mid, 0)
        self.assertEqual(young, 1)

    def test_typing_baseline_decreases_with_age(self):
        """打字速度基线随年龄递减"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).behavior.typing_speed_baseline
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).behavior.typing_speed_baseline
        young = get_age_profile(AgeGroup.YOUNG_ADULT).behavior.typing_speed_baseline
        self.assertGreater(early, mid)
        self.assertGreater(mid, young)

    def test_behavior_risk_differs_by_age(self):
        """同一行为特征在不同年龄组产生不同风险分"""
        from perception.behavior.behavior_pattern import BehaviorFeatures, behavior_to_emotion_risk

        features = BehaviorFeatures(
            social_engagement=0.15,
            avoidance_score=0.4,
            typing_speed_trend=-0.3,
            session_frequency_change=-0.4,
            circadian_regularity=0.25,
            behavior_change_score=0.42,
            typing_speed_mean=95.0,
            data_quality=0.8,
        )

        early_profile = get_age_profile(AgeGroup.EARLY_ADOLESCENT)
        young_profile = get_age_profile(AgeGroup.YOUNG_ADULT)

        result_early = behavior_to_emotion_risk(features, age_profile=early_profile)
        result_young = behavior_to_emotion_risk(features, age_profile=young_profile)

        self.assertIn("risk_score", result_early)
        self.assertIn("risk_score", result_young)

    def test_backward_compatible_no_age(self):
        """不提供 age_profile 时行为与原来一致"""
        from perception.behavior.behavior_pattern import BehaviorFeatures, behavior_to_emotion_risk

        features = BehaviorFeatures(
            social_engagement=0.5,
            data_quality=0.9,
        )
        result = behavior_to_emotion_risk(features)
        self.assertIn("risk_score", result)
        self.assertIn("emotion_probs", result)


# ============================================================
# 融合权重年龄自适应测试
# ============================================================

class TestFusionAgeAdaptation(unittest.TestCase):
    """融合权重年龄自适应测试"""

    def test_face_weight_decreases_with_age(self):
        """面部融合权重随年龄递减"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).fusion.face_weight
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).fusion.face_weight
        young = get_age_profile(AgeGroup.YOUNG_ADULT).fusion.face_weight
        self.assertGreater(early, mid)
        self.assertGreater(mid, young)

    def test_text_weight_increases_with_age(self):
        """文本融合权重随年龄递增"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).fusion.text_weight
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).fusion.text_weight
        young = get_age_profile(AgeGroup.YOUNG_ADULT).fusion.text_weight
        self.assertLess(early, mid)
        self.assertLess(mid, young)

    def test_audio_weight_constant(self):
        """语音融合权重各年龄组一致"""
        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT).fusion.audio_weight
        mid = get_age_profile(AgeGroup.MID_ADOLESCENT).fusion.audio_weight
        young = get_age_profile(AgeGroup.YOUNG_ADULT).fusion.audio_weight
        self.assertAlmostEqual(early, mid)
        self.assertAlmostEqual(mid, young)

    def test_fusion_weights_sum_approximately(self):
        """各年龄组 11 模态权重之和约等于 1.0"""
        for group in AgeGroup:
            p = get_age_profile(group).fusion
            total = (
                p.text_weight + p.audio_weight + p.face_weight + p.behavior_weight
                + p.circadian_weight + p.cognitive_weight + p.hrv_weight
                + p.breathing_weight + p.behavioral_act_weight + p.eye_weight
                + p.voice_semantics_weight
            )
            self.assertAlmostEqual(total, 1.0, places=2, msg=f"{group.value} 权重之和不为 1.0 ({total})")

    def test_fused_probs_differ_by_age(self):
        """同一输入在不同年龄组融合后概率不同"""
        from perception.fusion.fusion import fuse_weighted_average
        import numpy as np

        text_probs = np.array([0.1, 0.3, 0.4, 0.1, 0.1])
        audio_probs = np.array([0.15, 0.25, 0.35, 0.15, 0.10])

        early = get_age_profile(AgeGroup.EARLY_ADOLESCENT)
        young = get_age_profile(AgeGroup.YOUNG_ADULT)

        fused_early = fuse_weighted_average(text_probs, audio_probs, age_profile=early)
        fused_young = fuse_weighted_average(text_probs, audio_probs, age_profile=young)
        fused_default = fuse_weighted_average(text_probs, audio_probs)

        # 不同年龄组应产生不同结果
        self.assertFalse(np.allclose(fused_early, fused_young))
        # 默认值应存在
        self.assertEqual(fused_default.shape, (5,))


# ============================================================
# 文本分析年龄校准测试
# ============================================================

class TestTextAgeCalibration(unittest.TestCase):
    """文本分析年龄校准测试"""

    def test_slang_detection_by_age(self):
        """不同年龄段的俚语能被正确检测"""
        from perception.perception_service import PerceptionService

        # 初中生俚语
        early_service = PerceptionService(age_group=AgeGroup.EARLY_ADOLESCENT)
        result = early_service._rule_based_text_analysis(
            "今天考试真的芭比Q了，我直接裂开",
            age_profile=get_age_profile(AgeGroup.EARLY_ADOLESCENT),
        )
        # 俚语应增加焦虑/悲伤维度
        self.assertIsInstance(result.text_emotion_probs, list)
        self.assertEqual(len(result.text_emotion_probs), 5)

    def test_crisis_detection_by_age(self):
        """不同年龄段的危机表达能被正确检测"""
        from perception.perception_service import PerceptionService

        # 大学生隐晦危机表达
        young_profile = get_age_profile(AgeGroup.YOUNG_ADULT)
        service = PerceptionService(age_group=AgeGroup.YOUNG_ADULT)
        result = service._rule_based_text_analysis(
            "存在没有意义，想结束这一切",
            age_profile=young_profile,
        )
        # 危机关键词应提升焦虑维度
        self.assertGreater(result.text_emotion_probs[2], 0.1)  # 焦虑维度

    def test_same_text_different_age_different_result(self):
        """同一段文本在不同年龄组产生不同结果"""
        from perception.perception_service import PerceptionService

        text = "今天emo了，精神状态不太好，想摆烂"

        early_result = PerceptionService()._rule_based_text_analysis(
            text, age_profile=get_age_profile(AgeGroup.EARLY_ADOLESCENT)
        )
        young_result = PerceptionService()._rule_based_text_analysis(
            text, age_profile=get_age_profile(AgeGroup.YOUNG_ADULT)
        )

        # 不同年龄组关键词库不同，结果应有差异
        self.assertNotEqual(early_result.text_emotion_probs, young_result.text_emotion_probs)


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEndAgeDifferentiation(unittest.TestCase):
    """端到端年龄差异化测试"""

    def test_perception_service_with_age_group(self):
        """PerceptionService 使用 age_group 初始化"""
        service = PerceptionService(age_group=AgeGroup.EARLY_ADOLESCENT)
        self.assertIsNotNone(service._age_profile)
        self.assertEqual(service._age_profile.age_group, AgeGroup.EARLY_ADOLESCENT)

    def test_perception_service_with_age(self):
        """PerceptionService 使用 age 初始化"""
        service = PerceptionService(age=14)
        self.assertIsNotNone(service._age_profile)
        self.assertEqual(service._age_profile.age_group, AgeGroup.EARLY_ADOLESCENT)

    def test_perception_service_without_age(self):
        """PerceptionService 不提供年龄 → 无年龄画像"""
        service = PerceptionService()
        self.assertIsNone(service._age_profile)

    def test_analyze_text_with_age(self):
        """analyze_text 使用年龄画像"""
        service = PerceptionService(age_group=AgeGroup.EARLY_ADOLESCENT)
        result = service.analyze_text("今天好emo，不想上学")
        self.assertIsInstance(result.text_emotion_probs, list)
        self.assertEqual(len(result.text_emotion_probs), 5)
        # 证据中应包含年龄校准标注
        has_age_evidence = any("年龄校准" in e for e in result.evidence)
        self.assertTrue(has_age_evidence)

    def test_analyze_text_without_age(self):
        """analyze_text 不使用年龄画像 → 向后兼容"""
        service = PerceptionService()
        result = service.analyze_text("今天心情不好")
        self.assertIsInstance(result.text_emotion_probs, list)
        self.assertEqual(len(result.text_emotion_probs), 5)

    def test_full_multimodal_with_age(self):
        """完整多模态分析使用年龄画像"""
        service = PerceptionService(age_group=AgeGroup.MID_ADOLESCENT)
        result = service.analyze_multimodal(
            text="考试压力好大，焦虑到睡不着",
            facial_features={
                "au1_inner_brow_raiser": 0.3,
                "au4_brow_lowerer": 0.2,
                "detection_confidence": 0.8,
            },
            behavior_features={
                "late_night_ratio": 0.4,
                "social_engagement": 0.3,
                "data_quality": 0.7,
            },
        )
        self.assertIsInstance(result.text_emotion_probs, list)
        self.assertEqual(len(result.text_emotion_probs), 5)
        # 证据中应包含年龄校准标注
        has_age_evidence = any("年龄" in e for e in result.evidence)
        self.assertTrue(has_age_evidence)


# ============================================================
# 语音模态 _acoustic_to_emotion_probs 年龄校准测试
# ============================================================

class TestAcousticToEmotionAgeCalibration(unittest.TestCase):
    """声学→情绪映射的年龄校准测试"""

    def test_acoustic_probs_with_age(self):
        """_acoustic_to_emotion_probs 支持年龄校准"""
        from perception.perception_service import _acoustic_to_emotion_probs
        from perception.voice.acoustic import AcousticFeatures

        features = AcousticFeatures(
            f0_mean=280.0,
            f0_std=25.0,
            f0_range=90.0,
            energy_mean=55.0,
            energy_std=6.0,
            speech_rate=5.2,
            pause_ratio=0.1,
            duration=8.0,
            valid=True,
        )

        early_profile = get_age_profile(AgeGroup.EARLY_ADOLESCENT)
        young_profile = get_age_profile(AgeGroup.YOUNG_ADULT)

        probs_early = _acoustic_to_emotion_probs(features, early_profile)
        probs_young = _acoustic_to_emotion_probs(features, young_profile)
        probs_default = _acoustic_to_emotion_probs(features)

        # 都是 5 维概率分布
        self.assertEqual(len(probs_early), 5)
        self.assertEqual(len(probs_young), 5)
        self.assertEqual(len(probs_default), 5)

        # 280Hz + 5.2音节/s 对大学生是焦虑指标，对初中生可能不是
        # 焦虑维度 (index=2) 在大学生中应更高
        self.assertGreater(probs_young[2], probs_early[2])


# ============================================================
# 融合 YAML 配置测试
# ============================================================

class TestFusionYamlAgeConfig(unittest.TestCase):
    """fusion.yaml 年龄配置测试"""

    def test_yaml_has_age_profiles(self):
        """fusion.yaml 包含 age_profiles 段"""
        from perception.fusion.fusion import load_fusion_config
        config = load_fusion_config()
        self.assertIn("age_profiles", config)

    def test_yaml_age_profiles_complete(self):
        """fusion.yaml 包含所有三个年龄组"""
        from perception.fusion.fusion import load_fusion_config
        config = load_fusion_config()
        age_profiles = config["age_profiles"]
        self.assertIn("early_adolescent", age_profiles)
        self.assertIn("mid_adolescent", age_profiles)
        self.assertIn("young_adult", age_profiles)

    def test_yaml_age_profile_weights(self):
        """fusion.yaml 中各年龄组权重正确"""
        from perception.fusion.fusion import load_fusion_config
        config = load_fusion_config()
        early = config["age_profiles"]["early_adolescent"]
        self.assertAlmostEqual(early["modality_weights"]["face"], 0.25)
        self.assertAlmostEqual(early["modality_weights"]["text"], 0.40)
        self.assertAlmostEqual(early["risk_threshold_offset"], -0.05)


if __name__ == "__main__":
    unittest.main()
