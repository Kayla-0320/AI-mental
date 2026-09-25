"""
PerceptionService 单元测试

验证三个核心方法均能返回合法的 EmotionResult。
"""
import os
import tempfile
import time
import unittest

from shared.dataclasses import EmotionResult
from perception.perception_service import PerceptionService


class TestPerceptionService(unittest.TestCase):
    """PerceptionService 测试套件"""

    def setUp(self) -> None:
        self.service = PerceptionService()

    # ----------------------------------------------------------
    # analyze_text
    # ----------------------------------------------------------

    def test_analyze_text_returns_emotion_result(self):
        """analyze_text 必须返回 EmotionResult"""
        result = self.service.analyze_text("今天心情不太好")
        self.assertIsInstance(result, EmotionResult)

    def test_analyze_text_probs_length(self):
        """text_emotion_probs 必须为 5 维"""
        result = self.service.analyze_text("测试文本")
        self.assertEqual(len(result.text_emotion_probs), 5)

    def test_analyze_text_probs_are_float(self):
        """text_emotion_probs 元素必须为 float"""
        result = self.service.analyze_text("测试文本")
        for p in result.text_emotion_probs:
            self.assertIsInstance(p, float)

    def test_analyze_text_audio_risk_prob_is_none(self):
        """纯文本分析 audio_risk_prob 必须为 None"""
        result = self.service.analyze_text("测试文本")
        self.assertIsNone(result.audio_risk_prob)

    def test_analyze_text_confidence_range(self):
        """confidence 必须在 [0.0, 1.0] 范围内"""
        result = self.service.analyze_text("测试文本")
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_analyze_text_has_timestamp(self):
        """timestamp 必须为合理的 Unix 时间戳"""
        before = time.time()
        result = self.service.analyze_text("测试文本")
        after = time.time()
        self.assertGreaterEqual(result.timestamp, before)
        self.assertLessEqual(result.timestamp, after)

    def test_analyze_text_has_evidence(self):
        """evidence 必须为非空列表"""
        result = self.service.analyze_text("测试文本")
        self.assertIsInstance(result.evidence, list)
        self.assertGreater(len(result.evidence), 0)

    # ----------------------------------------------------------
    # analyze_audio
    # ----------------------------------------------------------

    def test_analyze_audio_returns_emotion_result(self):
        """analyze_audio 必须返回 EmotionResult"""
        # 创建临时假文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake wav data")
            wav_path = f.name

        try:
            result = self.service.analyze_audio(wav_path)
            self.assertIsInstance(result, EmotionResult)
            self.assertIsNotNone(result.audio_risk_prob)
        finally:
            os.unlink(wav_path)

    def test_analyze_audio_file_not_found(self):
        """音频文件不存在时必须抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.service.analyze_audio("/nonexistent/path/audio.wav")

    def test_analyze_audio_has_risk_prob(self):
        """语音分析结果 audio_risk_prob 不为 None"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake wav data")
            wav_path = f.name

        try:
            result = self.service.analyze_audio(wav_path)
            self.assertIsNotNone(result.audio_risk_prob)
            self.assertIsInstance(result.audio_risk_prob, float)
        finally:
            os.unlink(wav_path)

    # ----------------------------------------------------------
    # analyze_multimodal
    # ----------------------------------------------------------

    def test_multimodal_text_only_returns_emotion_result(self):
        """仅传文本时必须返回 EmotionResult（降级模式）"""
        result = self.service.analyze_multimodal(text="今天心情不太好")
        self.assertIsInstance(result, EmotionResult)
        self.assertIsNone(result.audio_risk_prob)

    def test_multimodal_text_only_has_fallback_evidence(self):
        """evidence 中必须包含融合模式说明（HUAF-TC 层级或传统降级）"""
        result = self.service.analyze_multimodal(text="测试文本")
        has_info = any(
            "降级" in e or "基础感知层" in e or "L1" in e or "数字孪生" in e
            for e in result.evidence
        )
        self.assertTrue(has_info, f"缺少融合模式说明，evidence={result.evidence}")

    def test_multimodal_with_audio_returns_emotion_result(self):
        """同时传入文本和音频时必须返回 EmotionResult"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake wav data")
            wav_path = f.name

        try:
            result = self.service.analyze_multimodal(
                text="测试文本", wav_path=wav_path
            )
            self.assertIsInstance(result, EmotionResult)
            self.assertIsNotNone(result.audio_risk_prob)
        finally:
            os.unlink(wav_path)

    def test_multimodal_with_audio_has_fused_probs(self):
        """融合模式下 text_emotion_probs 为加权平均结果"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake wav data")
            wav_path = f.name

        try:
            result = self.service.analyze_multimodal(
                text="测试文本", wav_path=wav_path
            )
            self.assertEqual(len(result.text_emotion_probs), 5)
            # 融合后的概率应介于文本和语音之间
            for p in result.text_emotion_probs:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)
        finally:
            os.unlink(wav_path)

    def test_multimodal_audio_failure_fallback(self):
        """音频解析失败时必须降级为文本单模态"""
        # 传入不存在的音频路径，应触发异常捕获并降级
        result = self.service.analyze_multimodal(
            text="测试文本", wav_path="/nonexistent/audio.wav"
        )
        self.assertIsInstance(result, EmotionResult)
        self.assertIsNone(result.audio_risk_prob)
        # 应包含融合模式说明（HUAF-TC 层级或传统降级）
        has_info = any(
            "降级" in e or "基础感知层" in e or "L1" in e or "数字孪生" in e
            for e in result.evidence
        )
        self.assertTrue(has_info, f"缺少融合模式说明，evidence={result.evidence}")

    def test_multimodal_with_audio_has_fusion_evidence(self):
        """融合模式 evidence 中必须包含融合说明"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake wav data")
            wav_path = f.name

        try:
            result = self.service.analyze_multimodal(
                text="测试文本", wav_path=wav_path
            )
            has_fusion = any("融合" in e for e in result.evidence)
            self.assertTrue(has_fusion, f"缺少融合说明，evidence={result.evidence}")
        finally:
            os.unlink(wav_path)


class TestCrisisKeywordScreen(unittest.TestCase):
    """危机关键词筛查测试

    背景：危机词表原先只写在 _rule_based_text_analysis 内部，而该分支只有在
    predict() 抛异常时才会走到，导致正常路径上的 analyze_text() 拿不到危机信号。
    现在筛查独立于情感引擎，任何路径都必须留下结构化痕迹。
    """

    def setUp(self) -> None:
        self.service = PerceptionService()

    def test_direct_hit_is_screened(self):
        """字面命中危机词必须被检出"""
        result = self.service.analyze_text("我不想活了，想结束这一切")
        self.assertGreaterEqual(len(result.crisis_keywords), 2)

    def test_neutral_text_has_no_hits(self):
        """正常文本不得误报"""
        result = self.service.analyze_text("今天天气很好，和朋友一起吃饭")
        self.assertEqual(result.crisis_keywords, [])

    def test_evidence_records_crisis(self):
        """命中后 evidence 必须留下可追溯记录"""
        result = self.service.analyze_text("我想死")
        self.assertTrue(
            any("危机" in e for e in result.evidence),
            f"evidence 缺少危机记录: {result.evidence}",
        )

    def test_probs_stay_normalized(self):
        """抬高通道后概率仍须归一化"""
        result = self.service.analyze_text("我想死，撑不住了")
        self.assertAlmostEqual(sum(result.text_emotion_probs), 1.0, places=6)

    def test_screen_is_independent_of_engine(self):
        """筛查必须独立于情感引擎：直接对引擎产物调用也要生效"""
        base = EmotionResult(
            text_emotion_probs=[0.05, 0.10, 0.15, 0.05, 0.65],
            audio_risk_prob=None,
            confidence=0.65,
            timestamp=time.time(),
            evidence=["文本情感分析（jieba+词典）：主导情绪=中性"],
        )
        screened = self.service._with_crisis_screen(base, "我想死，撑不住了")
        self.assertGreaterEqual(len(screened.crisis_keywords), 2)
        self.assertTrue(any("[危机筛查]" in e for e in screened.evidence))
        # 原对象不被就地修改
        self.assertEqual(base.crisis_keywords, [])

    def test_screen_is_idempotent(self):
        """重复筛查不得重复叠加概率"""
        base = EmotionResult(
            text_emotion_probs=[0.2] * 5, audio_risk_prob=None,
            confidence=0.5, timestamp=time.time(), evidence=[],
        )
        once = self.service._with_crisis_screen(base, "我想死")
        twice = self.service._with_crisis_screen(once, "我想死")
        self.assertEqual(once.text_emotion_probs, twice.text_emotion_probs)
        self.assertEqual(once.crisis_keywords, twice.crisis_keywords)


if __name__ == "__main__":
    unittest.main()
