"""
数据脱敏模块单元测试

覆盖：
- summarize() 从 EmotionResult 和 dict 生成 FeatureSummary
- 差分隐私噪声注入
- validate_no_raw_data() 检测各类数据泄露
- 批量脱敏
- epsilon 预算计算
"""
from __future__ import annotations

import time
import uuid

import pytest

from privacy.sanitizer import (
    compute_epsilon_for_budget,
    summarize,
    summarize_batch,
    validate_no_raw_data,
)
from shared.dataclasses import EmotionResult, FeatureSummary


# ============================================================
# summarize 测试
# ============================================================

class TestSummarize:
    @pytest.fixture
    def sample_emotion_result(self):
        """样本 EmotionResult"""
        return EmotionResult(
            text_emotion_probs=[0.1, 0.2, 0.3, 0.1, 0.3],
            audio_risk_prob=0.4,
            confidence=0.85,
            timestamp=time.time(),
            evidence=["用户表达了焦虑情绪"],
        )

    def test_summarize_from_emotion_result(self, sample_emotion_result):
        """从 EmotionResult 生成 FeatureSummary"""
        summary = summarize(sample_emotion_result)
        assert isinstance(summary, FeatureSummary)
        assert len(summary.text_emotion_probs) == 5
        assert summary.audio_risk_prob == 0.4
        assert summary.confidence == 0.85

    def test_summarize_from_dict(self):
        """从 dict 生成 FeatureSummary"""
        data = {
            "text_emotion_probs": [0.2, 0.3, 0.1, 0.2, 0.2],
            "audio_risk_prob": 0.6,
            "confidence": 0.9,
        }
        summary = summarize(data)
        assert isinstance(summary, FeatureSummary)
        assert len(summary.text_emotion_probs) == 5

    def test_summarize_session_id_is_uuid(self, sample_emotion_result):
        """session_id 是合法 UUID"""
        summary = summarize(sample_emotion_result)
        # 应能解析为 UUID
        parsed = uuid.UUID(summary.session_id)
        assert parsed.version == 4

    def test_summarize_session_ids_unique(self, sample_emotion_result):
        """每次调用生成不同的 session_id"""
        s1 = summarize(sample_emotion_result)
        s2 = summarize(sample_emotion_result)
        assert s1.session_id != s2.session_id

    def test_summarize_has_timestamp(self, sample_emotion_result):
        """包含时间戳"""
        summary = summarize(sample_emotion_result)
        assert summary.timestamp > 0

    def test_summarize_no_raw_text(self, sample_emotion_result):
        """不包含原始文本"""
        summary = summarize(sample_emotion_result)
        # evidence 中的原始文本不应出现在 summary 中
        summary_str = str(summary.text_emotion_probs)
        assert "焦虑" not in summary_str

    def test_summarize_invalid_type(self):
        """无效类型抛出 TypeError"""
        with pytest.raises(TypeError):
            summarize("invalid input")


# ============================================================
# 差分隐私测试
# ============================================================

class TestDifferentialPrivacy:
    @pytest.fixture
    def sample_result(self):
        return EmotionResult(
            text_emotion_probs=[0.2, 0.2, 0.2, 0.2, 0.2],
            audio_risk_prob=0.5,
            confidence=0.8,
            timestamp=time.time(),
            evidence=[],
        )

    def test_no_noise_without_epsilon(self, sample_result):
        """不指定 epsilon 时不添加噪声"""
        summary = summarize(sample_result, epsilon=None)
        assert summary.text_emotion_probs == [0.2, 0.2, 0.2, 0.2, 0.2]

    def test_noise_with_epsilon(self, sample_result):
        """指定 epsilon 后添加噪声"""
        summary = summarize(sample_result, epsilon=1.0, seed=42)
        # 噪声后值应不同于原始值
        assert summary.text_emotion_probs != [0.2, 0.2, 0.2, 0.2, 0.2]

    def test_noise_clipped_to_01(self, sample_result):
        """噪声后值仍在 [0, 1] 范围"""
        summary = summarize(sample_result, epsilon=0.1, seed=42)
        for p in summary.text_emotion_probs:
            assert 0.0 <= p <= 1.0

    def test_high_epsilon_less_noise(self, sample_result):
        """高 epsilon = 少噪声"""
        summary_low = summarize(sample_result, epsilon=0.1, seed=42)
        summary_high = summarize(sample_result, epsilon=100.0, seed=42)
        # 高 epsilon 的结果应更接近原始值
        diff_low = sum(abs(p - 0.2) for p in summary_low.text_emotion_probs)
        diff_high = sum(abs(p - 0.2) for p in summary_high.text_emotion_probs)
        assert diff_high < diff_low

    def test_audio_noise_with_epsilon(self, sample_result):
        """音频概率也添加噪声"""
        summary = summarize(sample_result, epsilon=1.0, seed=42)
        assert summary.audio_risk_prob != 0.5
        assert 0.0 <= summary.audio_risk_prob <= 1.0

    def test_seed_reproducibility(self, sample_result):
        """相同 seed 产生相同结果"""
        s1 = summarize(sample_result, epsilon=1.0, seed=42)
        s2 = summarize(sample_result, epsilon=1.0, seed=42)
        assert s1.text_emotion_probs == s2.text_emotion_probs


# ============================================================
# validate_no_raw_data 测试
# ============================================================

class TestValidateNoRawData:
    def test_valid_summary_passes(self):
        """合法 summary 通过验证"""
        summary = FeatureSummary(
            text_emotion_probs=[0.1, 0.2, 0.3, 0.1, 0.3],
            audio_risk_prob=0.4,
            confidence=0.85,
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
        )
        assert validate_no_raw_data(summary) is True

    def test_chinese_text_detected(self):
        """检测中文字符串泄露"""
        summary = FeatureSummary(
            text_emotion_probs=[0.1, 0.2, 0.3, 0.1, 0.3],
            audio_risk_prob=0.4,
            confidence=0.85,
            session_id="我感到很焦虑不知道该怎么办",  # 非法 session_id
            timestamp=time.time(),
        )
        with pytest.raises(ValueError, match="中文字符串"):
            validate_no_raw_data(summary)

    def test_english_text_detected(self):
        """检测长英文字符串泄露"""
        summary = FeatureSummary(
            text_emotion_probs=[0.1, 0.2, 0.3, 0.1, 0.3],
            audio_risk_prob=0.4,
            confidence=0.85,
            session_id="thisisaverylongenglishstringthatshouldnotbehere",
            timestamp=time.time(),
        )
        with pytest.raises(ValueError, match="英文字符串"):
            validate_no_raw_data(summary)

    def test_pii_detected(self):
        """检测 PII 信息泄露"""
        summary = FeatureSummary(
            text_emotion_probs=[0.1, 0.2, 0.3, 0.1, 0.3],
            audio_risk_prob=0.4,
            confidence=0.85,
            session_id="test@test.com",
            timestamp=time.time(),
        )
        with pytest.raises(ValueError, match="PII"):
            validate_no_raw_data(summary)

    def test_invalid_uuid_detected(self):
        """检测非法 UUID"""
        summary = FeatureSummary(
            text_emotion_probs=[0.1, 0.2, 0.3, 0.1, 0.3],
            audio_risk_prob=0.4,
            confidence=0.85,
            session_id="not-a-uuid",
            timestamp=time.time(),
        )
        with pytest.raises(ValueError, match="UUID"):
            validate_no_raw_data(summary)

    def test_confidence_out_of_range(self):
        """检测 confidence 超出范围"""
        summary = FeatureSummary(
            text_emotion_probs=[0.1, 0.2, 0.3, 0.1, 0.3],
            audio_risk_prob=0.4,
            confidence=1.5,  # 超出 [0, 1]
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
        )
        with pytest.raises(ValueError, match="confidence"):
            validate_no_raw_data(summary)

    def test_probs_out_of_range(self):
        """检测概率值超出范围"""
        summary = FeatureSummary(
            text_emotion_probs=[0.1, -0.5, 0.3, 0.1, 0.3],  # -0.5 超出范围
            audio_risk_prob=0.4,
            confidence=0.85,
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
        )
        with pytest.raises(ValueError, match="超出"):
            validate_no_raw_data(summary)

    def test_non_numeric_probs_detected(self):
        """检测非数值概率"""
        summary = FeatureSummary(
            text_emotion_probs=[0.1, "text", 0.3, 0.1, 0.3],
            audio_risk_prob=0.4,
            confidence=0.85,
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
        )
        with pytest.raises(ValueError, match="不是数值"):
            validate_no_raw_data(summary)


# ============================================================
# 批量脱敏测试
# ============================================================

class TestBatchSummarize:
    def test_batch_summarize(self):
        """批量脱敏"""
        results = [
            EmotionResult(
                text_emotion_probs=[0.1, 0.2, 0.3, 0.1, 0.3],
                audio_risk_prob=0.4,
                confidence=0.85,
                timestamp=time.time(),
                evidence=[],
            ),
            EmotionResult(
                text_emotion_probs=[0.3, 0.1, 0.2, 0.3, 0.1],
                audio_risk_prob=0.6,
                confidence=0.9,
                timestamp=time.time(),
                evidence=[],
            ),
        ]
        summaries = summarize_batch(results)
        assert len(summaries) == 2
        assert all(isinstance(s, FeatureSummary) for s in summaries)

    def test_batch_with_epsilon(self):
        """批量脱敏 + 差分隐私"""
        results = [
            EmotionResult(
                text_emotion_probs=[0.2, 0.2, 0.2, 0.2, 0.2],
                audio_risk_prob=0.5,
                confidence=0.8,
                timestamp=time.time(),
                evidence=[],
            ),
        ]
        summaries = summarize_batch(results, epsilon=1.0)
        assert len(summaries) == 1
        # 噪声后值应不同
        assert summaries[0].text_emotion_probs != [0.2, 0.2, 0.2, 0.2, 0.2]


# ============================================================
# epsilon 预算计算测试
# ============================================================

class TestEpsilonBudget:
    def test_basic_composition(self):
        """基本组合"""
        epsilon = compute_epsilon_for_budget(total_budget=10.0, n_queries=5)
        assert epsilon == 2.0  # 10 / 5

    def test_advanced_composition(self):
        """高级组合"""
        epsilon = compute_epsilon_for_budget(
            total_budget=10.0, n_queries=5, composition="advanced"
        )
        assert epsilon > 0

    def test_invalid_composition(self):
        """无效组合方式"""
        with pytest.raises(ValueError, match="未知"):
            compute_epsilon_for_budget(10.0, 5, composition="invalid")
