"""
语音声学特征提取模块测试
"""
import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.voice.acoustic import (
    AcousticFeatures,
    extract_from_buffer,
    features_to_emotion_risk,
    features_to_summary,
    _extract_features_numpy,
)


class TestAcousticFeatures:
    """测试声学特征数据结构"""

    def test_default_values(self):
        features = AcousticFeatures()
        assert features.f0_mean == 0.0
        assert features.valid is False

    def test_from_buffer_short_audio(self):
        """短音频（< 0.5 秒）应返回无效特征"""
        audio = np.zeros(1000, dtype=np.float32)
        features = extract_from_buffer(audio, sample_rate=16000)
        assert features.valid is False

    def test_from_buffer_sine_wave(self):
        """正弦波音频应提取到基频"""
        sr = 16000
        duration = 2.0
        t = np.arange(int(sr * duration)) / sr
        # 200 Hz 正弦波
        audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        features = extract_from_buffer(audio, sample_rate=sr)
        assert features.valid is True
        assert features.duration == pytest.approx(2.0, abs=0.1)

    def test_numpy_fallback(self):
        """纯 numpy 特征提取应工作"""
        sr = 16000
        duration = 2.0
        t = np.arange(int(sr * duration)) / sr
        audio = np.sin(2 * np.pi * 150 * t).astype(np.float32)
        features = _extract_features_numpy(audio, sr)
        assert features.valid is True
        assert features.duration == pytest.approx(2.0, abs=0.1)


class TestFeaturesToEmotionRisk:
    """测试声学特征 → 情绪风险映射"""

    def test_invalid_features_return_zero(self):
        features = AcousticFeatures(valid=False)
        assert features_to_emotion_risk(features) == 0.0

    def test_normal_voice_low_risk(self):
        features = AcousticFeatures(
            f0_mean=200.0,
            energy_mean=-20.0,
            speech_rate=3.0,
            pause_ratio=0.2,
            f0_range=100.0,
            energy_std=5.0,
            valid=True,
        )
        risk = features_to_emotion_risk(features)
        assert 0.0 <= risk <= 1.0

    def test_depression_indicators(self):
        """低基频 + 低能量 → 应有一定风险"""
        features = AcousticFeatures(
            f0_mean=100.0,  # 低基频
            energy_mean=-35.0,  # 低能量
            speech_rate=2.0,
            pause_ratio=0.5,  # 高停顿
            f0_range=30.0,  # 低基频范围
            energy_std=3.0,
            valid=True,
        )
        risk = features_to_emotion_risk(features)
        assert risk > 0.0

    def test_anxiety_indicators(self):
        """高语速 → 应有风险"""
        features = AcousticFeatures(
            f0_mean=250.0,
            energy_mean=-15.0,
            speech_rate=6.0,  # 高语速
            pause_ratio=0.1,
            f0_range=150.0,
            energy_std=12.0,  # 高能量波动
            valid=True,
        )
        risk = features_to_emotion_risk(features)
        assert risk > 0.0

    def test_risk_bounded(self):
        features = AcousticFeatures(
            f0_mean=100.0,
            energy_mean=-40.0,
            speech_rate=8.0,
            pause_ratio=0.6,
            f0_range=20.0,
            energy_std=15.0,
            valid=True,
        )
        risk = features_to_emotion_risk(features)
        assert 0.0 <= risk <= 1.0


class TestFeaturesToSummary:
    """测试特征摘要"""

    def test_summary_contains_risk_prob(self):
        features = AcousticFeatures(
            f0_mean=200.0,
            energy_mean=-20.0,
            speech_rate=3.0,
            pause_ratio=0.2,
            f0_range=100.0,
            energy_std=5.0,
            duration=2.0,
            valid=True,
        )
        summary = features_to_summary(features)
        assert "risk_prob" in summary
        assert "f0_mean" in summary
        assert "valid" in summary
        assert 0.0 <= summary["risk_prob"] <= 1.0

    def test_invalid_features_summary(self):
        features = AcousticFeatures(valid=False)
        summary = features_to_summary(features)
        assert summary["risk_prob"] == 0.0
        assert summary["valid"] is False
