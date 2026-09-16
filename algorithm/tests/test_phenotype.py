"""
数字表型特征提取单元测试

覆盖：
- 五类特征提取
- 数据缺失处理
- 时序变化检测
- 基线偏离度（Z-score）
- PhenotypeVector 数据结构
"""
from __future__ import annotations

import numpy as np
import pytest

from assessment.phenotype import (
    AGE_GROUP_BASELINES,
    DEFAULT_TIME_WINDOW_DAYS,
    compute_baseline_deviation,
    detect_temporal_changes,
    extract_digital_phenotype,
    _estimate_feature_std,
    _extract_assessment_features,
    _extract_behavior_features,
    _extract_emotion_features,
    _extract_physiological_features,
    _extract_sleep_features,
)
from shared.dataclasses import PhenotypeFeature, PhenotypeVector


# ============================================================
# 数据类测试
# ============================================================

class TestPhenotypeFeature:
    def test_feature_creation(self):
        """创建特征"""
        f = PhenotypeFeature(
            name="sleep_duration_mean",
            value=7.5,
            confidence=0.9,
            source="wearable",
            unit="hours",
        )
        assert f.name == "sleep_duration_mean"
        assert f.value == 7.5
        assert f.confidence == 0.9
        assert f.significant_change is False

    def test_feature_missing_value(self):
        """缺失值特征"""
        f = PhenotypeFeature(
            name="sleep_duration_mean",
            value=None,
            confidence=0.0,
            source="missing",
        )
        assert f.value is None
        assert f.confidence == 0.0


class TestPhenotypeVector:
    def test_vector_creation(self):
        """创建表型向量"""
        pv = PhenotypeVector(
            user_id="user_1",
            time_window_days=14,
            sleep_features=[
                PhenotypeFeature("sleep_duration_mean", 7.5, 0.9, "wearable"),
            ],
        )
        assert pv.user_id == "user_1"
        assert len(pv.sleep_features) == 1

    def test_all_features(self):
        """all_features 返回所有特征"""
        pv = PhenotypeVector(
            user_id="user_1",
            time_window_days=14,
            sleep_features=[PhenotypeFeature("f1", 1.0, 0.9, "src")],
            emotion_features=[PhenotypeFeature("f2", 2.0, 0.8, "src")],
            behavior_features=[PhenotypeFeature("f3", 3.0, 0.7, "src")],
            assessment_features=[PhenotypeFeature("f4", 4.0, 0.6, "src")],
            physiological_features=[PhenotypeFeature("f5", 5.0, 0.5, "src")],
        )
        assert len(pv.all_features()) == 5

    def test_feature_dict(self):
        """feature_dict 返回 {名称: 值}"""
        pv = PhenotypeVector(
            user_id="user_1",
            time_window_days=14,
            sleep_features=[PhenotypeFeature("f1", 1.0, 0.9, "src")],
        )
        d = pv.feature_dict()
        assert d["f1"] == 1.0

    def test_missing_features(self):
        """missing_features 返回缺失特征名"""
        pv = PhenotypeVector(
            user_id="user_1",
            time_window_days=14,
            sleep_features=[
                PhenotypeFeature("f1", 1.0, 0.9, "src"),
                PhenotypeFeature("f2", None, 0.0, "missing"),
            ],
        )
        missing = pv.missing_features()
        assert "f2" in missing
        assert "f1" not in missing


# ============================================================
# 特征提取测试
# ============================================================

class TestSleepFeatureExtraction:
    def test_extract_with_data(self):
        """有数据时提取特征"""
        data = {
            "durations": [7.0, 7.5, 8.0, 6.5, 7.2],
            "bedtimes": [23.0, 23.5, 22.5, 24.0, 23.0],
            "wake_times": [7.0, 7.0, 6.5, 7.5, 7.0],
        }
        features = _extract_sleep_features(data)
        assert len(features) == 3
        assert features[0].name == "sleep_duration_mean"
        assert features[0].value is not None
        assert features[0].value > 0

    def test_extract_with_none(self):
        """无数据时返回 None 值"""
        features = _extract_sleep_features(None)
        assert len(features) == 3
        assert all(f.value is None for f in features)
        assert all(f.source == "missing" for f in features)


class TestEmotionFeatureExtraction:
    def test_extract_with_data(self):
        """有数据时提取情感特征"""
        data = {
            "valence_scores": [0.6, 0.7, 0.5, 0.8, 0.4],
            "emotion_labels": ["positive", "neutral", "anxiety", "positive", "neutral"],
        }
        features = _extract_emotion_features(data)
        assert len(features) == 3
        assert features[0].name == "emotion_valence_mean"
        assert features[0].value is not None

    def test_extract_with_none(self):
        """无数据时返回 None"""
        features = _extract_emotion_features(None)
        assert all(f.value is None for f in features)


class TestBehaviorFeatureExtraction:
    def test_extract_with_data(self):
        """有数据时提取行为特征"""
        data = {
            "daily_sessions": [3, 4, 5, 2, 6],
            "social_interactions": [5, 6, 4, 7, 3],
            "active_hours": [6.0, 7.0, 5.5, 8.0, 6.5],
        }
        features = _extract_behavior_features(data)
        assert len(features) == 3
        assert features[0].name == "activity_level"

    def test_extract_with_none(self):
        """无数据时返回 None"""
        features = _extract_behavior_features(None)
        assert all(f.value is None for f in features)


class TestAssessmentFeatureExtraction:
    def test_extract_with_data(self):
        """有数据时提取测评特征"""
        data = {
            "phq9_scores": [5, 6, 4, 7],
            "gad7_scores": [4, 5, 3, 6],
            "timestamps": [0, 1, 2, 3],
        }
        features = _extract_assessment_features(data)
        assert len(features) == 3
        assert features[0].name == "phq9_latest"
        assert features[0].value == 7.0  # 最后一个

    def test_extract_with_none(self):
        """无数据时返回 None"""
        features = _extract_assessment_features(None)
        assert all(f.value is None for f in features)


class TestPhysiologicalFeatureExtraction:
    def test_extract_with_data(self):
        """有数据时提取生理特征"""
        data = {
            "heart_rate_variability": [50, 55, 48, 52, 53],
            "daily_steps": [6000, 7000, 5500, 8000, 6500],
            "resting_hr": [72, 70, 74, 71, 73],
        }
        features = _extract_physiological_features(data)
        assert len(features) == 3
        assert features[0].name == "heart_rate_variability"

    def test_extract_with_none(self):
        """无数据时返回 None"""
        features = _extract_physiological_features(None)
        assert all(f.value is None for f in features)


# ============================================================
# 核心接口测试
# ============================================================

class TestExtractDigitalPhenotype:
    def test_extract_returns_vector(self):
        """extract 返回 PhenotypeVector"""
        pv = extract_digital_phenotype("user_123")
        assert isinstance(pv, PhenotypeVector)
        assert pv.user_id == "user_123"
        assert pv.time_window_days == DEFAULT_TIME_WINDOW_DAYS

    def test_extract_has_all_categories(self):
        """提取结果包含五类特征"""
        pv = extract_digital_phenotype("user_123")
        assert len(pv.sleep_features) >= 1
        assert len(pv.emotion_features) >= 1
        assert len(pv.behavior_features) >= 1
        assert len(pv.assessment_features) >= 1
        assert len(pv.physiological_features) >= 1

    def test_extract_custom_window(self):
        """自定义时间窗口"""
        pv = extract_digital_phenotype("user_123", time_window_days=7)
        assert pv.time_window_days == 7

    def test_extract_has_timestamp(self):
        """包含时间戳"""
        pv = extract_digital_phenotype("user_123")
        assert pv.timestamp > 0

    def test_extract_features_have_confidence(self):
        """特征包含置信度"""
        pv = extract_digital_phenotype("user_123")
        for f in pv.all_features():
            assert 0.0 <= f.confidence <= 1.0

    def test_extract_features_have_source(self):
        """特征包含数据来源"""
        pv = extract_digital_phenotype("user_123")
        for f in pv.all_features():
            assert f.source in ("wearable", "chat_log", "app_usage", "assessment", "missing")


# ============================================================
# 时序变化检测测试
# ============================================================

class TestTemporalChangeDetection:
    def test_detect_changes_marks_significant(self):
        """标记显著变化"""
        current = PhenotypeVector(
            user_id="u1", time_window_days=14,
            sleep_features=[PhenotypeFeature("sleep_duration_mean", 5.0, 0.9, "wearable")],
        )
        previous = PhenotypeVector(
            user_id="u1", time_window_days=14,
            sleep_features=[PhenotypeFeature("sleep_duration_mean", 8.0, 0.9, "wearable")],
        )
        result = detect_temporal_changes(current, previous)
        # 差值 3h，std 估计 1.0 → Cohen's d = 3.0 > 0.5
        assert result.sleep_features[0].significant_change is True

    def test_detect_changes_no_significant(self):
        """无显著变化"""
        current = PhenotypeVector(
            user_id="u1", time_window_days=14,
            sleep_features=[PhenotypeFeature("sleep_duration_mean", 7.1, 0.9, "wearable")],
        )
        previous = PhenotypeVector(
            user_id="u1", time_window_days=14,
            sleep_features=[PhenotypeFeature("sleep_duration_mean", 7.0, 0.9, "wearable")],
        )
        result = detect_temporal_changes(current, previous)
        # 差值 0.1h，std 估计 1.0 → Cohen's d = 0.1 < 0.5
        assert result.sleep_features[0].significant_change is False

    def test_detect_changes_missing_value(self):
        """缺失值不标记变化"""
        current = PhenotypeVector(
            user_id="u1", time_window_days=14,
            sleep_features=[PhenotypeFeature("sleep_duration_mean", None, 0.0, "missing")],
        )
        previous = PhenotypeVector(
            user_id="u1", time_window_days=14,
            sleep_features=[PhenotypeFeature("sleep_duration_mean", 7.0, 0.9, "wearable")],
        )
        result = detect_temporal_changes(current, previous)
        assert result.sleep_features[0].significant_change is False


# ============================================================
# 基线偏离度测试
# ============================================================

class TestBaselineDeviation:
    def test_compute_deviation(self):
        """计算 Z-score"""
        deviation = compute_baseline_deviation("user_123")
        assert isinstance(deviation, dict)
        assert len(deviation) > 0

    def test_deviation_has_z_score(self):
        """Z-score 字段存在"""
        deviation = compute_baseline_deviation("user_123")
        for name, info in deviation.items():
            assert "z_score" in info
            assert "age_group" in info
            assert "percentile_approx" in info

    def test_deviation_age_group(self):
        """年龄段字段正确"""
        deviation = compute_baseline_deviation("user_123")
        for name, info in deviation.items():
            assert info["age_group"] in ("12-17", "18-25", "26-35")

    def test_deviation_missing_feature(self):
        """缺失特征返回 None Z-score"""
        pv = PhenotypeVector(
            user_id="u1", time_window_days=14,
            sleep_features=[PhenotypeFeature("sleep_duration_mean", None, 0.0, "missing")],
        )
        deviation = compute_baseline_deviation("u1", phenotype=pv)
        assert deviation["sleep_duration_mean"]["z_score"] is None

    def test_deviation_z_score_range(self):
        """Z-score 值合理"""
        pv = PhenotypeVector(
            user_id="u1", time_window_days=14,
            sleep_features=[PhenotypeFeature("sleep_duration_mean", 7.0, 0.9, "wearable")],
        )
        deviation = compute_baseline_deviation("u1", phenotype=pv)
        z = deviation["sleep_duration_mean"]["z_score"]
        assert z is not None
        assert -5 < z < 5  # 合理的 Z-score 范围


# ============================================================
# 辅助函数测试
# ============================================================

class TestHelpers:
    def test_estimate_feature_std(self):
        """估计特征标准差"""
        std = _estimate_feature_std("sleep_duration_mean")
        assert std > 0

    def test_estimate_unknown_feature(self):
        """未知特征返回默认标准差"""
        std = _estimate_feature_std("unknown_feature")
        assert std == 1.0

    def test_age_group_baselines_complete(self):
        """基线数据完整"""
        for group, baselines in AGE_GROUP_BASELINES.items():
            assert "sleep_duration_mean" in baselines
            assert "phq9_latest" in baselines
