"""
面部表情 + 行为模式 + 对抗去偏见 测试
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

# 确保 algorithm 根目录在 sys.path
ALGO_ROOT = Path(__file__).resolve().parent.parent
if str(ALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGO_ROOT))


# ============================================================
# 面部表情测试
# ============================================================

class TestFacialFeatures:
    """面部特征提取测试"""

    def test_empty_landmarks(self):
        from perception.face.facial_expression import extract_facial_features
        features = extract_facial_features([])
        assert features.frame_count == 0
        assert features.detection_confidence == 0.0

    def test_basic_extraction(self):
        from perception.face.facial_expression import extract_facial_features
        # 生成 68 点假关键点
        landmarks = np.zeros((68, 2))
        # 左眼
        landmarks[36:42] = [[100, 100], [110, 95], [120, 93], [130, 95], [120, 100], [110, 102]]
        # 右眼
        landmarks[42:48] = [[150, 100], [160, 95], [170, 93], [180, 95], [170, 100], [160, 102]]
        # 嘴巴
        landmarks[48:68] = [[110, 150], [115, 148], [120, 146], [125, 145], [130, 144],
                            [135, 144], [140, 145], [145, 146], [150, 148], [155, 150],
                            [150, 155], [145, 157], [140, 158], [135, 159], [130, 159],
                            [125, 158], [120, 157], [115, 155], [130, 150], [130, 145]]
        # 眉毛
        landmarks[17:27] = [[95, 85], [105, 80], [115, 78], [125, 80], [135, 78],
                            [145, 78], [155, 80], [165, 78], [175, 80], [185, 85]]
        # 鼻子和其他
        landmarks[27:37] = [[140, 90], [140, 100], [140, 110], [140, 120], [140, 130],
                            [130, 135], [135, 137], [140, 138], [145, 137], [150, 135]]
        landmarks[0] = [80, 120]
        landmarks[8] = [140, 180]
        landmarks[16] = [200, 120]

        sequence = [landmarks.copy() for _ in range(20)]
        features = extract_facial_features(sequence, fps=30.0)

        assert features.frame_count == 20
        assert features.fps == 30.0
        assert features.eye_aspect_ratio >= 0
        assert features.mouth_width_ratio >= 0
        assert features.detection_confidence > 0

    def test_emotion_risk_mapping(self):
        from perception.face.facial_expression import FacialFeatures, features_to_emotion_risk
        features = FacialFeatures(
            au1_inner_brow_raiser=0.6,
            au4_brow_lowerer=0.7,
            au15_lip_corner_depressor=0.5,
            micro_expression_count=3,
            detection_confidence=0.9,
            frame_count=100,
        )
        result = features_to_emotion_risk(features)
        assert result["risk_score"] >= 0
        assert result["risk_label"] in ("low_risk", "moderate_risk", "high_risk")
        assert "source" in result["metadata"]
        assert result["metadata"]["source"] == "facial_expression"

    def test_summary(self):
        from perception.face.facial_expression import FacialFeatures, features_to_summary
        features = FacialFeatures(
            au6_cheek_raiser=0.8,
            au12_lip_corner_puller=0.9,
            detection_confidence=0.85,
            frame_count=50,
        )
        summary = features_to_summary(features)
        assert "dominant_emotion" in summary
        assert "risk_score" in summary
        assert "head_pose" in summary


# ============================================================
# 行为模式测试
# ============================================================

class TestBehaviorPattern:
    """行为模式分析测试"""

    def _make_events(self, num=50, bias="normal"):
        from perception.behavior.behavior_pattern import BehaviorEvent
        events = []
        base_time = datetime(2026, 9, 1, 10, 0, 0)
        for i in range(num):
            if bias == "late_night":
                hour = np.random.choice([0, 1, 2, 3, 4])
            elif bias == "normal":
                hour = np.random.randint(8, 22)
            else:
                hour = np.random.randint(0, 24)

            events.append(BehaviorEvent(
                timestamp=base_time + timedelta(hours=i * 6, minutes=np.random.randint(0, 60)),
                event_type=np.random.choice(["chat", "assessment", "healing", "community"]),
                duration_seconds=np.random.uniform(60, 600),
                message_count=np.random.randint(1, 20),
                typing_speed=np.random.uniform(20, 80),
                message_length_avg=np.random.uniform(10, 100),
            ))
        return events

    def test_empty_events(self):
        from perception.behavior.behavior_pattern import extract_behavior_features
        features = extract_behavior_features([])
        assert features.total_events == 0
        assert features.observation_days == 14

    def test_basic_extraction(self):
        from perception.behavior.behavior_pattern import extract_behavior_features
        events = self._make_events(50)
        features = extract_behavior_features(events, observation_days=14)
        assert features.total_events == 50
        assert features.observation_days == 14
        assert 0 <= features.late_night_ratio <= 1
        assert 0 <= features.circadian_regularity <= 1
        assert 0 <= features.social_engagement <= 1

    def test_late_night_bias(self):
        from perception.behavior.behavior_pattern import extract_behavior_features, BehaviorEvent
        # 直接生成深夜时段的事件（全部在 0-4 点）
        events = []
        base = datetime(2026, 9, 1, 1, 0, 0)  # 凌晨 1 点开始
        for i in range(100):
            # 每个事件间隔 2 小时，但都在 0-4 点范围内循环
            hour_offset = (i * 2) % 5  # 0, 2, 4, 1, 3, 0, 2, ...
            events.append(BehaviorEvent(
                timestamp=base.replace(hour=hour_offset) + timedelta(days=i // 5),
                event_type="chat",
                duration_seconds=120,
            ))
        features = extract_behavior_features(events)
        # 所有事件都在凌晨 0-4 点
        assert features.late_night_ratio > 0.8

    def test_emotion_risk(self):
        from perception.behavior.behavior_pattern import BehaviorFeatures, behavior_to_emotion_risk
        features = BehaviorFeatures(
            late_night_ratio=0.5,
            social_engagement=0.05,
            avoidance_score=0.7,
            circadian_regularity=0.1,
            session_frequency_change=-0.6,
            typing_speed_trend=-0.4,
            message_length_trend=-0.3,
            behavior_change_score=0.6,
            data_quality=0.9,
        )
        result = behavior_to_emotion_risk(features)
        assert result["risk_score"] > 0.2
        assert "source" in result["metadata"]

    def test_summary(self):
        from perception.behavior.behavior_pattern import extract_behavior_features, behavior_to_summary
        events = self._make_events(30)
        features = extract_behavior_features(events)
        summary = behavior_to_summary(features)
        assert "risk_score" in summary
        assert "recommendations" in summary
        assert isinstance(summary["recommendations"], list)
        assert len(summary["recommendations"]) > 0


# ============================================================
# 对抗去偏见测试
# ============================================================

class TestAdversarialDebias:
    """对抗去偏见模块测试"""

    def test_grl(self):
        from perception.text_emotion.adversarial_debias import GradientReversalLayer
        grl = GradientReversalLayer(lambda_val=1.0)
        x = torch.randn(4, 10, requires_grad=True)
        y = grl(x)
        # 前向传播应等于输入
        assert torch.allclose(x, y)

    def test_grl_backward(self):
        from perception.text_emotion.adversarial_debias import GradientReversalLayer
        grl = GradientReversalLayer(lambda_val=2.0)
        x = torch.randn(4, 10, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()
        # 反向传播梯度应反转
        assert x.grad is not None
        assert torch.allclose(x.grad, -2.0 * torch.ones_like(x.grad))

    def test_adversarial_config(self):
        from perception.text_emotion.adversarial_debias import AdversarialConfig
        config = AdversarialConfig()
        assert config.num_emotion_labels == 5
        assert config.seed == 42

    def test_biased_data_generation(self):
        from perception.text_emotion.adversarial_debias import generate_biased_fake_data
        texts, labels, sensitive = generate_biased_fake_data(num_per_class=10, seed=42)
        assert len(texts) == 50  # 5 classes * 10
        assert len(labels) == 50
        assert len(sensitive) == 50
        assert all(0 <= l <= 4 for l in labels)
        assert all(0 <= s <= 2 for s in sensitive)

    def test_disparity_computation(self):
        from perception.text_emotion.adversarial_debias import _compute_disparity
        preds = [1, 2, 3, 0, 0, 1, 2, 3]
        sensitive = [0, 0, 0, 0, 1, 1, 1, 1]
        result = _compute_disparity(preds, sensitive, 5)
        assert "disparity" in result
        assert "group_rates" in result
        assert 0 <= result["disparity"] <= 1

    def test_fairness_report_structure(self):
        from perception.text_emotion.adversarial_debias import FairnessReport
        report = FairnessReport(
            overall_accuracy=0.85,
            demographic_parity_diff=0.03,
            equalized_odds_diff=0.05,
        )
        assert report.overall_accuracy == 0.85
        assert report.demographic_parity_diff == 0.03
