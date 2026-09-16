"""
融合模块单元测试

验证三种融合策略的正确性、降级逻辑和评估流程。
"""
import numpy as np
import pytest

from shared.dataclasses import EmotionResult
from perception.fusion.fusion import (
    StackingFusion,
    fuse_dynamic_weighting,
    fuse_multimodal,
    fuse_weighted_average,
    load_fusion_config,
)
from perception.fusion.evaluate import evaluate_fusion, generate_report


# ============================================================
# 测试数据
# ============================================================

@pytest.fixture
def sample_probs():
    """生成测试用概率分布"""
    np.random.seed(42)
    n_samples = 20
    n_classes = 5

    text_probs = np.random.dirichlet(np.ones(n_classes), size=n_samples)
    audio_probs = np.random.dirichlet(np.ones(n_classes), size=n_samples)
    labels = np.random.randint(0, n_classes, size=n_samples)

    return text_probs, audio_probs, labels


@pytest.fixture
def config():
    """加载融合配置"""
    return load_fusion_config()


# ============================================================
# 加权平均测试
# ============================================================

class TestWeightedAverage:

    def test_output_shape(self, sample_probs, config):
        """融合后形状必须与输入一致"""
        text_probs, audio_probs, _ = sample_probs
        fused = fuse_weighted_average(text_probs, audio_probs, config)
        assert fused.shape == text_probs.shape

    def test_output_sums_to_one(self, sample_probs, config):
        """融合后概率和必须为 1"""
        text_probs, audio_probs, _ = sample_probs
        fused = fuse_weighted_average(text_probs, audio_probs, config)
        np.testing.assert_allclose(fused.sum(axis=1), 1.0, atol=1e-6)

    def test_audio_none_fallback(self, sample_probs, config):
        """audio_probs 为 None 时必须降级为文本单模态"""
        text_probs, _, _ = sample_probs
        fused = fuse_weighted_average(text_probs, None, config)
        np.testing.assert_array_equal(fused, text_probs)

    def test_single_sample(self, config):
        """支持单样本 (n_classes,) 输入"""
        text_probs = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
        audio_probs = np.array([0.2, 0.1, 0.2, 0.2, 0.3])
        fused = fuse_weighted_average(text_probs, audio_probs, config)
        assert fused.shape == (5,)
        np.testing.assert_allclose(fused.sum(), 1.0, atol=1e-6)


# ============================================================
# Stacking 测试
# ============================================================

class TestStacking:

    def test_fit_predict(self, sample_probs, config):
        """fit 后 predict 必须返回概率分布"""
        text_probs, audio_probs, labels = sample_probs
        stacking = StackingFusion(config)
        stacking.fit(text_probs, audio_probs, labels)

        assert stacking.is_fitted
        result = stacking.predict(text_probs, audio_probs)
        assert result.shape[0] == text_probs.shape[0]
        assert result.shape[1] == text_probs.shape[1]

    def test_not_fitted_fallback(self, sample_probs, config):
        """未训练时必须降级为文本单模态"""
        text_probs, audio_probs, _ = sample_probs
        stacking = StackingFusion(config)

        assert not stacking.is_fitted
        result = stacking.predict(text_probs, audio_probs)
        np.testing.assert_array_equal(result, text_probs)

    def test_audio_none_fallback(self, sample_probs, config):
        """audio_probs 为 None 时必须降级"""
        text_probs, _, labels = sample_probs
        stacking = StackingFusion(config)
        stacking.fit(text_probs, text_probs, labels)

        result = stacking.predict(text_probs, None)
        np.testing.assert_array_equal(result, text_probs)


# ============================================================
# 动态加权测试
# ============================================================

class TestDynamicWeighting:

    def test_output_shape(self, sample_probs, config):
        """融合后形状必须与输入一致"""
        text_probs, audio_probs, _ = sample_probs
        fused = fuse_dynamic_weighting(
            text_probs[0], audio_probs[0], 0.8, 0.7, config
        )
        assert fused.shape == text_probs[0].shape

    def test_audio_none_fallback(self, sample_probs, config):
        """audio_probs 为 None 时必须降级"""
        text_probs, _, _ = sample_probs
        fused = fuse_dynamic_weighting(text_probs[0], None, 0.8, None, config)
        np.testing.assert_array_equal(fused, text_probs[0])

    def test_low_text_confidence(self, sample_probs, config):
        """文本置信度低于阈值时必须仅用语音"""
        text_probs, audio_probs, _ = sample_probs
        fused = fuse_dynamic_weighting(
            text_probs[0], audio_probs[0], 0.1, 0.8, config
        )
        np.testing.assert_array_equal(fused, audio_probs[0])

    def test_low_audio_confidence(self, sample_probs, config):
        """语音置信度低于阈值时必须仅用文本"""
        text_probs, audio_probs, _ = sample_probs
        fused = fuse_dynamic_weighting(
            text_probs[0], audio_probs[0], 0.8, 0.1, config
        )
        np.testing.assert_array_equal(fused, text_probs[0])


# ============================================================
# 统一接口测试
# ============================================================

class TestFuseMultimodal:

    def _make_emotion_result(self, probs, confidence=0.8, audio_risk=None):
        import time
        return EmotionResult(
            text_emotion_probs=probs.tolist(),
            audio_risk_prob=audio_risk,
            confidence=confidence,
            timestamp=time.time(),
            evidence=["测试证据"],
        )

    def test_returns_emotion_result(self, config):
        """必须返回 EmotionResult"""
        text_probs = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
        audio_probs = np.array([0.2, 0.1, 0.2, 0.2, 0.3])

        text_result = self._make_emotion_result(text_probs)
        audio_result = self._make_emotion_result(audio_probs, audio_risk=0.5)

        fused = fuse_multimodal(text_result, audio_result, config=config)
        assert isinstance(fused, EmotionResult)

    def test_audio_none_degradation(self, config):
        """audio_result 为 None 时必须降级"""
        text_probs = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
        text_result = self._make_emotion_result(text_probs)

        fused = fuse_multimodal(text_result, None, config=config)
        assert isinstance(fused, EmotionResult)
        assert fused.audio_risk_prob is None
        assert any("降级" in e for e in fused.evidence)

    def test_all_strategies(self, config):
        """三种策略都必须能正常执行"""
        text_probs = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
        audio_probs = np.array([0.2, 0.1, 0.2, 0.2, 0.3])

        text_result = self._make_emotion_result(text_probs)
        audio_result = self._make_emotion_result(audio_probs)

        for strategy in ["weighted_average", "stacking", "dynamic"]:
            fused = fuse_multimodal(
                text_result, audio_result,
                strategy=strategy, config=config,
            )
            assert isinstance(fused, EmotionResult)
            assert len(fused.text_emotion_probs) == 5


# ============================================================
# 评估模块测试
# ============================================================

class TestEvaluate:

    def test_evaluate_returns_all_strategies(self, sample_probs, config):
        """evaluate_fusion 必须返回所有策略的指标"""
        text_probs, audio_probs, labels = sample_probs
        results = evaluate_fusion(text_probs, audio_probs, labels, config)

        expected_keys = {
            "text_only", "audio_only",
            "weighted_average", "stacking", "dynamic_weighting",
        }
        assert set(results.keys()) == expected_keys

    def test_metrics_range(self, sample_probs, config):
        """所有指标必须在合理范围内"""
        text_probs, audio_probs, labels = sample_probs
        results = evaluate_fusion(text_probs, audio_probs, labels, config)

        for strategy, metrics in results.items():
            assert 0.0 <= metrics["auc"] <= 1.0, f"{strategy} AUC 越界"
            assert 0.0 <= metrics["f1_macro"] <= 1.0, f"{strategy} F1 macro 越界"
            assert 0.0 <= metrics["f1_weighted"] <= 1.0, f"{strategy} F1 weighted 越界"

    def test_generate_report(self, sample_probs, config):
        """generate_report 必须输出 Markdown 字符串"""
        text_probs, audio_probs, labels = sample_probs
        results = evaluate_fusion(text_probs, audio_probs, labels, config)
        report = generate_report(results)

        assert "# 多模态融合策略对比报告" in report
        assert "加权平均" in report
        assert "Stacking" in report
        assert "动态加权" in report
        assert "AUC" in report
