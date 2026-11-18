"""画质头回归：工件摘要、特征维度、MOS 范围与阈值逻辑。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from perception import quality_head


def frames(seed: int = 7) -> list[np.ndarray]:
    """三张确定性随机帧，避免依赖真实摄像头。"""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(3)]


def test_manifest_and_model_digest_match() -> None:
    manifest = json.loads((quality_head.ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["feature_dimension"] == quality_head.VISUAL_DIMENSIONS + quality_head.PIXEL_DESCRIPTORS
    assert quality_head.sha256_file(quality_head.ARTIFACTS / "model.pkl") == manifest["model_sha256"]
    predictor, loaded = quality_head.load()
    assert loaded["model_sha256"] == manifest["model_sha256"]
    assert hasattr(predictor, "predict")


def test_descriptors_shape_is_stable() -> None:
    assert quality_head.descriptors(frames()[0]).shape == (quality_head.PIXEL_DESCRIPTORS,)
    assert np.isfinite(quality_head.descriptors(frames()[0])).all()


def test_score_returns_mos_in_five_point_range() -> None:
    visuals = np.random.default_rng(1).normal(size=(3, quality_head.VISUAL_DIMENSIONS)).astype(np.float32)
    estimate = quality_head.score(visuals, frames())
    assert len(estimate.mos) == 3
    assert all(1.0 <= value <= 5.0 for value in estimate.mos)
    assert estimate.worst == min(estimate.mos)
    assert estimate.low_quality == (estimate.worst < estimate.threshold)
    assert "public_KonIQ_MOS_not_project_human_rating" in estimate.limitations


def test_low_quality_flag_follows_threshold_not_emotion() -> None:
    """只验证阈值逻辑；桩模型不冒充真实画质结论。"""

    class Stub:
        def predict(self, values: np.ndarray) -> np.ndarray:
            assert values.shape[1] == quality_head.VISUAL_DIMENSIONS + quality_head.PIXEL_DESCRIPTORS
            return np.array([1.2, 3.0, 4.0])

    estimate = quality_head.score(np.zeros((3, quality_head.VISUAL_DIMENSIONS), np.float32), frames(),
                                  predictor=Stub(), manifest={"low_quality_mos_threshold": 2.5, "model_sha256": "stub"})
    assert estimate.low_quality is True and estimate.worst == 1.2 and estimate.mean == pytest.approx(2.7333, abs=1e-3)


def test_rejects_vectors_that_do_not_match_frames() -> None:
    with pytest.raises(ValueError):
        quality_head.score(np.zeros((3, 4), np.float32), frames())
    with pytest.raises(ValueError):
        quality_head.score(np.zeros((2, quality_head.VISUAL_DIMENSIONS), np.float32), frames())
