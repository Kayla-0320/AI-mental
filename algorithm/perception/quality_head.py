"""冻结画质头：公开自然图像 MOS 训练的辅助质量分，不替代人工评分。

边界：
- 训练标签是 KonIQ-10k 的众包 MOS（1–5），不是本项目真人盲评，也不是情绪准确率；
- 只在摄像头缩略帧上做辅助质量参考；不参与风险判定、升级或安全阈值；
- 默认影子输出（只记录），是否据此退回文字专家由 ``INNOVATION_QUALITY_GUARD`` 决定。
"""
from __future__ import annotations

import hashlib
import json
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "quality_head"
VISUAL_DIMENSIONS = 512
PIXEL_DESCRIPTORS = 10
_LOCK = threading.RLock()
_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class QualityEstimate:
    """单次三帧的辅助质量结论；MOS 越低表示画面越差。"""

    mos: list[float]
    worst: float
    mean: float
    low_quality: bool
    threshold: float
    model_sha256: str
    limitations: list[str]

    def metrics(self) -> dict[str, float]:
        """转成可写入质量记录的数字字段。"""
        return {"quality_mos_worst": self.worst, "quality_mos_mean": self.mean,
                "quality_mos_low": 1.0 if self.low_quality else 0.0}


def sha256_file(path: Path) -> str:
    """流式计算文件摘要，避免把模型整体读入内存比较。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptors(rgb: np.ndarray) -> np.ndarray:
    """曝光、对比度、梯度与频率统计；与 round18 训练时的实现保持逐项一致。"""
    import cv2

    gray = cv2.cvtColor(cv2.resize(np.asarray(rgb, dtype=np.uint8), (128, 128)), cv2.COLOR_RGB2GRAY).astype(float) / 255
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    hist = np.bincount((gray * 255).astype(np.uint8).ravel(), minlength=256) / gray.size
    return np.array([gray.mean(), gray.std(), np.mean(gray < .05), np.mean(gray > .95),
                     np.log1p(lap.var() * 1000), np.mean(np.hypot(gx, gy)),
                     -np.sum(hist * np.log2(hist + 1e-12)) / 8,
                     *np.asarray(rgb, float).std((0, 1)) / 255], float)


def load(folder: Path | None = None) -> tuple[Any, dict[str, Any]]:
    """按摘要校验后加载模型；路径、mtime 或摘要变化时重新加载。"""
    directory = Path(folder or ARTIFACTS)
    model_path = directory / "model.pkl"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_dimension") != VISUAL_DIMENSIONS + PIXEL_DESCRIPTORS:
        raise ValueError("画质头特征维度与工件不符")
    digest = sha256_file(model_path)
    if digest != manifest.get("model_sha256"):
        raise ValueError("画质头模型摘要不符")
    key = (str(directory.resolve()), model_path.stat().st_mtime_ns, digest)
    with _LOCK:
        if _CACHE.get("key") != key:
            with model_path.open("rb") as stream:
                predictor = pickle.load(stream)
            _CACHE.clear()
            _CACHE.update({"key": key, "predictor": predictor, "manifest": manifest})
        return _CACHE["predictor"], _CACHE["manifest"]


def score(visuals: np.ndarray, frames: Sequence[np.ndarray], *, folder: Path | None = None,
          predictor: Any = None, manifest: dict[str, Any] | None = None) -> QualityEstimate:
    """把 512 维视觉向量与 10 个像素描述拼成 522 维后逐帧打分。"""
    visuals = np.asarray(visuals, dtype=np.float32)
    if visuals.ndim != 2 or visuals.shape[1] != VISUAL_DIMENSIONS or not np.isfinite(visuals).all():
        raise ValueError("画质头需要与帧对齐的 512 维视觉向量")
    if len(frames) != len(visuals):
        raise ValueError("视觉向量与帧数量不一致")
    if not frames:
        raise ValueError("至少需要一帧")
    pixels = np.stack([descriptors(np.asarray(frame)) for frame in frames])
    features = np.column_stack((visuals, pixels))
    if predictor is None:
        predictor, manifest = load(folder)
    manifest = manifest or {}
    threshold = float(manifest.get("low_quality_mos_threshold", 2.5))
    raw = np.asarray(predictor.predict(features), dtype=float).ravel()
    if raw.shape != (len(frames),) or not np.isfinite(raw).all():
        raise ValueError("画质头输出与帧数不一致")
    mos = np.clip(raw, 1.0, 5.0)
    worst = float(mos.min())
    return QualityEstimate(mos=[float(value) for value in mos], worst=worst, mean=float(mos.mean()),
                           low_quality=worst < threshold, threshold=threshold,
                           model_sha256=str(manifest.get("model_sha256", "")),
                           limitations=["public_KonIQ_MOS_not_project_human_rating",
                                        "single_image_resized128_features",
                                        "no_emotion_benefit_measured"])
