"""仅使用真实可用模态的本地文本模型参考，不接管风险或升级条件。"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from shared import ModelObservation, MultimodalFeatureBatch

LABELS = ["negative", "neutral", "positive"]
_LOCK = threading.RLock()
_CACHE: dict[str, Any] = {}


def unavailable(source_id: str, source_sha: str, observed_at: str, reason: str) -> ModelObservation:
    """缺少实际模型证据时显式返回不可用，不填充中性概率。"""
    return ModelObservation(False, "sentiment_polarity_3", LABELS, "", "", {"status": "uncalibrated", "method": "not_applied"},
                            {"source_message_id": source_id, "source_sha256": source_sha}, [], observed_at,
                            ["model_reference_only", "not_self_report", "not_clinical_risk"], reason)


def infer_observation(text: str, source_id: str, source_sha: str, observed_at: str,
                      now: datetime | None = None) -> ModelObservation:
    """核对出处和时间后实际编码；其余模态缺失，不伪造音视频观测。"""
    if not isinstance(text, str) or not text.strip() or len(text) > 4000 or not isinstance(source_id, str) or not source_id.strip() or len(source_id) > 160:
        raise ValueError("文本或消息来源无效")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != source_sha:
        raise ValueError("消息出处摘要不匹配")
    try:
        stamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("时间必须含时区")
        clock = now or datetime.now(timezone.utc)
        age = (clock - stamp).total_seconds()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("消息时间无效") from exc
    if age < -60 or age > 48 * 3600:
        return unavailable(source_id, source_sha, observed_at, "source_stale_or_future")
    directory = Path(os.environ.get("ROUND3_FUSION_ARTIFACT", str(Path(__file__).resolve().parents[2] / "artifacts/round3_fusion")))
    encoder_dir = Path(os.environ.get("ROUND3_VLM_ENCODER", "D:/ai/models/encoders/chinese-clip-vit-base-patch16"))
    try:
        with _LOCK:
            manifest_bytes = (directory / "manifest.json").read_bytes()
            key = hashlib.sha256(manifest_bytes).hexdigest() + str((directory / "weights.pt").stat().st_mtime_ns)
            key += str(encoder_dir.resolve()) + str((encoder_dir / "pytorch_model.bin").stat().st_mtime_ns)
            if _CACHE.get("key") != key:
                from perception.knowledge_moe.features import get_clip_encoder
                from perception.knowledge_moe.runtime import FusionRuntime

                runtime = FusionRuntime(directory)
                if runtime.meta["class_names"] != LABELS or "text" not in runtime.meta["dimensions"]:
                    raise ValueError("研究任务与实时极性参考不匹配")
                encoder = get_clip_encoder(encoder_dir.resolve(), device=os.environ.get("ROUND3_ENCODER_DEVICE", "cpu"))
                if runtime.meta.get("live_text_encoder_fingerprint") != encoder.fingerprint:
                    raise ValueError("实时文本编码器与训练工件身份不一致")
                _CACHE.clear()
                _CACHE.update({"key": key, "runtime": runtime, "encoder": encoder})
            runtime, encoder = _CACHE["runtime"], _CACHE["encoder"]
            text_features = encoder.encode_text([text])
            dimensions = runtime.meta["dimensions"]
            if text_features.shape != (1, dimensions["text"]) or not np.isfinite(text_features).all():
                raise ValueError("实时文本编码维数或数值非法")
            features = {name: text_features.tolist() if name == "text" else [[0.0] * dim] for name, dim in dimensions.items()}
            observed = {name: [name == "text"] for name in dimensions}
            quality = {name: [1.0 if name == "text" else 0.0] for name in dimensions}
            result = runtime.predict(MultimodalFeatureBatch([source_id], features, observed, quality,
                                      runtime.meta["task"], LABELS, runtime.meta["encoder_fingerprints"], raw_text=[text]))
            if result.abstained[0]:
                return unavailable(source_id, source_sha, observed_at, "model_abstained")
            text_calibration = runtime.meta.get("text_only_calibration", {})
            validated = (isinstance(text_calibration, dict) and text_calibration.get("status") == "calibrated"
                         and text_calibration.get("fit_split") == "calibration"
                         and text_calibration.get("condition") == "text_present_audio_visual_missing"
                         and type(text_calibration.get("n")) is int and text_calibration["n"] > 0
                         and type(text_calibration.get("temperature")) in (int, float)
                         and np.isfinite(text_calibration["temperature"])
                         and .01 <= text_calibration["temperature"] <= 100)
            probabilities = result.probabilities[0]
            if validated:
                # 还原softmax相对logits后应用匹配缺失模式的独立校准温度。
                z = np.log(np.maximum(probabilities, 1e-30)) * runtime.temperature / text_calibration["temperature"]
                probabilities = (np.exp(z - z.max()) / np.exp(z - z.max()).sum()).tolist()
            calibration = {"status": "calibrated" if validated else "uncalibrated",
                           "method": "text_only_source_temperature" if validated else "source_temperature_not_validated_for_text_only"}
            return ModelObservation(True, "sentiment_polarity_3", LABELS, runtime.meta["task"],
                                    result.model_id, calibration,
                                    {"source_message_id": source_id, "source_sha256": source_sha},
                                    probabilities, observed_at,
                                    ["text_only_audio_visual_missing", "source_domain_not_target_population_validation",
                                     "not_self_report", "not_clinical_risk", "calibration_fitted_on_source_domain",
                                     "long_text_may_be_truncated_by_encoder"])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        # 只提供稳定错误类别，模型路径/原文不进入对外错误。
        return unavailable(source_id, source_sha, observed_at, "model_unavailable_or_artifact_mismatch")
