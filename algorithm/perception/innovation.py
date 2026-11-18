"""业务图文推理：真实CLIP编码、原始像素质量与合格专家分派。"""
from __future__ import annotations

import base64
from dataclasses import asdict
import io
import os
import re
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from data_quality.raw_signals import assess_visual_signal
from shared import InnovationObservation, MultimodalFeatureBatch, RawSignalQuality

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
_LOCK = threading.RLock()
_CACHE: dict[str, Any] = {}


def visual_quality(frames: np.ndarray) -> RawSignalQuality:
    """复用第十七轮中心均匀块代理，不接收标签或模型概率。"""
    import cv2

    result = assess_visual_signal(frames)
    if not len(frames):
        return result
    blocked = []
    for frame in frames:
        gray = cv2.cvtColor(cv2.resize(frame.astype(np.uint8), (128, 128)), cv2.COLOR_RGB2GRAY)
        center = gray[32:96, 32:96]
        outside = np.concatenate((gray[:24].ravel(), gray[104:].ravel(),
                                  gray[24:104, :24].ravel(), gray[24:104, 104:].ravel()))
        blocked.append(np.std(center) < 2 and 20 < float(center.mean()) < 235 and np.std(outside) > 10)
    fraction = float(np.mean(blocked))
    result.metrics["uniform_center_block_fraction"] = fraction
    if fraction >= .8:
        result.quality_score = min(result.quality_score, .25)
        result.reason_codes.append("uniform_center_occlusion_proxy")
    result.limitations.append("uniform_center_proxy_not_all_occlusions_or_human_quality_gold")
    return result


def decode_frames(payload: list[dict], now_ms: float) -> np.ndarray:
    """最多三张有时序的新鲜JPEG缩略图，解码前检查尺寸和资源上限。"""
    from PIL import Image

    if not isinstance(payload, list) or len(payload) != 3:
        raise ValueError("需要三张实际采样帧")
    frames, stamps = [], []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("帧格式无效")
        stamp, data = item.get("sampledAt"), item.get("jpeg")
        if type(stamp) not in (int, float) or not np.isfinite(stamp) or not 0 <= now_ms - stamp <= 10000:
            raise ValueError("帧已过期或来自未来")
        if not isinstance(data, str) or not data.startswith("data:image/jpeg;base64,") or len(data) > 100000:
            raise ValueError("帧编码或大小无效")
        try:
            with Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1], validate=True))) as image:
                if image.format != "JPEG" or not 32 <= image.width <= 512 or not 32 <= image.height <= 512:
                    raise ValueError("帧尺寸无效")
                frames.append(np.asarray(image.convert("RGB").resize((224, 224))))
        except (OSError, TypeError) as exc:
            raise ValueError("帧解码失败") from exc
        stamps.append(stamp)
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("帧时间必须递增")
    return np.stack(frames)


def _quality_guard_mode() -> str:
    """默认只记录画质头分数；只有显式设置时才用它的结论退回文字专家。"""
    value = os.environ.get("INNOVATION_QUALITY_GUARD", "shadow").strip().lower()
    if value in {"enforce", "1", "on", "true"}:
        return "enforce"
    if value in {"off", "0", "false"}:
        return "off"
    return "shadow"


def frame_quality(frames: np.ndarray, visuals: np.ndarray, result: InnovationObservation) -> dict[str, Any] | None:
    """用冻结画质头给三帧打分；失败时保留规则质量，不影响主流程。"""
    try:
        from perception.quality_head import score

        estimate = score(visuals, [np.asarray(frame) for frame in frames])
    except Exception:  # noqa: BLE001 - 画质头不可用时不能影响辅助参考
        return None
    result.quality["visual"]["metrics"].update(estimate.metrics())
    notes = ["quality_mos_from_public_image_ratings_not_project_human_gold", *estimate.limitations]
    for note in notes:
        if note not in result.warnings:
            result.warnings.append(note)
    return {"low": estimate.low_quality, "worst": estimate.worst, "threshold": estimate.threshold}


def _models() -> tuple[Any, Any]:
    from perception.knowledge_moe.features import get_clip_encoder
    from perception.knowledge_moe.runtime import FusionRuntime

    folder = Path(os.environ.get("INNOVATION_FUSION_ARTIFACT", str(ARTIFACTS / "innovation_fusion")))
    encoder_path = Path(os.environ.get("ROUND3_VLM_ENCODER", "D:/ai/models/encoders/chinese-clip-vit-base-patch16"))
    key = (str(folder.resolve()), (folder / "manifest.json").stat().st_mtime_ns,
           (folder / "weights.pt").stat().st_mtime_ns, str(encoder_path.resolve()))
    if _CACHE.get("key") != key:
        runtime = FusionRuntime(folder)
        encoder = get_clip_encoder(encoder_path.resolve(), device=os.environ.get("ROUND3_ENCODER_DEVICE", "cpu"))
        if runtime.meta.get("live_text_encoder_fingerprint") != encoder.fingerprint:
            raise ValueError("实际编码器与部署权重不匹配")
        expected = {"text": encoder.fingerprint + ";component=text_l2_v1",
                    "visual": encoder.fingerprint + ";component=vision_l2_mean3_v1"}
        if runtime.meta["class_names"] != ["negative", "neutral", "positive"] or any(
                runtime.meta["encoder_fingerprints"].get(k) != v for k, v in expected.items()):
            raise ValueError("部署模型的任务或图文编码契约不匹配")
        _CACHE.clear()
        _CACHE.update(key=key, runtime=runtime, encoder=encoder)
    return _CACHE["runtime"], _CACHE["encoder"]


def observe(text: str, frames: list[dict] | None = None) -> InnovationObservation:
    """提供独立三类参考，不覆盖五类情绪、风险、量表或升级。"""
    result = InnovationObservation(warnings=["not_self_report_or_clinical_risk", "source_domain_only",
        "knowledge_user_attestation_is_not_independent_expert_review", "live_sampling_differs_from_clip_quantiles"])
    if os.environ.get("INNOVATION_ENABLED", "1") == "0":
        result.reason = "disabled"
        return result
    if not isinstance(text, str) or len(text) > 4000 or not text.strip():
        result.reason = "text_unavailable_or_too_long"
        return result
    if not any(char.isalnum() for char in re.sub(r"https?://\S+", "", text)):
        result.reason = "no_linguistic_evidence"
        return result
    images = None
    if frames:
        try:
            images = decode_frames(frames, time.time() * 1000)
            quality = visual_quality(images)
            result.quality["visual"] = asdict(quality)
            if not quality.usable or quality.quality_score <= .25:
                images = None
                result.warnings.append("visual_quality_fallback_to_text")
        except (ValueError, TypeError, OSError):
            images = None
            result.warnings.append("visual_invalid_or_stale_fallback_to_text")
    try:
        with _LOCK:
            runtime, encoder = _models()
            text_features = encoder.encode_text([text])
            visual = None
            if images is not None:
                from PIL import Image
                encoded = encoder.encode([Image.fromarray(frame) for frame in images], [text] * 3)
                visual = encoded["visual"].mean(axis=0, keepdims=True)
                guard = _quality_guard_mode()
                if guard != "off":
                    scored = frame_quality(images, encoded["visual"], result)
                    if scored and scored["low"] and guard == "enforce":
                        visual = None
                        images = None
                        result.warnings.append("quality_model_fallback_to_text")
            dimensions = runtime.meta["dimensions"]
            features = {name: np.zeros((1, size), dtype=np.float32) for name, size in dimensions.items()}
            features["text"] = text_features
            if visual is not None:
                features["visual"] = visual
            present = {name: [name == "text" or name == "visual" and visual is not None] for name in dimensions}
            quality_scores = {name: [1.0 if present[name][0] else 0.0] for name in dimensions}
            if visual is not None:
                quality_scores["visual"] = [result.quality["visual"]["quality_score"]]
            prediction = runtime.predict(MultimodalFeatureBatch(["current-turn"],
                {k: v.tolist() for k, v in features.items()}, present, quality_scores,
                runtime.meta["task"], result.labels, runtime.meta["encoder_fingerprints"], [text]))
            if prediction.abstained[0]:
                result.reason = "experts_abstained"
                return result
            result.available = True
            result.model_id = prediction.model_id
            result.probabilities = prediction.probabilities[0]
            if visual is None:
                calibration = runtime.meta.get("text_only_calibration", {})
                if calibration.get("status") == "calibrated":
                    z = np.log(np.maximum(result.probabilities, 1e-30)) * runtime.temperature / calibration["temperature"]
                    result.probabilities = (np.exp(z - z.max()) / np.exp(z - z.max()).sum()).tolist()
            result.expert_names = prediction.expert_names
            result.expert_weights = prediction.expert_weights[0]
            result.observed_modalities = [k for k in present if present[k][0]]
            result.calibration_status = "source_calibration_not_validated_for_live_TV" if visual is not None else runtime.meta.get("text_only_calibration", {}).get("status", "uncalibrated")
            if runtime.knowledge is not None:
                result.knowledge_matches = runtime.knowledge.retrieve(text)
            return result
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError):
        result.reason = "model_unavailable_or_artifact_mismatch"
        return result


def prompt_reference(result: InnovationObservation) -> str:
    """把辅助参考注入普通支持提示，不推断隐藏情绪或因果。"""
    if not result.available:
        return ""
    label = result.labels[int(np.argmax(result.probabilities))]
    return ("\n# 本轮辅助情感参考\n冻结图文专家对当前表达的三类参考为 " + label +
            "。仅供措辞参考，以用户原话和修正为准，不据此判断心理问题、推测掩饰、解释他人动机或改变支持目标。"
            "不要向用户复述内部分类或概率。知识索引仅用于模型概念检索，不能作为个体因果结论。\n")
