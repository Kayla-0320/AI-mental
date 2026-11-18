"""以可观察的信号故障判质量，不把跨模态情绪差异认定为错误。"""
from __future__ import annotations

import math
from numbers import Real

from shared import ModalityObservation, ModalityQualityReport


def _finite_real(value: object) -> bool:
    """拒绝布尔、字符串和溢出数值，不让不可控类型逃逸为 TypeError。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def assess_modality(
    observation: ModalityObservation, *, now: float | None = None,
    max_age_seconds: float = 10.0, future_tolerance_seconds: float = 1.0,
    reference_start: float | None = None, reference_end: float | None = None,
    minimum_overlap: float = 0.5,
) -> ModalityQualityReport:
    """产生质量掩码和理由；无时间元数据时显式标记未核验，而不伪称实时对齐。"""
    if not _finite_real(max_age_seconds) or max_age_seconds <= 0:
        raise ValueError("max_age_seconds 必须为正有限数")
    if not _finite_real(future_tolerance_seconds) or future_tolerance_seconds < 0:
        raise ValueError("future_tolerance_seconds 必须非负")
    if not _finite_real(minimum_overlap) or not 0 <= minimum_overlap <= 1:
        raise ValueError("minimum_overlap 必须在[0,1]")
    if not isinstance(observation.modality, str) or not observation.modality.strip():
        raise ValueError("modality 不能为空")
    if not isinstance(observation.features, (list, tuple)):
        raise ValueError("features 必须是一维数值列表")
    reasons: list[str] = []
    usable = observation.available is True
    if not isinstance(observation.available, bool):
        reasons.append("invalid_availability_flag")
    elif not usable:
        reasons.append("missing_modality")
    clean: list[float] = []
    for value in observation.features:
        if not _finite_real(value):
            clean.append(0.0)
            usable = False
            if "non_finite_or_invalid_feature" not in reasons:
                reasons.append("non_finite_or_invalid_feature")
        else:
            clean.append(float(value))
    if not clean:
        usable = False
        reasons.append("empty_features")
    quality = observation.quality_score
    if not _finite_real(quality) or not 0 <= quality <= 1:
        usable = False
        reasons.append("invalid_quality_score")
        quality = 0.0
    elif quality == 0:
        usable = False
        reasons.append("zero_quality_score")
    if (now is not None and not _finite_real(now)) or (observation.observed_at is not None and not _finite_real(observation.observed_at)):
        usable = False
        reasons.append("invalid_timestamp")
    elif observation.observed_at is None or now is None:
        reasons.append("freshness_unverified")
    else:
        if observation.observed_at > now + future_tolerance_seconds:
            usable = False
            reasons.append("future_timestamp")
        elif now - observation.observed_at > max_age_seconds:
            usable = False
            reasons.append("stale_observation")
    segment = (observation.segment_start, observation.segment_end)
    reference = (reference_start, reference_end)
    overlap = 1.0
    if all(value is None for value in (*segment, *reference)):
        reasons.append("alignment_unverified")
    elif any(value is None for value in (*segment, *reference)):
        usable = False
        reasons.append("incomplete_alignment_metadata")
    elif not all(_finite_real(value) for value in (*segment, *reference)):
        usable = False
        reasons.append("invalid_alignment_metadata")
    else:
        start, end = segment
        ref_start, ref_end = reference
        if start >= end or ref_start >= ref_end:
            usable = False
            reasons.append("invalid_segment_duration")
        else:
            overlap = max(0.0, min(end, ref_end) - max(start, ref_start)) / (ref_end - ref_start)
            if overlap < minimum_overlap:
                usable = False
                reasons.append("segment_misaligned")
    if not usable:
        clean = [0.0] * len(clean)
    return ModalityQualityReport(
        modality=observation.modality, usable=bool(usable),
        reliability=float(quality * overlap) if usable else 0.0,
        reason_codes=reasons, sanitized_features=clean,
    )
