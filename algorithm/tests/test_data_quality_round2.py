"""数据审核和质量掩码的工程边界测试，不代表真实噪声检出率。"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from data_quality import assess_modality, audit_manifest
from shared import DatasetSampleRecord, ModalityObservation


def observation(**kwargs: object) -> ModalityObservation:
    """构造可观察质量元数据完整的独立样例。"""
    values = dict(modality="audio", features=[0.2, -0.5], available=True, quality_score=0.9,
                  observed_at=100.0, segment_start=1.0, segment_end=2.0)
    values.update(kwargs)
    return ModalityObservation(**values)


def assess(item: ModalityObservation) -> object:
    """在相同评估时刻和参考片段下运行质量核验。"""
    return assess_modality(item, now=102.0, reference_start=1.0, reference_end=2.0)


def test_clean_signal_remains_usable() -> None:
    report = assess(observation())
    assert report.usable
    assert report.reliability == pytest.approx(0.9)
    assert report.sanitized_features == [0.2, -0.5]
    assert report.reason_codes == []


@pytest.mark.parametrize("kwargs,reason", [
    ({"available": False}, "missing_modality"),
    ({"available": "false"}, "invalid_availability_flag"),
    ({"available": 1}, "invalid_availability_flag"),
    ({"features": []}, "empty_features"),
    ({"features": [math.nan, 1.0]}, "non_finite_or_invalid_feature"),
    ({"features": [math.inf, 1.0]}, "non_finite_or_invalid_feature"),
    ({"features": [True, 1.0]}, "non_finite_or_invalid_feature"),
    ({"quality_score": 0.0}, "zero_quality_score"),
    ({"quality_score": -1.0}, "invalid_quality_score"),
    ({"quality_score": math.nan}, "invalid_quality_score"),
    ({"observed_at": 80.0}, "stale_observation"),
    ({"observed_at": 104.0}, "future_timestamp"),
    ({"observed_at": math.nan}, "invalid_timestamp"),
    ({"observed_at": "now"}, "invalid_timestamp"),
    ({"observed_at": True}, "invalid_timestamp"),
    ({"segment_start": 3.0, "segment_end": 4.0}, "segment_misaligned"),
    ({"segment_start": 2.0, "segment_end": 2.0}, "invalid_segment_duration"),
    ({"segment_start": None}, "incomplete_alignment_metadata"),
    ({"segment_end": math.inf}, "invalid_alignment_metadata"),
    ({"segment_end": "two"}, "invalid_alignment_metadata"),
    ({"segment_end": True}, "invalid_alignment_metadata"),
])
def test_invalid_modality_is_zeroed_with_reason(kwargs: dict[str, object], reason: str) -> None:
    result = assess(observation(**kwargs))
    assert not result.usable
    assert result.reliability == 0
    assert reason in result.reason_codes
    assert all(value == 0 for value in result.sanitized_features)


def test_offline_features_do_not_fabricate_alignment_or_freshness() -> None:
    item = ModalityObservation(modality="visual", features=[1.0, 0.0])
    result = assess_modality(item)
    assert result.usable
    assert set(result.reason_codes) == {"freshness_unverified", "alignment_unverified"}


def test_partial_overlap_adjusts_reliability_only_after_explicit_alignment() -> None:
    item = observation(segment_start=1.5, segment_end=2.5)
    result = assess(item)
    assert result.usable
    assert result.reliability == pytest.approx(0.45)


def test_emotion_disagreement_is_never_a_quality_failure() -> None:
    text = assess(observation(modality="text", features=[1.0, 0.0, 0.0]))
    visual = assess(observation(modality="visual", features=[0.0, 0.0, 1.0]))
    assert text.usable and visual.usable
    assert text.reliability == visual.reliability


@pytest.mark.parametrize("kwargs", [{"minimum_overlap": math.nan}, {"minimum_overlap": 1.1}, {"max_age_seconds": 0}, {"future_tolerance_seconds": -1}])
def test_quality_configuration_must_be_finite_and_in_range(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        assess_modality(observation(), **kwargs)


@pytest.mark.parametrize("value", ["now", True, math.nan])
def test_invalid_reference_time_fails_closed(value: object) -> None:
    result = assess_modality(observation(), now=value, reference_start=1.0, reference_end=2.0)
    assert not result.usable
    assert "invalid_timestamp" in result.reason_codes


@pytest.mark.parametrize("value", ["two", True, math.nan])
def test_invalid_reference_segment_fails_closed(value: object) -> None:
    result = assess_modality(observation(), now=102.0, reference_start=1.0, reference_end=value)
    assert not result.usable
    assert "invalid_alignment_metadata" in result.reason_codes


def test_features_must_be_list_like() -> None:
    with pytest.raises(ValueError):
        assess_modality(observation(features=None))


def records() -> list[DatasetSampleRecord]:
    """构造来源和标签域一致的三份独立切分。"""
    return [DatasetSampleRecord(dataset_id="dataset", sample_id=f"clip-{i}", source_group=f"source-{i}", split=split, label_domain="sentiment3", label=i, content_hashes={"media": f"hash-{i}"}) for i, split in enumerate(("train", "valid", "test"))]


def audit(items: list[DatasetSampleRecord], **kwargs: object) -> object:
    """以固定标签定义审核样本。"""
    return audit_manifest(items, class_count=3, expected_label_domain="sentiment3", **kwargs)


def test_clean_manifest_is_accepted() -> None:
    result = audit(records())
    assert result.valid
    assert result.sample_count == 3
    assert result.split_counts == {"train": 1, "valid": 1, "test": 1}


def test_source_group_reuse_across_split_is_rejected() -> None:
    rows = records()
    rows[2] = replace(rows[2], source_group=rows[0].source_group)
    result = audit(rows)
    assert not result.valid
    assert result.cross_split_source_groups == ["source-0"]


def test_cross_dataset_upstream_overlap_is_detected() -> None:
    rows = records()
    rows[2] = replace(rows[2], dataset_id="mmeverse", source_group=rows[0].source_group)
    assert audit(rows).cross_split_source_groups == ["source-0"]


def test_relaxed_source_protocol_still_reports_overlap() -> None:
    rows = records()
    rows[2] = replace(rows[2], source_group=rows[0].source_group)
    result = audit(rows, require_source_disjoint=False)
    assert result.valid
    assert result.cross_split_source_groups == ["source-0"]


def test_exact_content_across_split_is_rejected() -> None:
    rows = records()
    rows[2] = replace(rows[2], content_hashes=rows[0].content_hashes)
    result = audit(rows)
    assert not result.valid
    assert result.cross_split_content_hashes == ["media:hash-0"]


def test_different_modalities_do_not_collide() -> None:
    rows = records()
    rows[0] = replace(rows[0], content_hashes={"audio": "x"})
    rows[2] = replace(rows[2], content_hashes={"visual": "x"})
    assert audit(rows).valid


@pytest.mark.parametrize("kwargs,reason", [
    ({"label_domain": "clinical_diagnosis"}, "mixed_label_domain"),
    ({"label": 3}, "invalid_label"),
    ({"label": True}, "invalid_label"),
    ({"source_group": ""}, "missing_identity"),
    ({"source_group": "   "}, "missing_identity"),
    ({"split": "validation"}, "invalid_split"),
])
def test_manifest_rejects_invalid_contract(kwargs: dict[str, object], reason: str) -> None:
    rows = records()
    rows[0] = replace(rows[0], **kwargs)
    result = audit(rows)
    assert not result.valid
    assert any(issue.startswith(reason) for issue in result.issues)


def test_duplicate_identity_in_same_split_is_rejected() -> None:
    rows = records()
    rows.append(rows[0])
    assert "duplicate_sample:dataset:clip-0" in audit(rows).issues


def test_empty_manifest_does_not_pass() -> None:
    assert not audit([]).valid


def test_source_identity_type_is_not_silently_accepted() -> None:
    rows = records()
    rows[0] = replace(rows[0], source_group=["video1"])
    with pytest.raises(ValueError):
        audit(rows)
