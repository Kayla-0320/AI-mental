"""固定标签域和来源边界，识别切分中的实体或内容泄漏。"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from shared import DatasetAuditReport, DatasetSampleRecord


def audit_manifest(
    samples: Sequence[DatasetSampleRecord], *, class_count: int,
    expected_label_domain: str, require_source_disjoint: bool = True,
) -> DatasetAuditReport:
    """审核统一样本清单；相同原始来源跨数据集时也必须使用相同规范来源键。"""
    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count < 2:
        raise ValueError("class_count 必须至少为2")
    if not isinstance(expected_label_domain, str) or not expected_label_domain.strip():
        raise ValueError("必须显式指定标签域，不能混并情感极性、情绪类别与诊断")
    issues: list[str] = []
    seen: set[tuple[str, str]] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for index, sample in enumerate(samples):
        if not isinstance(sample, DatasetSampleRecord):
            raise ValueError("清单条目必须使用共享 DatasetSampleRecord")
        if not all(isinstance(value, str) for value in (sample.dataset_id, sample.sample_id, sample.source_group, sample.split, sample.label_domain)):
            raise ValueError("身份、来源、切分和标签域必须为字符串")
        if not isinstance(sample.content_hashes, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in sample.content_hashes.items()):
            raise ValueError("内容摘要必须为字符串键值字典")
        key = (sample.dataset_id, sample.sample_id)
        if key in seen:
            issues.append(f"duplicate_sample:{sample.dataset_id}:{sample.sample_id}")
        seen.add(key)
        if not all(value.strip() for value in (sample.dataset_id, sample.sample_id, sample.source_group)):
            issues.append(f"missing_identity:{index}")
        if sample.split not in {"train", "valid", "test"}:
            issues.append(f"invalid_split:{sample.sample_id}")
        split_counts[sample.split] += 1
        if sample.label_domain != expected_label_domain:
            issues.append(f"mixed_label_domain:{sample.sample_id}")
        if isinstance(sample.label, bool) or not isinstance(sample.label, int) or not 0 <= sample.label < class_count:
            issues.append(f"invalid_label:{sample.sample_id}")
        if sample.source_group:
            group_splits[sample.source_group].add(sample.split)
        for modality, fingerprint in sample.content_hashes.items():
            if fingerprint:
                hash_splits[f"{modality}:{fingerprint}"].add(sample.split)
    crossed_groups = sorted(key for key, splits in group_splits.items() if len(splits) > 1)
    crossed_hashes = sorted(key for key, splits in hash_splits.items() if len(splits) > 1)
    if require_source_disjoint and crossed_groups:
        issues.append(f"source_group_leakage:{len(crossed_groups)}")
    if crossed_hashes:
        issues.append(f"exact_content_leakage:{len(crossed_hashes)}")
    if not samples:
        issues.append("empty_manifest")
    for split in ("train", "valid", "test"):
        if split_counts[split] == 0:
            issues.append(f"empty_split:{split}")
    return DatasetAuditReport(
        valid=not issues, sample_count=len(samples), split_counts=dict(split_counts),
        issues=issues, cross_split_source_groups=crossed_groups,
        cross_split_content_hashes=crossed_hashes,
    )
