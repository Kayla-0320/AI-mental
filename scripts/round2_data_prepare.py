"""将摘要匹配的作者特征转换为可审计、来源分组的安全 NPZ 训练输入。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))
from data_quality import audit_manifest
from shared import DatasetSampleRecord

EXPECTED_SHA256 = "c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f"
LABEL_DOMAIN = "ch_sims_v1_sentiment_polarity_3"


def file_hash(path: Path) -> str:
    """分块核对固定作者文件，不接受未知 pickle。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_features(values: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """仅在有效时间范围均池化；失效模态给显式掩码而不冒充中性观测。"""
    if values.ndim != 3 or len(lengths) != len(values):
        raise ValueError("时序特征/长度结构错误")
    pooled = np.zeros((len(values), values.shape[-1]), dtype=np.float32)
    mask = np.zeros(len(values), dtype=bool)
    quality = np.zeros(len(values), dtype=np.float32)
    counts = {"zero_length": 0, "invalid_length": 0, "non_finite": 0, "usable": 0}
    for i, length in enumerate(lengths):
        if not np.isfinite(length) or length != int(length) or not 0 <= length <= values.shape[1]:
            counts["invalid_length"] += 1
            continue
        length = int(length)
        if length == 0:
            counts["zero_length"] += 1
            continue
        window = values[i, :length]
        if not np.isfinite(window).all():
            counts["non_finite"] += 1
            continue
        vector = window.mean(axis=0, dtype=np.float64)
        if not np.isfinite(vector).all():
            counts["non_finite"] += 1
            continue
        pooled[i] = vector
        mask[i] = True
        quality[i] = 1.0
        counts["usable"] += 1
    return pooled, mask, quality, counts


def prepare(source: Path, output: Path) -> dict[str, object]:
    """保留原始分割诊断，同时输出严格按原视频组独立的主实验划分。"""
    source_sha = file_hash(source)
    if source_sha != EXPECTED_SHA256:
        raise ValueError("拒绝加载非作者摘要匹配 pickle；它可能执行不可信代码")
    with source.open("rb") as stream:
        data = pickle.load(stream)  # 仅接受固定作者 SHA 匹配的公开研究文件。
    output.mkdir(parents=True, exist_ok=True)
    original_split = np.concatenate([np.repeat(s, len(data[s]["id"])) for s in ("train", "valid", "test")])
    arrays: dict[str, np.ndarray] = {"original_split": original_split}
    counts: dict[str, object] = {}
    pooled_modalities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    for name, key in (("text", "text"), ("audio", "audio"), ("visual", "vision")):
        parts: list[np.ndarray] = []
        part_masks: list[np.ndarray] = []
        part_quality: list[np.ndarray] = []
        counts[name] = {}
        for split in ("train", "valid", "test"):
            raw = data[split]
            if name == "text":
                length = np.asarray(raw["text_bert"])[:, 1, :].sum(axis=1)
            else:
                length = np.asarray(raw[f"{key}_lengths"])
            vectors, usable, quality, stat = pool_features(np.asarray(raw[key]), length)
            parts.append(vectors)
            part_masks.append(usable)
            part_quality.append(quality)
            counts[name][split] = stat
        arrays[name] = np.concatenate(parts)
        pooled_modalities.append(arrays[name])
        masks.append(np.concatenate(part_masks))
        qualities.append(np.concatenate(part_quality))
    arrays["modality_mask"] = np.stack(masks, axis=1)
    arrays["quality"] = np.stack(qualities, axis=1)
    arrays["sample_ids"] = np.concatenate([np.asarray(data[s]["id"]).astype(str) for s in ("train", "valid", "test")])
    arrays["raw_text"] = np.concatenate([np.asarray(data[s]["raw_text"]).astype(str) for s in ("train", "valid", "test")])
    arrays["source_groups"] = np.asarray(["ch-sims:" + key.split("$_$")[0] for key in arrays["sample_ids"]])
    raw_labels = np.concatenate([np.asarray(data[s]["classification_labels"]) for s in ("train", "valid", "test")])
    if not np.isfinite(raw_labels).all() or not np.isin(raw_labels, [0, 1, 2]).all():
        raise ValueError("原始标签不在作者三分类标签域中")
    arrays["labels"] = raw_labels.astype(np.int64)
    for modality, suffix in (("text", "T"), ("audio", "A"), ("visual", "V")):
        arrays[f"labels_{modality}"] = np.concatenate([np.asarray(data[s][f"classification_labels_{suffix}"]).astype(np.int64) for s in ("train", "valid", "test")])
        arrays[f"valence_{modality}"] = np.concatenate([np.asarray(data[s][f"regression_labels_{suffix}"]).astype(np.float32) for s in ("train", "valid", "test")])
    arrays["valence"] = np.concatenate([np.asarray(data[s]["regression_labels"]).astype(np.float32) for s in ("train", "valid", "test")])
    all_index = np.arange(len(arrays["labels"]))
    train_valid, test_index = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(all_index, groups=arrays["source_groups"]))
    train_part, valid_part = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(train_valid, groups=arrays["source_groups"][train_valid]))
    assigned = np.empty(len(all_index), dtype="<U5")
    assigned[train_valid[train_part]] = "train"
    assigned[train_valid[valid_part]] = "valid"
    assigned[test_index] = "test"
    # 同一次同步观测的完整多模态表示摘要用于重复检测；不把同一句短文本当重复音视频。
    fingerprints = [hashlib.sha256(b"".join(np.ascontiguousarray(v[i]).tobytes() for v in pooled_modalities)).hexdigest() for i in all_index]
    records = [DatasetSampleRecord(dataset_id="CH-SIMS-v1", sample_id=str(arrays["sample_ids"][i]), source_group=str(arrays["source_groups"][i]), split=str(original_split[i]), label_domain=LABEL_DOMAIN, label=int(raw_labels[i]), content_hashes={"paired_pooled_features": fingerprints[i]}) for i in all_index]
    original_audit = audit_manifest(records, class_count=3, expected_label_domain=LABEL_DOMAIN)
    strict_records = [DatasetSampleRecord(dataset_id=r.dataset_id, sample_id=r.sample_id, source_group=r.source_group, split=str(assigned[i]), label_domain=r.label_domain, label=r.label, content_hashes=r.content_hashes) for i, r in enumerate(records)]
    strict_audit = audit_manifest(strict_records, class_count=3, expected_label_domain=LABEL_DOMAIN)
    if not strict_audit.valid:
        raise ValueError(f"严格分组审核未通过：{strict_audit.issues}")
    for split in ("train", "valid", "test"):
        index = assigned == split
        np.savez_compressed(output / f"{split}.npz", **{name: value[index] for name, value in arrays.items()})
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in strict_records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    metadata: dict[str, object] = {
        "schema_version": "paired_features_v1", "dataset": "CH-SIMS-v1", "task": LABEL_DOMAIN,
        "class_names": ["negative", "neutral", "positive"], "modality_order": ["text", "audio", "visual"],
        "feature_dims": {name: arrays[name].shape[1] for name in ("text", "audio", "visual")},
        "feature_description": "author-provided frozen features; masked temporal mean over true lengths; not VLM embeddings",
        "encoder_fingerprints": {name: f"source_pkl_sha256:{source_sha};field:{field};preprocessing:valid_length_mean_v1" for name, field in (("text", "text"), ("audio", "audio"), ("visual", "vision"))},
        "source_sha256": source_sha, "seed": 42, "split_protocol": "source video group holdout:20% test;25% remaining valid; not actor/film disjoint",
        "original_audit": asdict(original_audit), "strict_audit": asdict(strict_audit), "quality_counts": counts,
        "sample_count": len(all_index),
        "split_group_counts": {split: len(set(arrays["source_groups"][assigned == split])) for split in ("train", "valid", "test")},
        "split_class_counts": {split: np.bincount(arrays["labels"][assigned == split], minlength=3).tolist() for split in ("train", "valid", "test")},
        "coverage": {name: float(arrays["modality_mask"][:, i].mean()) for i, name in enumerate(("text", "audio", "visual"))},
        "all_modalities_missing": int((~arrays["modality_mask"].any(axis=1)).sum()),
        "unimodal_label_disagreement": {name: float((arrays[f"labels_{name}"] != arrays["labels"]).mean()) for name in ("text", "audio", "visual")},
        "disagreement_policy": "preserve independent modality labels; mismatch alone never causes relabeling, dropping or trust inference",
        "normalization": "none here; fit feature scaler on train only in downstream experiment",
        "alignment_limit": "author sample IDs and lengths only; original media timing and active speaker not independently rechecked",
        "split_checksums": {split: file_hash(output / f"{split}.npz") for split in ("train", "valid", "test")},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output_dir)


if __name__ == "__main__":
    main()
