"""准备独立的文字中性辅助任务；不将七类情绪硬映射为三类极性。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import unicodedata
from typing import Any

import numpy as np
from round6_emotiontalk_data import digest, save

EMOTIONS = {"happy", "surprised", "sad", "disgusted", "angry", "fearful", "neutral"}


def clean_text(text: str) -> str:
    """移除重叠、笑声等转录标签，避免把情绪提示当成词句。"""
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", "", text)).strip()


def key_for(text: str) -> str:
    """统一字符形式，仅在重复与交叉重合审查使用。"""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", unicodedata.normalize("NFKC", clean_text(text)).casefold())


def neutral_label(annotation: dict[str, Any]) -> tuple[int, float]:
    """只使用有明确类别及五人投票的标签；缺失绝不补成中性。"""
    emotion = annotation.get("emotion_result")
    votes = annotation.get("data", {})
    if emotion not in EMOTIONS or len(votes) != 5:
        raise ValueError("缺失或非法的情绪标注")
    values = [row.get("emotion") for row in votes.values()]
    if any(value not in EMOTIONS for value in values):
        raise ValueError("非法投票类别")
    label = int(emotion == "neutral")
    agreement = sum(int(value == "neutral") == label for value in values) / len(values)
    return label, agreement


def prepare(data: Path, bundle: Path, previous: Path, output: Path) -> dict[str, Any]:
    """按官方组留出、隔离所有既有来源文字，再训练专用二分类辅助任务。"""
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("准备结果已存在，拒绝覆盖")
    annotations: dict[str, dict[str, Any]] = {}
    multi: dict[str, dict[str, Any]] = {}
    for name, target in [("text_annotations.jsonl", annotations), ("multimodal_annotations.jsonl", multi)]:
        for line in (data / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in target:
                raise ValueError("重复样本 ID")
            target[row["id"]] = row["annotation"]
    if set(annotations) != set(multi):
        raise ValueError("跨模态样本集合不符")
    old: set[str] = set()
    for split in ["train", "selection", "calibration", "test"]:
        with np.load(bundle / (split + ".npz"), allow_pickle=False) as values:
            old.update(key_for(str(text)) for text in values["raw_text"])
    old_files = [previous / "external/train.jsonl", previous / "external/external_valid.jsonl",
                 previous / "fresh_holdout/holdout.jsonl", previous / "router_confirmation/holdout.jsonl"]
    for path in old_files:
        if not path.exists():
            raise ValueError(f"既有数据清单缺失: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            old.add(key_for(json.loads(line)["text"]))
    grams = {text[i:i + 8] for text in old for i in range(max(0, len(text) - 7))}
    rows: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    continuous: Counter[str] = Counter()
    joint: Counter[str] = Counter()
    review: list[dict[str, Any]] = []
    for sample_id, value in sorted(annotations.items()):
        group, dialogue = sample_id.split("_")[:2]
        split = "valid" if group in {"G00001", "G00012"} else "test" if group in {"G00003", "G00015"} else "train"
        label, agreement = neutral_label(value)
        raw = value["content"]
        text = clean_text(raw)
        key = key_for(text)
        score = float(multi[sample_id]["Continuous label_result"])
        votes = [float(row["Continuous_label"]) for row in multi[sample_id]["data"].values()]
        if not np.isfinite(score) or not all(np.isfinite(votes)) or not -2 <= score <= 2:
            raise ValueError("非法连续标注")
        if abs(sum(votes) / len(votes) - score) > 1e-5:
            raise ValueError("连续分数与原始投票不符")
        continuous[str(score)] += 1
        joint[f"{multi[sample_id]['emotion_result']}|{'negative' if score < 0 else 'positive' if score > 0 else 'zero'}"] += 1
        if not key or not re.search(r"[\u4e00-\u9fff]", text):
            removed[split + "_empty_or_no_chinese"] += 1
            continue
        if key in old or any(key[i:i + 8] in grams for i in range(max(0, len(key) - 7))):
            removed[split + "_previous_source_overlap"] += 1
            continue
        rows.append({"id": "emotiontalk/" + sample_id, "group": group, "dialogue": group + "_" + dialogue,
                     "speaker": str(value["speaker_id"]), "split": split, "text": text, "raw_text": raw,
                     "label": label, "source_emotion": value["emotion_result"], "binary_vote_agreement": agreement,
                     "label_kind": "text_emotion_neutral_vs_other_auxiliary_not_sentiment_polarity", "synthetic": False})
        if split == "train" and multi[sample_id]["emotion_result"] == "happy" and score <= -.6:
            review.append({"id": sample_id, "text": text, "text_emotion": value["emotion_result"],
                           "multimodal_emotion": "happy", "continuous_score": score, "votes": votes,
                           "status": "human_review_pending; different annotation axes, not automatically declared erroneous"})
    labels: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        labels[key_for(row["text"])].add(row["label"])
    conflicts = {key for key, values in labels.items() if len(values) > 1}
    seen: set[str] = set()
    splits: dict[str, list[dict[str, Any]]] = {}
    # 测试优先保留；矛盾文本全部排除，绝不自动改标。
    for split in ["test", "valid", "train"]:
        kept: list[dict[str, Any]] = []
        for row in rows:
            if row["split"] != split:
                continue
            key = key_for(row["text"])
            if key in conflicts:
                removed[split + "_conflicting_duplicate_text"] += 1
            elif key in seen:
                removed[split + "_duplicate_text"] += 1
            else:
                kept.append(row)
                seen.add(key)
        splits[split] = kept
        (output / (split + ".jsonl")).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    random.Random(42).shuffle(review)
    save(output / "review_queue.json", review[:40])
    report = {"seed": 42, "revision": "adbc17fc944e8cf2873643906160c6ca0259ab61", "raw_samples": len(annotations),
              "task": "binary text emotional neutrality auxiliary learning; no seven-to-three polarity mapping",
              "source_split": {"valid": ["G00001", "G00012"], "test": ["G00003", "G00015"], "train": "other official groups"},
              "counts": {name: {"n": len(values), "labels": dict(Counter(row["label"] for row in values)),
                                "groups": sorted({row["group"] for row in values})} for name, values in splits.items()},
              "removed": dict(removed), "conflicting_text_keys_removed": len(conflicts),
              "continuous_score_counts": dict(continuous), "multimodal_emotion_vs_score_sign": dict(joint),
              "continuous_labels_used_for_training": False, "annotation_relabeling": False,
              "human_review_completed": False, "human_review_queue_size": min(40, len(review)),
              "dedup": "NFKC exact and 8-character overlap with all prior CH-SIMS/ECISA files; cross-split exact dedup",
              "limitations": ["source group split does not imply actor isolation", "text emotional neutrality differs from sentiment polarity neutrality",
                              "no clinical outcome labels; acted dialogues", "no media input in this initial auxiliary experiment"],
              "input_hashes": {name: digest(data / name) for name in ["text_annotations.jsonl", "multimodal_annotations.jsonl"]},
              "split_hashes": {name: digest(output / (name + ".jsonl")) for name in splits}}
    save(output / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["data", "bundle", "previous", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.bundle, args.previous, args.output)
