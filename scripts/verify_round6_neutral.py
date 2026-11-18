"""逐样本复算 EmotionTalk 接入与三组实验，验证冻结留出和原始标签。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from round6_emotiontalk_data import digest, save
from round6_prepare_neutral import key_for, neutral_label


def verify(root: Path, previous: Path, bundle: Path) -> dict[str, Any]:
    """直接读取保存预测与原始标注，不重新拟合模型。"""
    checks: list[str] = []

    def check(condition: bool, name: str) -> None:
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    acquisition = json.loads((root / "data/acquisition.json").read_text(encoding="utf-8"))
    for name, expected in acquisition["annotation_sha256"].items():
        check(digest(root / "data" / name) == expected, "annotation_hash_" + name)
    annotations = {row["id"]: row["annotation"] for row in map(json.loads, (root / "data/text_annotations.jsonl").read_text(encoding="utf-8").splitlines())}
    check(len(annotations) == 19250, "all_source_utterances_acquired")
    sets: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    transcription_residuals: list[dict[str, str]] = []
    manifest = json.loads((root / "neutral_data/manifest.json").read_text(encoding="utf-8"))
    for name in ["train", "valid", "test"]:
        source = root / "neutral_data" / (name + ".jsonl")
        check(digest(source) == manifest["split_hashes"][name], "split_hash_" + name)
        rows = list(map(json.loads, source.read_text(encoding="utf-8").splitlines()))
        sets[name] = rows
        keys = {key_for(row["text"]) for row in rows}
        check(len(keys) == len(rows) and not keys.intersection(seen), "source_exact_text_dedup_" + name)
        seen |= keys
        check(all(row["label"] == neutral_label(annotations[row["id"].split("/", 1)[1]])[0] for row in rows), "labels_match_source_neutrality_" + name)
        check(all(not re.search(r"\[[^\]]*\]", row["text"]) for row in rows), "complete_transcription_tags_removed_" + name)
        for row in rows:
            if "[" in row["text"] or "]" in row["text"]:
                transcription_residuals.append({"split": name, "id": row["id"], "raw_text": row["raw_text"], "text": row["text"],
                                                "status": "malformed overlap tag residual retained in frozen experiment; not human-corrected"})
    group_sets = [{row["group"] for row in sets[name]} for name in ["train", "valid", "test"]]
    check(all(not group_sets[i].intersection(group_sets[j]) for i in range(3) for j in range(i)), "source_groups_disjoint")
    check(not manifest["continuous_labels_used_for_training"] and not manifest["annotation_relabeling"], "no_unreviewed_polarity_remapping")
    check(manifest["human_review_completed"] is False, "manual_review_not_claimed")
    results = json.loads((root / "neutral_transfer/results.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "neutral_transfer/selection_locked.json").read_text(encoding="utf-8"))
    chosen = max(lock["development"], key=lambda name: (lock["development"][name]["development"]["macro_f1"], lock["development"][name]["development"]["accuracy"]))
    check(results["winner"] == lock["winner"] == chosen, "selection_matches_development_rule")
    check(lock["test_not_yet_loaded"] is True, "selection_lock_before_test_load_recorded")
    with np.load(bundle / "test.npz", allow_pickle=False) as original:
        ids, labels = original["sample_ids"], original["labels"]
    check(len(ids) == 821, "historical821_unchanged")
    source_ids = np.array([row["id"] for row in sets["test"]])
    source_labels = np.array([row["label"] for row in sets["test"]])
    metrics: dict[str, Any] = {}
    for name, record in results["evaluations"].items():
        with np.load(root / "neutral_transfer" / (name + "_predictions.npz"), allow_pickle=False) as saved:
            check(np.array_equal(ids, saved["sample_ids"]) and np.array_equal(labels, saved["historical_labels"]), "same_historical_ids_labels_" + name)
            scores = saved["historical_scores"]
            check(bool(np.isfinite(scores).all()), "finite_full_coverage_" + name)
            pred = scores.argmax(1)
            score = float(f1_score(labels, pred, labels=[0, 1, 2], average="macro", zero_division=0))
            accuracy = float(accuracy_score(labels, pred))
            check(abs(score - record["historical"]["macro_f1"]) < 1e-12 and abs(accuracy - record["historical"]["accuracy"]) < 1e-12, "historical_metric_recomputed_" + name)
            check(np.array_equal(source_ids, saved["source_ids"]) and np.array_equal(source_labels, saved["source_labels"]), "same_source_test_ids_labels_" + name)
            source_scores = saved["source_scores"]
            binary_pred = np.column_stack([np.logaddexp(source_scores[:, 0], source_scores[:, 2]), source_scores[:, 1]]).argmax(1)
            binary_f1 = float(f1_score(source_labels, binary_pred, labels=[0, 1], average="macro", zero_division=0))
            check(abs(binary_f1 - record["emotiontalk_binary_neutral"]["macro_f1"]) < 1e-12, "source_binary_metric_recomputed_" + name)
            ecisa_pred = saved["ecisa_scores"].argmax(1)
            ecisa_f1 = float(f1_score(saved["ecisa_labels"], ecisa_pred, labels=[0, 1, 2], average="macro", zero_division=0))
            check(abs(ecisa_f1 - record["ecisa_3class"]["macro_f1"]) < 1e-12, "ecisa_metric_recomputed_" + name)
            metrics[name] = {"macro_f1": score, "accuracy": accuracy, "correct": int((pred == labels).sum()),
                             "neutral_f1": float(f1_score(labels, pred, labels=[1], average=None, zero_division=0)[0]),
                             "neutral_correct": int(((pred == 1) & (labels == 1)).sum()), "neutral_support": int((labels == 1).sum()),
                             "source_binary_macro_f1": binary_f1, "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1, 2]).tolist()}
    check(results["deployed"] is False, "research_only_not_deployed")
    multitask: dict[str, Any] = {}
    joint_path = root / "multitask/results.json"
    if joint_path.exists():
        joint = json.loads(joint_path.read_text(encoding="utf-8"))
        joint_lock = json.loads((root / "multitask/selection_locked.json").read_text(encoding="utf-8"))
        joint_choice = max(joint_lock["development"], key=lambda name: (joint_lock["development"][name]["development"]["macro_f1"], joint_lock["development"][name]["development"]["accuracy"]))
        check(joint["winner"] == joint_lock["winner"] == joint_choice, "joint_selection_matches_development")
        for name, record in joint["evaluations"].items():
            with np.load(root / "multitask" / (name + "_predictions.npz"), allow_pickle=False) as saved:
                check(np.array_equal(ids, saved["sample_ids"]) and np.array_equal(labels, saved["labels"]), "joint_same_historical_ids_labels_" + name)
                check(bool(np.isfinite(saved["scores"]).all()), "joint_finite_full_coverage_" + name)
                pred = saved["scores"].argmax(1)
                score = float(f1_score(labels, pred, labels=[0, 1, 2], average="macro", zero_division=0))
                check(abs(score - record["historical"]["macro_f1"]) < 1e-12, "joint_historical_metric_recomputed_" + name)
                check(joint["development"][name]["training"]["zero_epoch_allowed"] is True, "joint_zero_epoch_control_" + name)
                multitask[name] = {"macro_f1": score, "accuracy": float(accuracy_score(labels, pred)),
                                   "neutral_f1": float(f1_score(labels, pred, labels=[1], average=None, zero_division=0)[0]),
                                   "neutral_correct": int(((pred == 1) & (labels == 1)).sum())}
        check(joint["deployed"] is False, "joint_research_only_not_deployed")
        multitask["winner"] = joint_choice
    save(root / "transcription_residuals.json", transcription_residuals)
    report = {"passed": len(checks), "checks": checks, "winner": chosen, "metrics": metrics,
              "multitask": multitask, "data_counts": {name: len(rows) for name, rows in sets.items()},
              "malformed_transcription_residual_count": len(transcription_residuals),
              "speaker_overlap": {f"train_{name}": sorted({row["speaker"] for row in sets["train"]} & {row["speaker"] for row in sets[name]}) for name in ["valid", "test"]},
              "synthetic": False, "human_review_completed": False}
    save(root / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["root", "previous", "bundle"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.previous, args.bundle)
