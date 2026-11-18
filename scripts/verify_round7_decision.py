"""独立复算第七轮划分、嵌套选择和预测；不训练、不修改锁定工件。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))
from experiments.round7_decision_oof import raw_predict, take


def load(path: Path) -> dict[str, np.ndarray]:
    """读取无对象数组。"""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def read(path: Path) -> Any:
    """读取实验JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """流式摘要验证。"""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def apply(p: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    """用另一实现重放规则。"""
    if rule["kind"] == "argmax":
        return np.argmax(p, axis=1)
    if rule["kind"] == "all_negative":
        return np.zeros(len(p), dtype=int)
    if rule["kind"] == "bias":
        z = np.log(p.clip(1e-15))
        z[:, 0] += rule["negative_bias"]
        z[:, 1] += rule["neutral_bias"]
        return z.argmax(1)
    if rule["kind"] == "ordinal":
        x = p[:, 2] - p[:, 0]
        result = np.ones(len(p), dtype=int)
        result[x < rule["lo"]] = 0
        result[x > rule["hi"]] = 2
        return result
    result = p.argmax(1)
    for i, row in enumerate(p):
        if row[0] >= rule["negative_guard"]:
            result[i] = 0
        elif row[1] >= rule["neutral_threshold"]:
            result[i] = 1
    return result


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """用sklearn独立复算，而非调用训练脚本指标函数。"""
    return {"macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)), "per_class_f1": f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "per_class_recall": recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def verify(root: Path, previous: Path) -> dict[str, Any]:
    """检查真实工件的来源隔离、选型重放和不可变性。"""
    checks: list[str] = []

    def check(value: bool, name: str) -> None:
        if not value:
            raise AssertionError(name)
        checks.append(name)

    report = read(root / "results.json")
    protocol = read(root / "preregister.json")
    lock = read(root / "selection_locked.json")
    check(sha(root / "source/round7_decision_oof.py") == protocol["code_sha256"], "executed_source_hash")
    check(sha(root / "model.pkl") == lock["model_sha256"], "final_model_hash")
    check(protocol["no_703_selection"] and lock["no_703_used"], "703_not_used_for_new_selection_recorded")
    check(protocol["gpu_used"] is False and protocol["seed"] == 42, "cpu_only_seed42_protocol")
    for path, identity in protocol["old_critical_hashes"].items():
        check(sha(Path(path)) == identity, "old_artifact_unchanged_" + Path(path).parent.name + "_" + Path(path).name)
    parts = [load(previous / "face_features" / (name + ".npz")) for name in ["train", "selection"]]
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    main_oof = load(root / "oof_B.npz")
    check(len(data["labels"]) == 2876, "training2876_identity")
    check(np.array_equal(data["sample_ids"], main_oof["sample_ids"]) and np.array_equal(data["labels"], main_oof["labels"]), "oof_training_order_labels")
    check(set(main_oof["fold"]) == {0, 1, 2, 3}, "four_outer_folds")
    for group in np.unique(data["source_groups"]):
        check(len(np.unique(main_oof["fold"][data["source_groups"] == group])) == 1, "single_outer_fold_for_" + str(group))
    for fold in range(4):
        fold_root = root / f"folds/{fold}"
        record = read(fold_root / "selection_locked.json")
        train_ids, valid_ids = set(record["train_ids"]), set(record["validation_ids"])
        check(not train_ids & valid_ids and not set(record["train_groups"]) & set(record["validation_groups"]), f"outer_{fold}_group_and_id_isolation")
        ix = np.flatnonzero(main_oof["fold"] == fold)
        check(set(data["sample_ids"][ix]) == valid_ids, f"outer_{fold}_exact_validation_ids")
        inner = load(fold_root / "inner_oof.npz")
        check(set(inner["sample_ids"]) == train_ids and not set(inner["sample_ids"]) & valid_ids, f"inner_{fold}_never_saw_outer_validation")
        check(all(len(np.unique(inner["fold"][inner["source_groups"] == group])) == 1 for group in np.unique(inner["source_groups"])), f"inner_{fold}_source_group_isolation")
        with (fold_root / "model.pkl").open("rb") as handle:
            model = pickle.load(handle)
        replay = raw_predict(model, take(data, ix))
        check(np.allclose(replay["scores"], main_oof["scores"][ix], atol=1e-10), f"outer_{fold}_saved_model_prediction_replay")
        for track in ["A", "B"]:
            oof = load(root / f"oof_{track}.npz")
            policy = record["policies"][track]
            scope = replay["face"] if track == "A" else replay["valid"]
            scale = np.where(replay["face"], policy["temperatures"]["face"], policy["temperatures"]["fallback"])
            p = softmax(replay["scores"][scope] / scale[scope, None], axis=1)
            check(np.allclose(p, oof["nested_probabilities"][ix[scope]], atol=1e-12), f"outer_{fold}_{track}_nested_calibration_replay")
            for floor in protocol["negative_recall_constraints"]:
                pred = apply(p, policy["rules"][str(floor)]["rule"])
                check(np.array_equal(pred, oof[f"nested_pred_{floor}"][ix[scope]]), f"outer_{fold}_{track}_{floor}_nested_rule_replay")
    for track in ["A", "B"]:
        oof = load(root / f"oof_{track}.npz")
        scope = oof["face"] if track == "A" else oof["valid"]
        policy = lock["policies"][track]
        scale = np.where(oof["face"], policy["temperatures"]["face"], policy["temperatures"]["fallback"])
        p = softmax(oof["scores"][scope] / scale[scope, None], axis=1)
        y = oof["labels"][scope]
        options = [(rule, scores(y, apply(p, rule))) for rule in protocol["rule_grid"]]
        for floor in protocol["negative_recall_constraints"]:
            feasible = [row for row in options if row[1]["per_class_recall"][0] >= floor]
            best = max(feasible, key=lambda row: (row[1]["macro_f1"], row[1]["accuracy"]))
            check(best[0] == policy["cost_curve"][str(floor)]["rule"], f"{track}_{floor}_oof_constrained_selection_independently_recomputed")
            nested_score = scores(y, oof[f"nested_pred_{floor}"][scope])
            check(abs(nested_score["macro_f1"] - report["nested_oof"][track]["nested_cost_curve"][str(floor)]["macro_f1"]) < 1e-12, f"{track}_{floor}_nested_metric_recomputed")
    with (root / "model.pkl").open("rb") as handle:
        final = pickle.load(handle)
    with (previous / "face_search/selected.pkl").open("rb") as handle:
        old_primary = pickle.load(handle)
    check(np.allclose(final["primary"].coef_, old_primary["model"].coef_, atol=1e-5) and np.allclose(final["primary"].intercept_, old_primary["model"].intercept_, atol=1e-5), "A_primary_parameters_unchanged")
    original = load(previous / "availability_router/historical_predictions.npz")
    metrics = {}
    for track in ["A", "B"]:
        pred = load(root / f"historical_{track}.npz")
        check(len(pred["labels"]) == 821 and np.array_equal(pred["sample_ids"], original["sample_ids"]) and np.array_equal(pred["labels"], original["labels"]), f"{track}_same821_ids_labels")
        scope = pred["face"] if track == "A" else np.ones(821, bool)
        replay = apply(pred["probabilities"][scope], lock["policies"][track]["main_rule"])
        check(np.array_equal(replay, pred["predictions"][scope]), f"{track}_historical_locked_rule_replay")
        if track == "A":
            check(np.array_equal(pred["predictions"][~scope], original["scores"][~scope].argmax(1)), "A_legacy_fallback_unchanged")
            check(np.isnan(pred["probabilities"][~scope]).all(), "A_does_not_claim_fallback_calibration")
            expected = original["scores"][scope].copy()
            expected[:, 1] -= old_primary["bias"]
            check(np.allclose(pred["scores"][scope], expected, atol=1e-5), "A_primary_only_removed_old_bias_before_new_policy")
        metrics[track] = scores(pred["labels"], pred["predictions"])
        check(abs(metrics[track]["macro_f1"] - report["historical"][track]["historical"]["macro_f1"]) < 1e-12, f"{track}_historical_metric_recomputed")
    external = load(root / "external.npz")
    old_external = load(previous / "router_confirmation/predictions.npz")
    check(np.array_equal(external["sample_ids"], old_external["sample_ids"]) and np.array_equal(external["labels"], old_external["labels"]) and len(external["labels"]) == 3363, "same_ecisa3363_ids_labels")
    check(np.array_equal(apply(external["probabilities"], lock["policies"]["B"]["main_rule"]), external["predictions"]), "B_external_locked_rule_replay")
    check(abs(scores(external["labels"], external["predictions"])["macro_f1"] - report["external_ecisa"]["B"]["macro_f1"]) < 1e-12, "B_external_metric_recomputed")
    check(report["deployed"] is False, "not_deployed")
    result = {"passed": len(checks), "checks": checks, "historical": metrics, "raw_rule_candidates": len(protocol["rule_grid"]),
              "nested_primary_only_n": int(main_oof["face"].sum()), "nested_full_pipeline_n": len(main_oof["labels"]), "no_human_adjudication": True}
    (root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": len(checks), "historical": metrics}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.previous)
