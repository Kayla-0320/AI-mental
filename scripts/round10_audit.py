"""独立复算、保存模型重放、来源隔离与无标签推理的最终审计。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from experiments import round10_robust_final as final
from experiments import round10_evidence as evidence
from experiments.round10_speech_evidence import data_for
from experiments.round10_fusion import OUTPUT, ROOT, save, load_npz


def digest(path: Path) -> str:
    """分块计算工件指纹。"""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    """审计失败就停止，不生成通过记录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("robust_final", "large_final", "speech_final", "holdout_final", "direct_final", "refined_final"), default="refined_final")
    args = parser.parse_args()
    loader, predictor = data_for, final.predict
    if args.stage in ("large_final", "holdout_final", "direct_final", "refined_final"):
        from experiments import round10_large_final as large_final
        from experiments import round10_large_speech as large
        large_final.configure()
        loader = large.data_for
        if args.stage == "holdout_final":
            from experiments.round10_holdout import predict as holdout_predict
            predictor = holdout_predict
        elif args.stage == "refined_final":
            from experiments.round10_refined import predict as refined_predict
            predictor = refined_predict
    elif args.stage == "speech_final":
        from experiments.round10_speech_final import predict as speech_predict
        predictor = speech_predict
    output = OUTPUT / args.stage
    result = json.loads((output / "results.json").read_text("utf-8"))
    predictions = load_npz(output / "historical_predictions.npz")
    data = loader(("test",))
    if not np.array_equal(predictions["sample_ids"], data["sample_ids"]):
        raise ValueError("预测ID不一致")
    original = load_npz(ROOT.parent / "output" / "round9_review_20261116" / "replay_predictions.npz")
    assert np.array_equal(predictions["sample_ids"], original["sample_ids"])
    assert np.array_equal(predictions["labels"], original["labels"])
    assert np.array_equal(predictions["labels"], data["labels"])
    assert predictions["probabilities"].shape == (5, 821, 3)
    assert np.isfinite(predictions["probabilities"]).all()
    assert np.allclose(predictions["probabilities"].sum(-1), 1.)
    runs = [float(f1_score(predictions["labels"], p.argmax(1), labels=[0, 1, 2], average="macro"))
            for p in predictions["probabilities"]]
    mean = float(np.mean(runs))
    ensemble = float(f1_score(predictions["labels"], predictions["ensemble"].argmax(1), labels=[0, 1, 2], average="macro"))
    assert abs(mean-result["mean_macro_f1"]) < 1e-12
    assert abs(ensemble-result["ensemble"]["macro_f1"]) < 1e-12
    with (output / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    assert len(set(data["sample_ids"])) == 821
    assert not set(model["training_sample_ids"]) & set(data["sample_ids"])
    assert not set(model["training_groups"]) & set(data["source_groups"])
    if args.stage == "holdout_final":
        assert len(model["training_sample_ids"]) == 2876
        assert len(model["adaptation_sample_ids"]) == 703
        assert not set(model["training_groups"]) & set(model["adaptation_groups"])
        assert not set(model["adaptation_groups"]) & set(data["source_groups"])
        assert not set(model["adaptation_sample_ids"]) & set(data["sample_ids"])
    else:
        assert len(model["training_sample_ids"]) == 3579
    unlabeled = {name: value for name, value in data.items() if name not in ("labels", "source_groups", "raw_text")}

    def forbidden(_: Any) -> Any:
        raise AssertionError("推理不得读取官方监督标签")

    original = evidence.supervision
    evidence.supervision = forbidden
    try:
        replay, route = predictor(model, unlabeled)
        assert np.allclose(replay, predictions["probabilities"], atol=1e-12, rtol=0.)
        assert np.array_equal(route, predictions["costly_needed"])
        unlabeled["labels"] = np.full(821, -999)
        unlabeled["source_groups"] = np.array(["unobserved"]*821)
        changed, _ = predictor(model, unlabeled)
        assert np.array_equal(changed, replay)
    finally:
        evidence.supervision = original
    development = loader(("train", "selection"))
    group_for_id = dict(zip(development["sample_ids"], development["source_groups"]))
    fold_checks = []
    checked_stages = ("large_speech", "large_speech5") if args.stage in ("large_final", "holdout_final", "direct_final", "refined_final") else ("speech_evidence", "speech_evidence5", "speech_bag")
    for stage in checked_stages:
        scored = []
        for fold in range(4):
            values = load_npz(OUTPUT / stage / f"fold{fold}_evidence.npz")
            training = {group_for_id[i] for i in values["train_sample_ids"]}
            validation = {group_for_id[i] for i in values["validation_sample_ids"]}
            assert not training & validation
            assert np.isfinite(values["inner_evidence"]).all()
            scored.extend(values["validation_sample_ids"].tolist())
        assert len(scored) == len(set(scored)) == 2876
        fold_checks.append({"stage": stage, "unique_oof_predictions": len(scored), "source_disjoint": True})
    manifest = json.loads((output / "model_manifest.json").read_text("utf-8"))
    assert digest(output / "model.pkl") == manifest["model_sha256"]
    assert digest(output / "selection_locked.json") == manifest["selection_sha256"]
    selection_check = None
    if args.stage == "refined_final":
        source = load_npz(OUTPUT / "large_final" / "selected_oof.npz")
        lock = json.loads((output / "selection_locked.json").read_text("utf-8"))
        registration = json.loads((output / "preregister.json").read_text("utf-8"))
        assert np.array_equal(source["sample_ids"], development["sample_ids"])
        _, inverse, counts = np.unique(development["source_groups"], return_inverse=True, return_counts=True)
        weights = 1/counts[inverse]
        grid_rows = []
        for negative in registration["absolute_bias_grid"]:
            for neutral in registration["absolute_bias_grid"]:
                bias = [negative, neutral, 0.]
                predicted = (np.log(np.maximum(source["probabilities"], 1e-12))+np.asarray(bias)).argmax(1)
                f1 = float(f1_score(development["labels"], predicted, labels=[0, 1, 2], average="macro"))
                weighted = float(f1_score(development["labels"], predicted, labels=[0, 1, 2], average="macro", sample_weight=weights))
                grid_rows.append((.5*(f1+weighted), f1, -sum(abs(v) for v in bias), bias))
        chosen = max(grid_rows, key=lambda row: row[:3])
        assert chosen[3] == lock["winner"]["bias"]
        assert abs(chosen[0]-lock["winner"]["selection_score"]) < 1e-12
        old_probabilities, _ = final.predict(model["model"], data)
        earlier = load_npz(OUTPUT / "large_final" / "historical_predictions.npz")
        assert np.allclose(old_probabilities, earlier["probabilities"], atol=1e-12, rtol=0.)
        assert digest(OUTPUT / "large_final" / "model.pkl") == manifest["base_model_sha256"]
        assert (output / "selection_locked.json").stat().st_mtime_ns <= (output / "model.pkl").stat().st_mtime_ns <= (output / "results.json").stat().st_mtime_ns
        selection_check = {"independently_recomputed_grid_points": len(grid_rows), "selected_bias": chosen[3],
                           "source_score": chosen[0], "base_predictions_unchanged": True, "lock_before_fit_before_evaluation": True}
    baseline = load_npz(OUTPUT / "final" / "historical_predictions.npz")
    baseline_runs = [float(f1_score(baseline["labels"], p.argmax(1), labels=[0, 1, 2], average="macro")) for p in baseline["baseline"]]
    record = {"stage": args.stage, "independent_library": "sklearn.metrics", "seed_macro_f1": runs, "mean_macro_f1": mean,
        "ensemble_macro_f1": ensemble, "ensemble_accuracy": float(accuracy_score(predictions["labels"], predictions["ensemble"].argmax(1))),
        "baseline_actual_five_seed_mean": float(np.mean(baseline_runs)), "matches_recorded": True,
        "serialized_model_prediction_max_error": float(np.max(abs(replay-predictions["probabilities"]))),
        "no_labels_or_source_metadata_required": True, "supervision_reader_forbidden_during_replay": True,
        "coverage": 1., "n": 821, "train_n": len(model["training_sample_ids"]), "oof_checks": fold_checks,
        "model_sha256_matches": True, "selection_sha256_matches": True,
        "historical_labels_and_ids_match_original_round9": True, "development_selection_check": selection_check}
    save(OUTPUT / "final_audit.json", record)
    save(output / "audit.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
