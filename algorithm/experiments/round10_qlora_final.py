"""按来源OOF选择适配融合成员，五种子重训并在原821条上验收。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
from scipy.special import softmax

from experiments import round10_qlora as adaptation
from experiments import round10_large_speech as large
from experiments import round10_robust_final as legacy
from experiments.round10_domain import weighted_f1
from experiments.round10_fusion import OUTPUT, load_npz, save, load_data
from experiments.round10_speech_final import paired_interval
from experiments.round7_decision_oof import metric, probability_metrics


def select(output: Path) -> None:
    """六个新候选与已锁定主线做有限软投票，不额外搜索偏置。"""
    if (output / "selection_locked.json").exists():
        raise ValueError("拒绝覆盖适配融合选择")
    data = load_data()
    source = load_npz(OUTPUT / "qlora_prefix" / "candidates_oof.npz")
    rows = json.loads((OUTPUT / "qlora_prefix" / "search_complete.json").read_text("utf-8"))["results"]
    old = load_npz(OUTPUT / "large_final" / "selected_oof.npz")
    old_lock = json.loads((OUTPUT / "large_final" / "selection_locked.json").read_text("utf-8"))
    if not np.array_equal(source["sample_ids"], data["sample_ids"]) or not np.array_equal(old["sample_ids"], data["sample_ids"]):
        raise ValueError("来源OOF顺序不一致")
    baseline = softmax(np.log(np.maximum(old["probabilities"], 1e-12))+np.asarray(old_lock["winner"]["bias"]), axis=1)
    save(output / "preregister.json", {"new_candidates": [row["config"] for row in rows],
        "new_weights": [0., .25, .5, .75, 1.], "additional_bias_search": False,
        "selection": ".5 pooled source OOF F1+.5 source-balanced F1; lower new weight breaks ties",
        "promotion": "strict source-selection-score improvement over locked large_final; no historical promotion if weight0",
        "final_seeds": [42, 43, 44, 45, 46], "final_training_n": 3579,
        "full_training_trajectory": "3epochs same schedule as search, select global locked epoch only",
        "primary_acceptance": "mean five-seed macro-F1 >=0.70268 on unchanged821,100%coverage",
        "adaptive_historical_diagnostic": True})
    candidates, best = [], None
    for i, (row, z) in enumerate(zip(rows, source["logits"])):
        new = softmax(z/row["scale"]+np.asarray(row["bias"]), axis=1)
        for weight in (0., .25, .5, .75, 1.):
            p = (1-weight)*baseline+weight*new
            measured = metric(data["labels"], p.argmax(1))
            source_f1 = weighted_f1(data["labels"], p.argmax(1), data["source_groups"])
            candidate = {"new_index": i, "new_config": row["config"], "new_weight": weight,
                         "scale": row["scale"], "bias": row["bias"], "metrics": measured,
                         "source_balanced_f1": source_f1, "selection_score": .5*(measured["macro_f1"]+source_f1)}
            candidates.append(candidate)
            key = candidate["selection_score"], measured["macro_f1"], -weight, -row["config"]["epoch"]
            if best is None or key > best[0]:
                best = key, candidate, p
    promoted = best[1]["new_weight"] > 0 and best[1]["selection_score"] > old_lock["winner"]["selection_score"]+1e-12
    save(output / "selection_locked.json", {"winner": best[1], "promoted": promoted,
        "legacy": "large_final", "legacy_development_score": old_lock["winner"]["selection_score"], "seeds": [42, 43, 44, 45, 46]})
    save(output / "candidates.json", candidates)
    np.savez_compressed(output / "selected_oof.npz", probabilities=best[2], labels=data["labels"], sample_ids=data["sample_ids"])
    print(json.dumps({"winner": best[1], "promoted": promoted}), flush=True)


def fit(output: Path) -> None:
    """每个种子分别训练适配器和融合头，不仅改变读出器随机种子。"""
    if (output / "model_manifest.json").exists():
        raise ValueError("拒绝覆盖适配最终训练")
    lock_path = output / "selection_locked.json"
    lock = json.loads(lock_path.read_text("utf-8"))
    if not lock["promoted"]:
        raise ValueError("开发验证未支持新增分支，禁止执行历史选型")
    data = large.data_for(("train", "selection", "calibration"))
    _, tokens = adaptation.tokenizer_and_tokens(data)
    cache = output / "training_cache"
    cache.mkdir(exist_ok=True)
    frozen, prefix = adaptation.cache_frozen(data, tokens, cache)
    recipe = lock["winner"]["new_config"]
    runs = []
    for seed in lock["seeds"]:
        destination = output / f"seed{seed}"
        destination.mkdir(exist_ok=True)
        completed = destination / "fit_complete.json"
        if not completed.exists():
            adaptation.train_fold(data, tokens, frozen, np.arange(len(data["sample_ids"])), np.array([], dtype=int),
                seed, recipe["kind"] == "qlora", destination, prefix, epochs=3, seed=seed)
            save(completed, {"seed": seed, "n": len(data["sample_ids"]), "selected_epoch": recipe["epoch"],
                 "training_ids_sha256": hashlib.sha256(data["sample_ids"].tobytes()).hexdigest()})
        runs.append({"seed": seed, "directory": str(destination / f"{recipe['kind']}_fold{seed}")})
    with (output / "training_metadata.pkl").open("wb") as handle:
        pickle.dump({"training_sample_ids": data["sample_ids"], "training_groups": np.unique(data["source_groups"])}, handle)
    save(output / "model_manifest.json", {"runs": runs, "training_n": len(data["sample_ids"]),
        "groups": len(np.unique(data["source_groups"])), "selection_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "source_trajectory": "fixed3epoch, globally locked checkpoint used for all seeds",
        "source_code_sha256": hashlib.sha256(Path(adaptation.__file__).read_bytes()).hexdigest(),
        "legacy_model_sha256": hashlib.sha256((OUTPUT / "large_final" / "model.pkl").read_bytes()).hexdigest(),
        "test_labels_used_in_training": False})


def neural_predict(output: Path, data: dict[str, np.ndarray]) -> np.ndarray:
    """重载保存的适配器与头，缓存只读取原始文字及预先冻结的输入特征。"""
    lock = json.loads((output / "selection_locked.json").read_text("utf-8"))
    manifest = json.loads((output / "model_manifest.json").read_text("utf-8"))
    safe = dict(data)
    safe["labels"] = np.zeros(len(data["sample_ids"]), dtype=int)
    _, tokens = adaptation.tokenizer_and_tokens(safe)
    cache = output / "inference_cache"
    cache.mkdir(exist_ok=True)
    frozen, prefix = adaptation.cache_frozen(safe, tokens, cache)
    logits = []
    recipe = lock["winner"]["new_config"]
    for run in manifest["runs"]:
        directory = Path(run["directory"])
        saved = torch.load(directory / f"epoch{recipe['epoch']}_head.pt", weights_only=True)
        adapter_path = directory / f"epoch{recipe['epoch']}_adapter" if recipe["kind"] == "qlora" else None
        model = adaptation.AdaptedFusion(saved["numeric_dimensions"], recipe["kind"] == "qlora", adapter_path)
        loaded = model.load_state_dict(saved["state_dict"], strict=False)
        if loaded.unexpected_keys or any(not name.startswith("encoder.") for name in loaded.missing_keys):
            raise ValueError("保存头权重缺失或多余")
        with (directory / "scaler.pkl").open("rb") as handle:
            scaler = pickle.load(handle)
        numeric, _ = adaptation.represented.transform(safe, adaptation.FEATURE_CONFIG, scaler)
        values = adaptation.inference(model, numeric.astype(np.float32), tokens, frozen, np.arange(len(safe["sample_ids"])), prefix)
        if not np.isfinite(values).all():
            raise ValueError("适配推理输出非有限")
        logits.append(values)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return softmax(np.stack(logits)/lock["winner"]["scale"]+np.asarray(lock["winner"]["bias"]), axis=-1)


def predict(output: Path, data: dict[str, np.ndarray]) -> np.ndarray:
    """两路种子按相同顺序组合，输出始终为五种子、全部样本、三类。"""
    lock = json.loads((output / "selection_locked.json").read_text("utf-8"))
    weight = lock["winner"]["new_weight"]
    new = neural_predict(output, data)
    if weight < 1:
        with (OUTPUT / "large_final" / "model.pkl").open("rb") as handle:
            earlier = pickle.load(handle)
        original, _ = legacy.predict(earlier, data)
        return weight*new+(1-weight)*original
    return new


def evaluate(output: Path) -> None:
    """训练完成和参数锁定后，执行一次五种子全覆盖验收。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖适配历史验收")
    data = large.data_for(("test",))
    with (output / "training_metadata.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    if set(metadata["training_groups"]) & set(data["source_groups"]):
        raise ValueError("来源交叉")
    old = load_npz(OUTPUT / "final" / "historical_predictions.npz")
    if not np.array_equal(old["sample_ids"], data["sample_ids"]) or not np.array_equal(old["labels"], data["labels"]):
        raise ValueError("历史样本或标签发生变化")
    p = predict(output, data)
    if p.shape != (5, 821, 3) or not np.isfinite(p).all() or not np.allclose(p.sum(-1), 1.):
        raise ValueError("五种子覆盖或概率不符合协议")
    runs = [{"seed": seed, **metric(data["labels"], row.argmax(1)), **probability_metrics(data["labels"], row)}
            for seed, row in zip(range(42, 47), p)]
    mean = float(np.mean([row["macro_f1"] for row in runs]))
    ensemble = p.mean(0)
    result = {"primary_metric": "five seed mean macro-F1", "mean_macro_f1": mean,
        "std_macro_f1": float(np.std([row["macro_f1"] for row in runs])), "runs": runs,
        "ensemble": {**metric(data["labels"], ensemble.argmax(1)), **probability_metrics(data["labels"], ensemble)},
        "baseline_reported": .6388, "relative_gain_vs_reported_baseline": mean/.6388-1,
        "required_mean_macro_f1": .70268, "target_achieved": mean >= .70268, "coverage": 1., "n": 821,
        "new_expert_fraction": 1., "adaptive_historical_diagnostic": True,
        "bootstrap": paired_interval(data["labels"], p, old["baseline"], data["source_groups"])}
    save(output / "results.json", result)
    np.savez_compressed(output / "historical_predictions.npz", probabilities=p, ensemble=ensemble,
        sample_ids=data["sample_ids"], source_groups=data["source_groups"], labels=data["labels"], seeds=np.arange(42, 47))
    print(json.dumps(result), flush=True)


def main() -> None:
    """开发选择、五种子训练和一次历史验收分离。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("select", "fit", "evaluate"))
    args = parser.parse_args()
    output = OUTPUT / "qlora_final"
    output.mkdir(exist_ok=True)
    {"select": select, "fit": fit, "evaluate": evaluate}[args.action](output)


if __name__ == "__main__":
    main()
