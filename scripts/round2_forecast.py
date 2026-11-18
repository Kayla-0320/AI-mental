"""复算公开青少年自评的参与者隔离、仅使用过去信息的短期预测实验。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm"))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from assessment.forecasting.predictor import CONTEXT, SCHEMA, TARGETS, ema_prediction, history_features
from assessment.forecasting.sequence import SmallEmotionGRU

SEED = 42
SOURCE = "https://zenodo.org/records/17341614"
EXPECTED_MD5 = "374b7591810619c6d729950f706ee5bc"


def load_sequences(path: Path) -> tuple[dict[int, dict[int, list[float]]], dict[str, Any]]:
    """只提取参与者、顺序和当下自评；不读取全程均值作为预测特征。"""
    raw = path.read_bytes()
    if hashlib.md5(raw).hexdigest() != EXPECTED_MD5:
        raise ValueError("source checksum mismatch; review source version before proceeding")
    frame = pd.read_excel(path, engine="openpyxl")
    needed = ["record ID", "time", "mMean pos", "mMean neg"]
    if not set(needed).issubset(frame.columns):
        raise ValueError("source schema mismatch")
    data = frame[needed].replace(-999, np.nan)
    complete = data.dropna()
    if complete.duplicated(["record ID", "time"]).any():
        raise ValueError("duplicate participant/time would create ambiguous future targets")
    sequences: dict[int, dict[int, list[float]]] = {}
    for person, step, positive, negative in complete.itertuples(index=False, name=None):
        if not all(math.isfinite(float(v)) for v in (person, step, positive, negative)):
            raise ValueError("non-finite observation")
        if person != int(person) or step != int(step) or not (1 <= step <= 36):
            raise ValueError("invalid participant or sampling index")
        if not (0 <= positive <= 100 and 0 <= negative <= 100):
            raise ValueError("self-report outside source scale")
        sequences.setdefault(int(person), {})[int(step)] = [positive / 100, negative / 100]
    return sequences, {
        "source": SOURCE, "source_md5": EXPECTED_MD5,
        "source_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(frame),
        "participant_ids_in_file": int(frame["record ID"].nunique()),
        "participants_with_observed_targets": len(sequences), "usable_rows": len(complete),
        "missing_target_or_id_rows": len(frame) - len(complete),
        "age_range_in_file": [float(frame["age"].min()), float(frame["age"].max())],
        "feature_columns": ["mMean pos", "mMean neg"],
        "excluded_future_derived_columns": ["gMean pos", "gMean neg", "gICC pos", "gICC neg",
                                            "smICC pos", "smICC neg"],
    }


def participant_split(ids: list[int]) -> dict[str, list[int]]:
    """固定种子将参与者分为训练/验证/测试，避免重叠窗口跨组泄漏。"""
    shuffled = np.random.default_rng(SEED).permutation(sorted(ids)).tolist()
    cut1, cut2 = int(len(ids) * 0.7), int(len(ids) * 0.85)
    return {"train": shuffled[:cut1], "validation": shuffled[cut1:cut2], "test": shuffled[cut2:]}


def windows(sequences: dict[int, dict[int, list[float]]], ids: list[int], horizon: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造连续过去窗口和指定未来目标，保留参与者用于聚类统计。"""
    if type(horizon) is not int or horizon not in (1, 3):
        raise ValueError("horizon must be a supported positive future sampling step (1 or 3)")
    x, y, groups, origins = [], [], [], []
    for person in sorted(ids):
        observations = sequences[person]
        for origin in sorted(observations):
            past = list(range(origin - CONTEXT + 1, origin + 1))
            target = origin + horizon
            if target not in observations or not all(step in observations for step in past):
                continue
            x.append([observations[step] for step in past])
            y.append(observations[target])
            groups.append(person)
            origins.append(origin)
    return np.asarray(x), np.asarray(y), np.asarray(groups), np.asarray(origins)


def participant_mae(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> float:
    """先算每个人的绝对误差，再等权汇总参与者。"""
    return float(np.mean([np.abs(y[groups == group] - prediction[groups == group]).mean()
                          for group in np.unique(groups)]))


def summary(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """报告原始 0–100 自评尺度的误差，避免把窗口数当独立人数。"""
    error = prediction - y
    return {"mae": float(np.abs(error).mean() * 100),
            "participant_equal_mae": participant_mae(y, prediction, groups) * 100,
            "positive_mae": float(np.abs(error[:, 0]).mean() * 100),
            "negative_mae": float(np.abs(error[:, 1]).mean() * 100),
            "rmse": float(np.sqrt((error ** 2).mean()) * 100)}


def train_gru(train: tuple[np.ndarray, ...], valid: tuple[np.ndarray, ...]
              ) -> tuple[SmallEmotionGRU, dict[str, Any]]:
    """训练集梯度拟合、验证集挑选 epoch；测试集不参与早停。"""
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = SmallEmotionGRU()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    tx, ty = torch.tensor(train[0], dtype=torch.float32), torch.tensor(train[1], dtype=torch.float32)
    vx = torch.tensor(valid[0], dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    best, best_state, best_epoch = float("inf"), copy.deepcopy(model.state_dict()), 0
    losses = []
    for epoch in range(1, 81):
        model.train()
        order = torch.randperm(len(tx), generator=generator)
        for batch in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(tx[batch]), ty[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(vx).numpy().clip(0, 1)
        score = participant_mae(valid[1], prediction, valid[2])
        losses.append(score)
        if score < best:
            best, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": 80, "selected_epoch": best_epoch,
                   "validation_participant_mae_by_epoch": losses}


def bootstrap_improvement(y: np.ndarray, baseline: np.ndarray, selected: np.ndarray,
                          groups: np.ndarray) -> list[float]:
    """以参与者为单位配对重采样，估计相对末值基线的误差减少。"""
    differences = np.array([np.abs(y[groups == group] - baseline[groups == group]).mean()
                            - np.abs(y[groups == group] - selected[groups == group]).mean()
                            for group in np.unique(groups)]) * 100
    rng = np.random.default_rng(SEED)
    estimates = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def run(source: Path, output: Path, artifact_dir: Path) -> dict[str, Any]:
    """执行预定两种跨度、五种方法的完整比较并保存可重放产物。"""
    sequences, audit = load_sequences(source)
    split = participant_split(list(sequences))
    output.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {"seed": SEED, "source_audit": audit, "participant_split": split,
                              "protocol": "participant_disjoint_past_only_v1", "horizons": {}}
    metadata: dict[str, Any] = {
        "schema": SCHEMA, "model_id": "adolescent_esm_research_20261115_v1", "context": CONTEXT,
        "targets": TARGETS, "scale": [0, 100], "fit_split": "train", "selection_split": "validation",
        "source_sha256": audit["source_sha256"], "horizons": {},
        "limitations": ["small_male_non_Chinese_sample", "overlapping_windows_not_independent",
                         "ordinal_sampling_index_not_clock_time", "no_intervention_effect_data"],
    }
    for horizon in (1, 3):
        parts = {name: windows(sequences, ids, horizon) for name, ids in split.items()}
        if any(len(part[0]) < 10 for part in parts.values()):
            raise ValueError("insufficient windows; do not fit a misleading model")
        train, valid, test = (parts[name] for name in ("train", "validation", "test"))
        # 拟合原尺度上特征，不用测试统计做归一化；Ridge 正则只看验证组。
        ridge_options = []
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            ridge = Ridge(alpha=alpha).fit(history_features(train[0]), train[1])
            score = participant_mae(valid[1], ridge.predict(history_features(valid[0])).clip(0, 1), valid[2])
            ridge_options.append((score, alpha, ridge))
        _, alpha, ridge = min(ridge_options, key=lambda item: item[0])
        gru, gru_info = train_gru(train, valid)
        predictions: dict[str, dict[str, np.ndarray]] = {}
        for name, part in parts.items():
            with torch.no_grad():
                gp = gru(torch.tensor(part[0], dtype=torch.float32)).numpy().clip(0, 1)
            predictions[name] = {"persistence": part[0][:, -1], "past_mean": part[0].mean(axis=1),
                                 "ema": ema_prediction(part[0]),
                                 "ridge": ridge.predict(history_features(part[0])).clip(0, 1), "gru": gp}
        selected_method = min(predictions["validation"],
                              key=lambda method: participant_mae(valid[1], predictions["validation"][method], valid[2]))
        residual = np.quantile(np.abs(predictions["validation"][selected_method] - valid[1]),
                               0.9, axis=0, method="higher")
        selected_test = predictions["test"][selected_method]
        record = {"horizon_sampling_steps": horizon, "context_sampling_steps": CONTEXT,
                  "window_counts": {name: len(part[0]) for name, part in parts.items()},
                  "participant_counts_with_windows": {name: len(np.unique(part[2])) for name, part in parts.items()},
                  "selected_method": selected_method, "ridge_alpha": alpha, "gru_training": gru_info,
                  "validation": {method: summary(valid[1], p, valid[2]) for method, p in predictions["validation"].items()},
                  "test": {method: summary(test[1], p, test[2]) for method, p in predictions["test"].items()},
                  "paired_participant_bootstrap_improvement_vs_persistence_ci95":
                      bootstrap_improvement(test[1], predictions["test"]["persistence"], selected_test, test[2]),
                  "validation_absolute_error_quantile": residual.tolist(),
                  "test_error_band_coverage_by_target": (np.abs(selected_test - test[1]) <= residual).mean(axis=0).tolist()}
        report["horizons"][str(horizon)] = record
        metadata["horizons"][str(horizon)] = {"selected_method": selected_method,
                                                "validation_absolute_error_quantile": residual.tolist()}
        prefix = f"h{horizon}_"
        weights[prefix + "coef"], weights[prefix + "intercept"] = ridge.coef_, ridge.intercept_
        for key, tensor in gru.state_dict().items():
            weights[prefix + "gru_" + key] = tensor.detach().numpy()
        np.savez_compressed(output / f"h{horizon}_test_predictions.npz", y=test[1],
                            participant=test[2], origin_step=test[3], **predictions["test"])
        print(json.dumps({"horizon": horizon, "selected": selected_method,
                          "test": record["test"]}, ensure_ascii=False), flush=True)
    weight_path = artifact_dir / "forecast_weights.npz"
    np.savez_compressed(weight_path, **weights)
    metadata["weights_file"] = weight_path.name
    metadata["weights_sha256"] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    (artifact_dir / "forecast.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "algorithm/artifacts/round2_forecast")
    args = parser.parse_args()
    run(args.source, args.output, args.artifact_dir)
