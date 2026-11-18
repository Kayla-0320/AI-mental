"""只使用过去自评的跨日预测与个人漂移研究，完全独立于线上风险和升级规则。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.round14_common import DATA, OUT, ROOT, SEED, freeze, save, sha256


def daily_samples(frame: pd.DataFrame, lookback: int = 14, minimum: int = 3) -> list[dict[str, Any]]:
    """目标严格为下一日；缺测不补零、不跨空档预测、历史窗口不含目标。"""
    required = {"person", "day", "value"}
    if not required <= set(frame.columns) or frame.duplicated(["person", "day"]).any():
        raise ValueError("逐人逐日记录缺字段或重复")
    if not np.isfinite(frame["value"].to_numpy(float)).all() or not frame.value.between(0, 1).all():
        raise ValueError("值须是已核实的有限0–1自评")
    result = []
    for person, rows in frame.groupby("person", sort=True):
        observations = {int(row.day): float(row.value) for row in rows.itertuples()}
        for origin in sorted(observations):
            days = [day for day in sorted(observations) if origin - lookback < day <= origin]
            # 超过三天未观测后重新暖启动，不能沿用过时均值。
            breaks = [index for index in range(1, len(days)) if days[index] - days[index - 1] > 3]
            if breaks: days = days[breaks[-1]:]
            if origin + 1 not in observations or len(days) < minimum: continue
            values = np.array([observations[day] for day in days])
            recent = values[-7:]
            slope = float(np.polyfit(np.array(days[-len(recent):]) - days[-len(recent)], recent, 1)[0]) if len(recent) > 1 else 0.
            x = [values[-1], float(np.mean(recent)), float(np.median(recent)), float(np.std(recent)),
                 float(np.mean(values)), slope, min(len(values), 14) / 14,
                 (days[-1] - days[-2]) / 14, np.sin(2 * np.pi * (origin + 1) / 7), np.cos(2 * np.pi * (origin + 1) / 7)]
            result.append({"person": str(person), "origin": origin, "target_day": origin + 1,
                           "history_days": days, "history": values.tolist(), "x": x,
                           "y": observations[origin + 1], "last": observations[origin]})
    return result


def participant_split(ids: list[str]) -> dict[str, list[str]]:
    """参与者不可跨组，固定种子保留可核对的完整分组。"""
    order = np.random.default_rng(SEED).permutation(sorted(ids)).tolist()
    n = len(order)
    return dict(zip(("train", "validation", "test"), (order[:int(n * .7)], order[int(n * .7):int(n * .85)], order[int(n * .85):])))


def select_partitions(samples: list[dict[str, Any]], mode: str) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """全局日历前推；未见用户协议同时执行参与者隔离和时间隔离。"""
    days = sorted({row["target_day"] for row in samples})
    cut1, cut2 = days[int(len(days) * .6)], days[int(len(days) * .8)]
    people = participant_split(sorted({row["person"] for row in samples}))
    partitions = {"train": [], "validation": [], "test": []}
    for row in samples:
        split = "train" if row["target_day"] < cut1 else ("validation" if row["target_day"] < cut2 else "test")
        if mode == "same_person_future" or row["person"] in people[split]: partitions[split].append(row)
    if any(not rows for rows in partitions.values()): raise ValueError("严格划分后有空分组，不降低协议换取结果")
    return partitions, {"mode": mode, "validation_start_day": cut1, "test_start_day": cut2,
        "participant_partition": people if mode == "unseen_person_future" else "shared people; globally separated time",
        "fit_max_target_day": max(row["target_day"] for row in partitions["train"]),
        "test_min_origin": min(row["origin"] for row in partitions["test"]),
        "selection_uses_test": False}


def predictions(config: dict[str, Any], rows: list[dict], train: list[dict], model: Any = None) -> np.ndarray:
    """训练总体先验与个人过去分开，所有状态都止于预测起点。"""
    kind = config["kind"]
    if kind == "ridge": return np.clip(model.predict(np.array([row["x"] for row in rows])), 0, 1)
    center = float(np.median([row["y"] for row in train]))
    output = []
    for row in rows:
        history = np.array(row["history"])
        if kind == "last": value = history[-1]
        elif kind == "pooled_median": value = center
        elif kind == "mean7": value = np.mean(history[-7:])
        elif kind == "median7": value = np.median(history[-7:])
        elif kind == "ema":
            value = history[0]
            for observation in history[1:]: value = (1 - config["alpha"]) * value + config["alpha"] * observation
        elif kind == "shrink": value = config["alpha"] * history[-1] + (1 - config["alpha"]) * np.median(history[-7:])
        elif kind == "personal_prior": value = (history.sum() + config["strength"] * center) / (len(history) + config["strength"])
        else: raise ValueError("未知跨日方法")
        output.append(value)
    return np.clip(np.array(output), 0, 1)


def metrics(rows: list[dict], pred: np.ndarray, scale: float, change: float) -> dict[str, Any]:
    """同时报告逐窗口和等权参与者误差；方向只在实测明显变化时统计。"""
    truth = np.array([row["y"] for row in rows]); last = np.array([row["last"] for row in rows])
    groups = np.array([row["person"] for row in rows]); error = np.abs(truth - pred)
    mask = np.abs(truth - last) >= change
    return {"n": len(rows), "people": len(set(groups)), "mae": float(error.mean() * scale),
        "participant_equal_mae": float(np.mean([error[groups == group].mean() for group in np.unique(groups)]) * scale),
        "rmse": float(np.sqrt(np.mean((truth - pred)**2)) * scale),
        "direction_n": int(mask.sum()), "direction_accuracy_on_changed_days": float(np.mean(np.sign(pred[mask] - last[mask]) == np.sign(truth[mask] - last[mask]))) if mask.any() else None,
        "stable_day_change_flag_rate_proxy": float(np.mean(np.abs(pred[~mask] - last[~mask]) >= change)) if (~mask).any() else None,
        "change_flag_is_not_clinical_alert": True}


def cluster_ci(rows: list[dict], first: np.ndarray, second: np.ndarray, scale: float) -> list[float]:
    """按参与者聚类自助法，负值表示第一种预测误差更低。"""
    groups = np.array([row["person"] for row in rows]); truth = np.array([row["y"] for row in rows])
    differences = np.abs(first - truth) - np.abs(second - truth)
    per_person = np.array([differences[groups == group].mean() for group in np.unique(groups)])
    rng = np.random.default_rng(SEED)
    values = per_person[rng.integers(0, len(per_person), size=(5000, len(per_person)))].mean(axis=1) * scale
    return np.quantile(values, [.025, .975]).tolist()


def drift_trace(frame: pd.DataFrame, warmup: int = 7, floor: float = .05) -> dict[str, Any]:
    """首七个已观测日建立个人参考，冻结后仅研究数值漂移，不输出风险等级。"""
    rows = []
    for person, series in frame.groupby("person", sort=True):
        series = series.sort_values("day")
        if len(series) <= warmup: continue
        # 预先固定的参考窗；后续不反向更改已产生的基线。
        initial = series.iloc[:warmup]
        if int(initial.day.iloc[-1] - initial.day.iloc[0]) > 14: continue
        mean = float(initial.value.mean()); std = max(float(initial.value.std(ddof=1)), floor)
        state = {"personal": [0., 0.], "fixed_0.3_0.1": [0., 0.]}
        last_day = int(initial.day.iloc[-1]); previous = float(initial.value.iloc[-1])
        for item in series.iloc[warmup:].itertuples():
            if item.day - last_day > 3: break  # 过长缺测需重新暖启动，当前轨迹停止。
            flags = {}
            for kind, (center, spread) in {"personal": (mean, std), "fixed_0.3_0.1": (.3, .1)}.items():
                z = (float(item.value) - center) / spread
                state[kind][0] = max(0., state[kind][0] + z - .5)
                state[kind][1] = max(0., state[kind][1] - z - .5)
                flags[kind] = max(state[kind]) > 4.
                if flags[kind]: state[kind] = [0., 0.]
            rows.append({"person": str(person), "day": int(item.day), "baseline_end_day": int(initial.day.iloc[-1]),
                "personal_mean": mean, "personal_std": std, "observed_value": float(item.value),
                "stable_step_proxy": abs(float(item.value) - previous) < .1, **flags})
            last_day = int(item.day); previous = float(item.value)
    summary = {kind: {"flagged_days": sum(row[kind] for row in rows),
        "stable_step_flag_rate_proxy": float(np.mean([row[kind] for row in rows if row["stable_step_proxy"]])) if any(row["stable_step_proxy"] for row in rows) else None}
        for kind in ("personal", "fixed_0.3_0.1")}
    return {"rows": rows, "summary": summary, "reference": "7 observed days spanning <=14 days; std floor .05",
        "limitation": "No independent change-point labels. Stable-step proxy is not a false-positive rate or proof of sensitivity. Fewer flags alone is not success.",
        "online_thresholds_changed": False, "clinical_alert": False}


def run_dataset(name: str, frame: pd.DataFrame, source: dict[str, Any], scale: float, change: float) -> dict[str, Any]:
    """预注册候选和分割，验证集选型，冻结后一次性产生测试结果及可加载模型。"""
    output = OUT / "longitudinal" / name; output.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(["person", "day"]).reset_index(drop=True)
    frame.to_csv(output / "daily_observations.csv", index=False)
    samples = daily_samples(frame)
    configs = [{"id": kind, "kind": kind} for kind in ("last", "pooled_median", "mean7", "median7")]
    configs += [{"id": f"ema_{a}", "kind": "ema", "alpha": a} for a in (.1, .3, .5)]
    configs += [{"id": f"shrink_{a}", "kind": "shrink", "alpha": a} for a in (.25, .5, .75)]
    configs += [{"id": f"prior_{s}", "kind": "personal_prior", "strength": s} for s in (3, 10)]
    configs += [{"id": f"ridge_{a}", "kind": "ridge", "alpha": a} for a in (1., 10., 100.)]
    modes = ("same_person_future", "unseen_person_future")
    splits = {mode: select_partitions(samples, mode) for mode in modes}
    protocol = {"seed": SEED, "source": source, "scale": scale, "meaningful_change_normalized": change,
        "features": "only past self-report days <=origin, counts, past variability, slope and known target-day cyclic index",
        "lookback_calendar_or_sampling_days": 14, "minimum_observed_days": 3, "reset_after_missing_days": 3,
        "candidate_models": configs, "selection": "minimum participant-equal validation MAE; train only fitted parameters",
        "partitions": {mode: settings for mode, (_, settings) in splits.items()},
        "no_refit_on_validation_or_test": True, "code_sha256": sha256(Path(__file__))}
    freeze(output / "preregister.json", protocol)
    results = {}
    for mode, (parts, settings) in splits.items():
        train, valid, test = [parts[key] for key in ("train", "validation", "test")]
        models, validation = {}, {}
        for config in configs:
            model = None
            if config["kind"] == "ridge":
                model = make_pipeline(StandardScaler(), Ridge(alpha=config["alpha"]))
                model.fit(np.array([row["x"] for row in train]), np.array([row["y"] for row in train]))
            models[config["id"]] = model
            validation[config["id"]] = metrics(valid, predictions(config, valid, train, model), scale, change)
        winner = min(validation, key=lambda identifier: validation[identifier]["participant_equal_mae"])
        save(output / f"{mode}_selection_locked.json", {"winner": winner, "validation": validation, "partitions": settings})
        test_pred = {config["id"]: predictions(config, test, train, models[config["id"]]) for config in configs}
        winner_config = next(config for config in configs if config["id"] == winner)
        valid_residual = np.abs(np.array([row["y"] for row in valid]) - predictions(winner_config, valid, train, models[winner]))
        radius = float(np.quantile(valid_residual, .9, method="higher"))
        truth = np.array([row["y"] for row in test]); selected = test_pred[winner]
        result = {"winner": winner, "validation": validation,
            "test": {identifier: metrics(test, pred, scale, change) for identifier, pred in test_pred.items()},
            "selected_minus_last_participant_mae_ci95": cluster_ci(test, selected, test_pred["last"], scale),
            "selected_minus_mean7_participant_mae_ci95": cluster_ci(test, selected, test_pred["mean7"], scale),
            "selected_minus_median7_participant_mae_ci95": cluster_ci(test, selected, test_pred["median7"], scale),
            "validation_derived_90pct_error_band_radius": radius * scale,
            "observed_test_error_band_coverage": float(np.mean(np.abs(truth - selected) <= radius)),
            "error_band_warning": "selection and residual estimation share validation; descriptive, no conformal or individual guarantee",
            "n_train": len(train), "n_validation": len(valid), "n_test": len(test),
            "test_people": sorted({row["person"] for row in test}), "protocol": settings}
        results[mode] = result
        np.savez_compressed(output / f"{mode}_test_predictions.npz", truth=truth,
            people=np.array([row["person"] for row in test]), origin=np.array([row["origin"] for row in test]), **test_pred)
        save(output / f"{mode}_test_windows.json", [{k: v for k, v in row.items() if k != "x"} for row in test])
        joblib.dump({"config": winner_config, "model": models[winner], "train_median": float(np.median([row["y"] for row in train])),
            "validation_error_radius": radius, "protocol": protocol, "scale": scale}, output / f"{mode}_model.joblib")
    save(output / "drift_results.json", drift_trace(frame))
    result = {"source": source, "results": results, "rows": len(frame), "people": int(frame.person.nunique()),
        "clinical_effectiveness_evidence": False, "deployed": False}
    save(output / "results.json", result)
    print(json.dumps({"dataset": name, "results": {mode: {"winner": r["winner"], "last": r["test"]["last"],
        "selected": r["test"][r["winner"]], "ci95": r["selected_minus_last_participant_mae_ci95"]} for mode, r in results.items()}}, ensure_ascii=False), flush=True)
    return result


def adolescent_daily() -> tuple[pd.DataFrame, dict[str, Any]]:
    """重用已曝光的青少年 ESM；只称下一采样日，不捏造真实日期或周末。"""
    path = ROOT / "round2" / "forecast" / "minimal_dataset_emolife.xlsx"
    frame = pd.read_excel(path).replace(-999, np.nan)
    frame = frame[["record ID", "time", "mMean neg"]].dropna()
    frame["day"] = ((frame["time"].astype(int) - 1) // 4).astype(int)
    daily = frame.groupby(["record ID", "day"])["mMean neg"].mean().reset_index()
    daily.columns = ["person", "day", "value"]; daily["person"] = daily.person.astype(str); daily["value"] /= 100
    return daily, {"source": "https://zenodo.org/records/17341614", "sha256": sha256(path),
        "prior_use": "round2 and later; reused-source diagnostic, NOT a new independent holdout",
        "day_semantics": "4 scheduled samples = one sampling day; actual dates unavailable; weekends excluded in original design",
        "target": "daily mean of observed negative-affect 0–100 self-reports; missing samples not zero",
        "people": "male adolescents age12–17; not project-user followup"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", choices=["adolescent", "studentlife"], required=True)
    args = parser.parse_args()
    if args.dataset == "adolescent":
        data, info = adolescent_daily(); run_dataset("adolescent_sampling_days_reused", data, info, 100., .1)
    else:
        from experiments.round14_studentlife import studentlife_daily
        data, info = studentlife_daily(); run_dataset("studentlife_calendar_days", data, info, 3., 1 / 3)
