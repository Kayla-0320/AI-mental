"""离线跨日模型的严格输入校验与预测；不接入平台，不输出风险等级。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from experiments.round14_common import OUT


def predict_pam(artifact: dict[str, Any], history: list[dict[str, Any]], target_day: int,
                source_schema: str = "studentlife_pam_valence_1_4") -> dict[str, Any]:
    """只允许已验证的PAM1–4效价，拒绝将用户0–100心情或风险分数直接代入。"""
    if source_schema != "studentlife_pam_valence_1_4": raise ValueError("模型仅验证了StudentLife PAM效价；其他量表须另行验证")
    if artifact["protocol"]["source"].get("source") != "https://zenodo.org/records/3529253": raise ValueError("模型来源与输入量表不一致")
    if type(target_day) is not int or not history: raise ValueError("目标日或历史无效")
    if len({str(row.get("person", "")) for row in history}) != 1 or not str(history[0].get("person", "")): raise ValueError("历史须属于同一个明确的匿名参与者")
    days = []; values = []
    for row in history:
        day, value = row.get("day"), row.get("value")
        if type(day) is not int or day >= target_day: raise ValueError("历史日无效或含目标/未来日")
        if type(value) not in {int, float} or not math.isfinite(value) or not 1 <= value <= 4: raise ValueError("PAM效价必须为1–4有限值")
        days.append(day); values.append((value - 1) / 3)
    if days != sorted(set(days)): raise ValueError("历史必须严格按日递增且不能重复")
    if days[-1] != target_day - 1: return {"status": "abstain", "reason": "latest_day_missing", "clinical_alert": False}
    included = [index for index, day in enumerate(days) if days[-1] - 14 < day <= days[-1]]
    days = [days[i] for i in included]; values = [values[i] for i in included]
    breaks = [i for i in range(1, len(days)) if days[i] - days[i - 1] > 3]
    if breaks: days, values = days[breaks[-1]:], values[breaks[-1]:]
    if len(values) < 3: return {"status": "abstain", "reason": "insufficient_observed_days_after_gap_reset", "clinical_alert": False}
    past = np.array(values); recent = past[-7:]
    slope = float(np.polyfit(np.array(days[-len(recent):]) - days[-len(recent)], recent, 1)[0])
    features = np.array([[past[-1], recent.mean(), np.median(recent), recent.std(), past.mean(), slope,
        min(len(values), 14) / 14, (days[-1] - days[-2]) / 14,
        np.sin(2 * np.pi * target_day / 7), np.cos(2 * np.pi * target_day / 7)]])
    config = artifact["config"]; kind = config["kind"]
    if kind == "last": value = past[-1]
    elif kind == "pooled_median": value = artifact["train_median"]
    elif kind == "mean7": value = recent.mean()
    elif kind == "median7": value = np.median(recent)
    elif kind == "ema":
        value = past[0]
        for observation in past[1:]: value = (1 - config["alpha"]) * value + config["alpha"] * observation
    elif kind == "shrink": value = config["alpha"] * past[-1] + (1 - config["alpha"]) * np.median(recent)
    elif kind == "personal_prior": value = (past.sum() + config["strength"] * artifact["train_median"]) / (len(past) + config["strength"])
    elif kind == "ridge": value = artifact["model"].predict(features)[0]
    else: raise ValueError("未知模型")
    forecast = float(1 + 3 * np.clip(value, 0, 1)); radius = 3 * artifact["validation_error_radius"]
    return {"status": "research_prediction", "origin_day": days[-1], "target_day": target_day,
        "predicted_pam_valence_1_4": forecast, "method": config["id"], "observed_history_days": len(days),
        "personal_recent_mean_1_4": float(1 + 3 * recent.mean()),
        "validation_error_band_1_4": [max(1., forecast - radius), min(4., forecast + radius)],
        "band_is_not_personal_coverage_guarantee": True, "clinical_alert": False,
        "warning": "offline exploratory model; no clinical prediction, no confirmed new-user benefit, no proactive strategy efficacy inference"}


def verify_artifacts() -> dict[str, Any]:
    """逐项回放两个冻结测试划分，验证可调用预测与实验输出完全一致。"""
    directory = OUT / "longitudinal" / "studentlife_calendar_days"; comparisons = {}
    for mode in ("same_person_future", "unseen_person_future"):
        artifact = joblib.load(directory / f"{mode}_model.joblib")
        samples = json.loads((directory / f"{mode}_test_windows.json").read_text(encoding="utf-8"))
        predictions = np.load(directory / f"{mode}_test_predictions.npz", allow_pickle=False)
        replay = []
        for row in samples:
            history = [{"person": row["person"], "day": day, "value": 1 + 3 * value} for day, value in zip(row["history_days"], row["history"], strict=True)]
            result = predict_pam(artifact, history, row["target_day"])
            replay.append((result["predicted_pam_valence_1_4"] - 1) / 3)
        maximum = float(np.max(np.abs(np.array(replay) - predictions[artifact["config"]["id"]])))
        if maximum > 1e-12: raise ValueError("离线运行时与冻结实验预测不一致")
        comparisons[mode] = {"n": len(samples), "maximum_normalized_error": maximum}
    return comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--history", type=Path)
    parser.add_argument("--artifact", type=Path, default=OUT / "longitudinal" / "studentlife_calendar_days" / "same_person_future_model.joblib")
    args = parser.parse_args()
    if args.history:
        payload = json.loads(args.history.read_text(encoding="utf-8"))
        print(json.dumps(predict_pam(joblib.load(args.artifact), payload["observations"], payload["target_day"], payload["source_schema"]), ensure_ascii=False))
    else: print(json.dumps(verify_artifacts()))
