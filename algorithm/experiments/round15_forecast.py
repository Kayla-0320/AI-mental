"""新用户的因果局部适应与留一用户前推验证；不改变线上风险或模型。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from experiments.round14_common import ROOT, freeze, save, sha256
from experiments.round14_longitudinal import daily_samples, select_partitions, metrics, cluster_ci

OUT = ROOT / "round15"


def forecast(history: list[float], config: dict[str, Any]) -> float:
    """只看已到达的个人观测；不依赖该用户在群体训练中的身份或全程均值。"""
    h = np.asarray(history, dtype=float)
    if len(h) < 3 or not np.isfinite(h).all() or np.any((h < 0) | (h > 1)): raise ValueError("需要至少三个有限0–1过去观测")
    kind = config["kind"]
    if kind == "last": return float(h[-1])
    if kind == "mean": return float(h[-config["window"]:].mean())
    if kind == "median": return float(np.median(h[-config["window"]:]))
    if kind == "blend": return float(config["weight"] * h[-1] + (1-config["weight"]) * np.median(h[-config["window"]:]))
    if kind == "ema":
        value = h[0]
        for point in h[1:]: value = (1-config["alpha"]) * value + config["alpha"] * point
        return float(value)
    if kind == "local":
        # 在历史内部先预测再看该时点结局，至少三个局部验证点；不足时回退保持法。
        candidates = [{"kind": "last"}, {"kind": "median", "window": 3}, {"kind": "mean", "window": 3}]
        if len(h) < 6: return float(h[-1])
        losses = [np.mean([abs(forecast(h[:i].tolist(), c)-h[i]) for i in range(max(3,len(h)-7),len(h))]) for c in candidates]
        return forecast(h.tolist(), candidates[int(np.argmin(losses))])
    raise ValueError("未知方法")


def configs() -> list[dict[str, Any]]:
    """固定小候选库，包含保守保持法与禁止全局先验收缩的个人方法。"""
    values = [{"kind": "last"}, {"kind": "local"}]
    values += [{"kind": k, "window": w} for k in ("mean", "median") for w in (2,3,7)]
    values += [{"kind": "blend", "window": w, "weight": a} for w in (3,7) for a in (.25,.5,.75)]
    values += [{"kind": "ema", "alpha": a} for a in (.1,.3,.5,.8)]
    return [{"id": "c"+str(i), **c} for i,c in enumerate(values)]


def predict(rows: list[dict], config: dict[str, Any]) -> np.ndarray:
    """逐窗口调用同一因果推理函数。"""
    return np.array([forecast(row["history"], config) for row in rows])


def choose(valid: list[dict], bank: list[dict]) -> tuple[dict, dict]:
    """按其他用户的验证期等权MAE选型，拒绝比保持法更差的尾部误差候选。"""
    scores = {}
    for config in bank:
        p = predict(valid,config); groups = np.array([r["person"] for r in valid]); truth = np.array([r["y"] for r in valid])
        person_errors = [np.abs(p[groups==g]-truth[groups==g]).mean() for g in np.unique(groups)]
        scores[config["id"]] = {"person_mae": float(np.mean(person_errors)), "tail_mae": float(np.quantile(person_errors,.75)), "rmse": float(np.sqrt(np.mean((p-truth)**2)))}
    base = scores[bank[0]["id"]]
    eligible = [c for c in bank if scores[c["id"]]["tail_mae"] <= base["tail_mae"]+1e-12 and scores[c["id"]]["rmse"] <= base["rmse"]+1e-12]
    winner = min(eligible,key=lambda c:scores[c["id"]]["person_mae"])
    return winner,scores


def run() -> None:
    """保留上一轮严格划分，另外逐用户留出；复用数据明确标为探索性。"""
    directory = OUT / "forecast"; directory.mkdir(parents=True,exist_ok=True)
    old = ROOT / "round14/longitudinal/studentlife_calendar_days"
    frame = pd.read_csv(old/"daily_observations.csv",dtype={"person":str},float_precision="round_trip")
    samples = daily_samples(frame); parts, settings = select_partitions(samples,"unseen_person_future")
    bank = configs()
    freeze(directory/"protocol.json", {"seed":42,"configs":bank,"source_sha256":sha256(old/"daily_observations.csv"),
        "code_sha256":sha256(Path(__file__)),"source_status":"reused and previously exposed; exploratory, not fresh independent validation",
        "selection":"validation other users only; minimize person MAE with tail and RMSE guard versus last",
        "strict_split":settings,"lopo":"each target user absent from all hyperparameter selection; validation earlier than test; 23 users with observed late targets; all49 IDs enumerated",
        "cold_start":"at least3 observed personal days; not zero-history prediction","platform_audio_kg":"deferred"})
    selected,scores = choose(parts["validation"],bank)
    save(directory/"strict_selection_lock.json",{"winner":selected,"scores":scores})
    test = parts["test"]; output = {"strict":evaluate(test,predict(test,selected)),"strict_winner":selected}
    # 保持既定全局日期界限，不为扩大样本而缩短预测间隔或修改尾部日期。
    validation = [r for r in samples if settings["validation_start_day"] <= r["target_day"] < settings["test_start_day"]]
    held = [r for r in samples if r["target_day"] >= settings["test_start_day"]]
    locks = {}; combined = []; predictions = []
    for person in sorted(frame.person.unique()):
        valid = [r for r in validation if r["person"] != person]; target = [r for r in held if r["person"] == person]
        choice,score = choose(valid,bank)
        locks[person] = {"winner":choice,"validation_people":sorted({r["person"] for r in valid}),"validation_max_target":max(r["target_day"] for r in valid),"test_n":len(target),"scores":score}
        combined.extend(target); predictions.extend(predict(target,choice).tolist())
    save(directory/"lopo_selection_locks.json",locks)
    output["lopo"] = evaluate(combined,np.array(predictions))
    output["all_ids"] = len(locks); output["ids_without_late_targets"] = sum(v["test_n"]==0 for v in locks.values())
    save(directory/"results.json",output)
    print(json.dumps(output,ensure_ascii=False),flush=True)


def evaluate(rows: list[dict], selected: np.ndarray) -> dict[str, Any]:
    """完整比较保持法、滚动均值和上一轮慢EMA，不把回测提升称作新来源确认。"""
    methods = {"selected":selected,"last":predict(rows,{"kind":"last"}),"mean7":predict(rows,{"kind":"mean","window":7}),"old_ema_0.1":predict(rows,{"kind":"ema","alpha":.1})}
    result = {name:metrics(rows,p,3.,1/3) for name,p in methods.items()}
    result["ci95_person_mae_delta"] = {name:cluster_ci(rows,selected,p,3.) for name,p in methods.items() if name!="selected"}
    name = "strict" if len(rows)==13 else "lopo"
    np.savez_compressed(OUT/"forecast"/(name+"_predictions.npz"),truth=np.array([r["y"] for r in rows]),people=np.array([r["person"] for r in rows]),**methods)
    save(OUT/"forecast"/(name+"_windows.json"),rows)
    return result


if __name__ == "__main__": run()
