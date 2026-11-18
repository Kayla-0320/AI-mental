"""CPU来源分组嵌套OOF决策实验；703退出选择，不改第五六轮工件。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
NAMES = ["roberta_mean", "bge_cls", "visual", "audio", "face_summary"]
FLOORS = [.75, .8, .85, .9, .95]
MAIN_FLOOR = .85


def sha(path: Path) -> str:
    """计算文件身份，不修改原始数据。"""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: Any) -> None:
    """写入本轮UTF-8元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> dict[str, np.ndarray]:
    """只读取非对象NPZ，避免pickle执行。"""
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def take(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    """所有样本字段同步切片。"""
    return {key: values[indices] for key, values in data.items()}


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """固定负、中、正三类口径，不用准确率代替macro-F1。"""
    matrix = np.bincount(3 * y.astype(int) + pred.astype(int), minlength=9).reshape(3, 3)
    tp = np.diag(matrix)
    support, selected = matrix.sum(1), matrix.sum(0)
    precision = np.divide(tp, selected, out=np.zeros(3), where=selected > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * tp, support + selected, out=np.zeros(3), where=(support + selected) > 0)
    return {"n": len(y), "macro_f1": float(f1.mean()), "accuracy": float(tp.sum() / len(y)),
            "per_class_f1": f1.tolist(), "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
            "confusion_matrix": matrix.tolist(), "pred_counts": selected.tolist(), "support": support.tolist()}


def availability(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """只读观测可用性，不读标签、来源或演员身份。"""
    text = data["modality_mask"][:, 0] & (data["quality"][:, 0] > 0)
    text &= np.array([bool(str(value).strip()) for value in data["raw_text"]])
    face = text & data["modality_mask"][:, 2] & (data["quality"][:, 2] > 0)
    face &= np.isfinite(data["face_summary"][:, 24]) & (data["face_summary"][:, 24] >= .6)
    return text, face


def matrix(data: dict[str, np.ndarray], scalers: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """训练折内拟合标准化；无效模态先归零，防止NaN乘零污染。"""
    fitting = scalers is None
    scalers = {} if fitting else scalers
    blocks = []
    for name in NAMES:
        values = data[name].copy()
        if name in ["visual", "audio"]:
            column = ["text", "audio", "visual"].index(name)
            valid = data["modality_mask"][:, column:column + 1] & (data["quality"][:, column:column + 1] > 0)
            values = np.where(valid, values * data["quality"][:, column:column + 1], 0.)
        if not np.isfinite(values).all():
            raise ValueError("有效表征含非有限值")
        if fitting:
            scalers[name] = StandardScaler().fit(values)
        block = scalers[name].transform(values) / np.sqrt(values.shape[1])
        blocks.append(block * (.5 if name == "face_summary" else 1.))
    return np.column_stack(blocks), scalers


def normalized(values: np.ndarray) -> np.ndarray:
    """保留第五轮主分支的逐行margin尺度约定，去掉703选出的中性偏置。"""
    return (values - values.mean(1, keepdims=True)) / np.maximum(values.std(1, keepdims=True), 1e-6)


def fit_pipeline(data: dict[str, np.ndarray], auxiliary: list[dict[str, Any]]) -> dict[str, Any]:
    """只在目标训练折和固定ECISA辅助训练上拟合CPU模型。"""
    x, scalers = matrix(data)
    primary = LogisticRegression(C=.1, class_weight="balanced", max_iter=700, random_state=SEED).fit(x, data["labels"])
    texts = [row["text"] for row in auxiliary] + data["raw_text"].astype(str).tolist()
    labels = np.concatenate([np.array([row["label"] for row in auxiliary]), data["labels"]])
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, max_features=80000, sublinear_tf=True)
    sparse = vectorizer.fit_transform(texts)
    fallback = LogisticRegression(C=1., class_weight="balanced", max_iter=500, random_state=SEED).fit(sparse, labels)
    return {"primary": primary, "scalers": scalers, "vectorizer": vectorizer, "fallback": fallback}


def raw_predict(model: dict[str, Any], data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """完整CPU管线：有效人脸走融合，其余有效文字走折内拟合的文字分类器。"""
    valid, face = availability(data)
    primary = np.full((len(valid), 3), np.nan)
    fallback = np.full_like(primary, np.nan)
    if face.any():
        x, _ = matrix(take(data, np.flatnonzero(face)), model["scalers"])
        primary[face] = normalized(model["primary"].decision_function(x))
    missing = valid & ~face
    if missing.any():
        x = model["vectorizer"].transform(data["raw_text"][missing].astype(str).tolist())
        fallback[missing] = normalized(model["fallback"].decision_function(x))
    scores = np.where(face[:, None], primary, fallback)
    return {"scores": scores, "primary": primary, "face": face, "valid": valid}


def probabilities(scores: np.ndarray, temperatures: dict[str, float], face: np.ndarray) -> np.ndarray:
    """应用仅从OOF拟合的温度；无效行保留NaN。"""
    scale = np.where(face, temperatures["face"], temperatures["fallback"])
    z = scores.astype(float) / scale[:, None]
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def temperature(scores: np.ndarray, y: np.ndarray) -> float:
    """只以当前训练侧OOF负对数似然拟合一个标量。"""
    objective = lambda value: float(np.mean(logsumexp(scores / np.exp(value), axis=1) - scores[np.arange(len(y)), y] / np.exp(value)))
    result = minimize_scalar(objective, bounds=(np.log(.05), np.log(20.)), method="bounded")
    return float(np.exp(result.x))


def calibrate(raw: dict[str, np.ndarray], y: np.ndarray, scope: np.ndarray) -> dict[str, Any]:
    """来源训练侧校准，少于30条或缺类的分支回退训练侧全局温度。"""
    global_t = temperature(raw["scores"][scope], y[scope])
    result: dict[str, Any] = {"global": global_t, "counts": {}}
    for name, mask in [("face", raw["face"]), ("fallback", ~raw["face"])]:
        selected = scope & mask
        result["counts"][name] = int(selected.sum())
        result[name] = temperature(raw["scores"][selected], y[selected]) if selected.sum() >= 30 and len(np.unique(y[selected])) == 3 else global_t
    return result


def rule_grid() -> list[dict[str, Any]]:
    """预先固定的决策族；不按历史集挑网格或按未知来源ID设阈值。"""
    result: list[dict[str, Any]] = [{"kind": "argmax"}]
    neutral = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .6, .7, .8, .9, .95, 1.01]
    for nt in [1.01, .95, .85, .7, .55, .4, .25]:
        for mt in neutral:
            result.append({"kind": "band", "negative_guard": nt, "neutral_threshold": mt})
    for negative in [-.5, 0., .25, .5, .75, 1., 1.5, 2.]:
        for middle in [-1.5, -1., -.5, 0., .25, .5, .75, 1., 1.5, 2.]:
            result.append({"kind": "bias", "negative_bias": negative, "neutral_bias": middle})
    for lo in [-.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0.]:
        for hi in [0., .1, .2, .3, .4, .5, .6, .7, .8, .9]:
            if hi > lo:
                result.append({"kind": "ordinal", "lo": lo, "hi": hi})
    result.append({"kind": "all_negative"})
    return result


def decide(p: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    """执行锁定的离线情感决策；输入缺失时弃权，绝不读真实标签。"""
    valid = np.isfinite(p).all(1)
    pred = np.full(len(p), -1, dtype=int)
    values = p[valid]
    out = values.argmax(1)
    if rule["kind"] == "band":
        out[values[:, 1] >= rule["neutral_threshold"]] = 1
        out[values[:, 0] >= rule["negative_guard"]] = 0
    elif rule["kind"] == "bias":
        out = (np.log(np.maximum(values, 1e-15)) + np.array([rule["negative_bias"], rule["neutral_bias"], 0.])).argmax(1)
    elif rule["kind"] == "ordinal":
        value = values[:, 2] - values[:, 0]
        out = np.where(value < rule["lo"], 0, np.where(value > rule["hi"], 2, 1))
    elif rule["kind"] == "all_negative":
        out[:] = 0
    elif rule["kind"] != "argmax":
        raise ValueError("未知决策族")
    pred[valid] = out
    return pred


def select_rules(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """训练侧OOF约束优化，负类召回是离线成本条件而非临床风险阈值。"""
    options = [{"rule": rule, "metrics": metric(y, decide(p, rule))} for rule in rule_grid()]
    selected = {}
    for floor in FLOORS:
        feasible = [row for row in options if row["metrics"]["per_class_recall"][0] >= floor]
        selected[str(floor)] = max(feasible, key=lambda row: (row["metrics"]["macro_f1"], row["metrics"]["accuracy"]))
    return {"selected": selected, "all_candidates": options}


def fold_plan(data: dict[str, np.ndarray], splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """固定种子42，完整来源组不可跨同一训练与留出折。"""
    cv = StratifiedGroupKFold(splits, shuffle=True, random_state=SEED)
    output = list(cv.split(np.zeros(len(data["labels"])), data["labels"], data["source_groups"]))
    for train, valid in output:
        if set(data["source_groups"][train]) & set(data["source_groups"][valid]):
            raise ValueError("来源组跨折")
    return output


def fit_oof(data: dict[str, np.ndarray], auxiliary: list[dict[str, Any]], splits: int) -> dict[str, np.ndarray]:
    """每个分数的标准化、融合头、TF-IDF及回退分类器都未见该行所在来源组。"""
    n = len(data["labels"])
    result = {"scores": np.full((n, 3), np.nan), "primary": np.full((n, 3), np.nan),
              "face": np.zeros(n, bool), "valid": np.zeros(n, bool), "fold": np.full(n, -1)}
    for fold, (train, valid) in enumerate(fold_plan(data, splits)):
        model = fit_pipeline(take(data, train), auxiliary)
        values = raw_predict(model, take(data, valid))
        for key in values:
            result[key][valid] = values[key]
        result["fold"][valid] = fold
        del model
        gc.collect()
    return result


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """概率校准单独评价，不以softmax转换本身宣称校准成功。"""
    confidence = p.max(1)
    correct = p.argmax(1) == y
    ece = 0.
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        ix = (confidence > lo) & (confidence <= hi)
        if ix.any():
            ece += ix.mean() * abs(confidence[ix].mean() - correct[ix].mean())
    return {"nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean()),
            "brier": float(((p - np.eye(3)[y]) ** 2).sum(1).mean()), "ece10": float(ece)}


def paired_ci(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> list[float]:
    """来源组配对区间，只作已锁定实验的条件性区间。"""
    units = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(1000):
        ix = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        values.append(metric(y[ix], a[ix])["macro_f1"] - metric(y[ix], b[ix])["macro_f1"])
    return np.quantile(values, [.025, .975]).tolist()


def text_only(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """外部文字未观测媒体保持缺失，不借来源名称或标签路由。"""
    n = len(rows)
    return {"raw_text": np.array([row["text"] for row in rows]), "labels": np.array([row["label"] for row in rows]),
            "sample_ids": np.array([row["id"] for row in rows]), "source_groups": np.array([row["group"] for row in rows]),
            "modality_mask": np.column_stack([np.ones(n, bool), np.zeros((n, 2), bool)]),
            "quality": np.column_stack([np.ones(n), np.zeros((n, 2))]), "face_summary": np.zeros((n, 29))}


def run(previous: Path, output: Path) -> dict[str, Any]:
    """先锁定嵌套OOF选择，再一次读取历史821和既有ECISA3363评测。"""
    output.mkdir(parents=True, exist_ok=True)
    if (output / "preregister.json").exists():
        raise ValueError("输出目录已用，拒绝覆盖已注册实验")
    started = time.monotonic()
    parts = [load(previous / "face_features" / (name + ".npz")) for name in ["train", "selection"]]
    training = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    auxiliary_path = previous / "external/train.jsonl"
    auxiliary = [json.loads(line) for line in auxiliary_path.read_text(encoding="utf-8").splitlines()]
    immutable_paths = [previous / "manifest.json", previous.parent / "round6/manifest.json", previous / "face_search/selected.pkl",
                       previous / "availability_router/selection_locked.json", previous / "availability_router/historical_predictions.npz",
                       previous / "router_confirmation/predictions.npz"]
    immutable = {str(path): sha(path) for path in immutable_paths}
    protocol = {"seed": SEED, "training_n": len(training["labels"]), "training_groups": len(np.unique(training["source_groups"])),
                "outer_folds": 4, "inner_folds": 3, "no_703_selection": True, "gpu_used": False,
                "features": "frozen pretrained RoBERTa/BGE/CLIP, MFCC, frozen face statistics; no supervised target-finetuned encoder",
                "primary": {"C": .1, "class_weight": "balanced", "names": NAMES, "face_weight": .5, "minimum_face_coverage": .6, "prior_703_bias_removed": -1.},
                "fallback": "CPU charTFIDF1..3,80000max,min_df2; balanced LR,C1; fitted only outer/inner target-training folds+fixed ECISA official training",
                "auxiliary_train_sha256": sha(auxiliary_path), "auxiliary_training_n": len(auxiliary),
                "track_A": "only primary branch postprocessing of unchanged round5 model; OOF selection conditional on face-available rows; old supervised fallback untouched and not used to fit OOF rules",
                "track_B": "complete CPU crossfitted pipeline with fold-fitted TFIDF fallback; model changed versus round5, isolate decision effect using same-pipeline argmax",
                "calibration": "train-side inner-OOF temperature per branch; <30rows or absent class uses global training-side temperature",
                "rule_grid": rule_grid(), "negative_recall_constraints": FLOORS, "main_constraint": MAIN_FLOOR,
                "constraint_meaning": "negative sentiment recall, not clinical risk detection or service escalation",
                "selection": "macro-F1 subject to train-side OOF negative recall>=floor; accuracy breaks ties; fixed4-fold nested evaluation reports selection optimism",
                "historical_status": "821 and ECISA3363 repeatedly examined, including reviewer threshold sweeps; both are exploratory/regression, not independent confirmation",
                "acceptance": "report historical macro-F1>=.66, negative recall>=.85, and negative recall no lower than round5 as separate checks; no post-test revision",
                "code_sha256": sha(Path(__file__)), "old_critical_hashes": immutable,
                "input_hashes": {name: sha(previous / "face_features" / (name + ".npz")) for name in ["train", "selection"]},
                "deployed": False}
    save(output / "preregister.json", protocol)
    (output / "source").mkdir()
    (output / "source" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    n = len(training["labels"])
    outer_raw = {"scores": np.full((n, 3), np.nan), "primary": np.full((n, 3), np.nan),
                 "face": np.zeros(n, bool), "valid": np.zeros(n, bool), "fold": np.full(n, -1)}
    nested = {track: {str(floor): np.full(n, -1) for floor in FLOORS} for track in ["A", "B"]}
    nested_p = {track: np.full((n, 3), np.nan) for track in ["A", "B"]}
    audit = []
    for fold, (train_ix, valid_ix) in enumerate(fold_plan(training, 4)):
        a, b = take(training, train_ix), take(training, valid_ix)
        print(json.dumps({"outer_fold_started": fold, "training": len(train_ix), "validation": len(valid_ix)}), flush=True)
        inner = fit_oof(a, auxiliary, 3)
        (output / f"folds/{fold}").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / f"folds/{fold}/inner_oof.npz", **inner, labels=a["labels"],
                            sample_ids=a["sample_ids"], source_groups=a["source_groups"])
        policies = {}
        for track, scope in [("A", inner["valid"] & inner["face"]), ("B", inner["valid"])]:
            temps = calibrate(inner, a["labels"], scope)
            p = probabilities(inner["scores"], temps, inner["face"])
            selected = select_rules(p[scope], a["labels"][scope])
            policies[track] = {"temperatures": temps, "rules": selected["selected"]}
        entry = {"fold": fold, "train_ids": a["sample_ids"].tolist(), "validation_ids": b["sample_ids"].tolist(),
                 "train_groups": sorted(set(a["source_groups"])), "validation_groups": sorted(set(b["source_groups"])),
                 "inner_fold_assignments": inner["fold"].tolist(), "policies": policies}
        save(output / f"folds/{fold}/selection_locked.json", entry)
        model = fit_pipeline(a, auxiliary)
        with (output / f"folds/{fold}/model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        raw = raw_predict(model, b)
        for key in raw:
            outer_raw[key][valid_ix] = raw[key]
        outer_raw["fold"][valid_ix] = fold
        for track in ["A", "B"]:
            p = probabilities(raw["scores"], policies[track]["temperatures"], raw["face"])
            scope = raw["valid"] & raw["face"] if track == "A" else raw["valid"]
            nested_p[track][valid_ix[scope]] = p[scope]
            for floor in FLOORS:
                values = decide(p[scope], policies[track]["rules"][str(floor)]["rule"])
                nested[track][str(floor)][valid_ix[scope]] = values
        audit.append(entry)
        print(json.dumps({"outer_fold_done": fold, "seconds": round(time.monotonic() - started, 1)}), flush=True)
        del model
        gc.collect()
    final_policies = {}
    nested_metrics = {}
    for track, scope in [("A", outer_raw["valid"] & outer_raw["face"]), ("B", outer_raw["valid"])]:
        temps = calibrate(outer_raw, training["labels"], scope)
        p = probabilities(outer_raw["scores"], temps, outer_raw["face"])
        choices = select_rules(p[scope], training["labels"][scope])
        final_policies[track] = {"temperatures": temps, "main_rule": choices["selected"][str(MAIN_FLOOR)]["rule"], "cost_curve": choices["selected"], "selection_n": int(scope.sum())}
        save(output / f"selection_{track}_all_candidates.json", choices)
        baseline = outer_raw["scores"][scope].argmax(1)
        nested_metrics[track] = {"scope_n": int(scope.sum()), "raw_argmax": metric(training["labels"][scope], baseline),
                                 "nested_cost_curve": {str(floor): metric(training["labels"][scope], nested[track][str(floor)][scope]) for floor in FLOORS},
                                 "nested_probability_before": probability_metrics(training["labels"][scope], probabilities(outer_raw["scores"][scope], {"face": 1., "fallback": 1.}, outer_raw["face"][scope])),
                                 "nested_probability_after": probability_metrics(training["labels"][scope], nested_p[track][scope])}
        np.savez_compressed(output / f"oof_{track}.npz", **outer_raw, labels=training["labels"], sample_ids=training["sample_ids"],
                            source_groups=training["source_groups"], nested_probabilities=nested_p[track],
                            **{f"nested_pred_{floor}": nested[track][str(floor)] for floor in FLOORS})
    final_model = fit_pipeline(training, auxiliary)
    with (previous / "face_search/selected.pkl").open("rb") as handle:
        original = pickle.load(handle)
    if not np.allclose(original["model"].coef_, final_model["primary"].coef_, atol=1e-5):
        raise ValueError("最终主头与第五轮不同，不能声明仅改变主分支决策")
    with (output / "model.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)
    save(output / "selection_locked.json", {"policies": final_policies, "nested_oof": nested_metrics, "test_not_loaded_this_run": True,
                                            "no_703_used": True, "model_sha256": sha(output / "model.pkl")})
    # 规则、校准器和模型至此全部锁定；以下仅一次固定评测，不扫描测试阈值。
    test = load(previous / "face_features/test.npz")
    saved_old = load(previous / "availability_router/historical_predictions.npz")
    if not np.array_equal(saved_old["sample_ids"], test["sample_ids"]) or not np.array_equal(saved_old["labels"], test["labels"]):
        raise ValueError("历史821身份不符")
    old_pred = saved_old["scores"].argmax(1)
    raw = raw_predict(final_model, test)
    if not raw["valid"].all():
        raise ValueError("历史样本缺必要文字，必须另报覆盖率")
    baseline = raw["scores"].argmax(1)
    evaluations: dict[str, Any] = {"round5": metric(test["labels"], old_pred), "cpu_argmax": metric(test["labels"], baseline)}
    for track in ["A", "B"]:
        p = probabilities(raw["scores"], final_policies[track]["temperatures"], raw["face"])
        pred = decide(p, final_policies[track]["main_rule"])
        if track == "A":
            pred[~raw["face"]] = old_pred[~raw["face"]]
            # A未校准旧神经回退，不保存来自B回退的概率冒充A的观测。
            p[~raw["face"]] = np.nan
        score = metric(test["labels"], pred)
        evaluations[track] = {"historical": score, "delta_vs_round5_ci95": paired_ci(test["labels"], pred, old_pred, test["source_groups"]),
                              "delta_vs_cpu_argmax_ci95": paired_ci(test["labels"], pred, baseline, test["source_groups"]),
                              "checks": {"macro_f1_ge_066": score["macro_f1"] >= .66, "negative_recall_ge_085": score["per_class_recall"][0] >= .85,
                                         "negative_recall_not_lower_than_round5": score["per_class_recall"][0] >= evaluations["round5"]["per_class_recall"][0]}}
        saved_scores = raw["scores"].copy()
        if track == "A":
            saved_scores[~raw["face"]] = saved_old["scores"][~raw["face"]]
        np.savez_compressed(output / f"historical_{track}.npz", scores=saved_scores, probabilities=p, predictions=pred, baseline_predictions=baseline,
                            round5_predictions=old_pred, labels=test["labels"], sample_ids=test["sample_ids"], source_groups=test["source_groups"], face=raw["face"])
    # 既有ECISA测试已在评审中看过，只作固定规则迁移和回归。
    source_path = previous / "router_confirmation/holdout.jsonl"
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
    external = text_only(rows)
    exraw = raw_predict(final_model, external)
    old_external = load(previous / "router_confirmation/predictions.npz")
    old_values = old_external["scores"]
    if not np.array_equal(old_external["sample_ids"], external["sample_ids"]) or not np.array_equal(old_external["labels"], external["labels"]):
        raise ValueError("外部样本ID错位")
    ex_old_pred = old_values.argmax(1)
    ex_p = probabilities(exraw["scores"], final_policies["B"]["temperatures"], exraw["face"])
    ex_pred = decide(ex_p, final_policies["B"]["main_rule"])
    external_metrics = {"round5": metric(external["labels"], ex_old_pred), "A": metric(external["labels"], ex_old_pred),
                        "cpu_argmax": metric(external["labels"], exraw["scores"].argmax(1)), "B": metric(external["labels"], ex_pred)}
    np.savez_compressed(output / "external.npz", scores=exraw["scores"], probabilities=ex_p, predictions=ex_pred,
                        round5_predictions=ex_old_pred, labels=external["labels"], sample_ids=external["sample_ids"], source_groups=external["source_groups"])
    for path, identity in immutable.items():
        if sha(Path(path)) != identity:
            raise ValueError("旧轮次关键工件被改变")
    report = {"protocol": protocol, "policies": final_policies, "nested_oof": nested_metrics, "historical": evaluations,
              "external_ecisa": external_metrics, "seconds": time.monotonic() - started, "deployed": False,
              "old_critical_artifacts_unchanged": True,
              "limitations": ["all historical/external tests already examined; not fresh independent confirmation", "OOF architecture inherited from prior703 selection",
                              "A primary-only conditional OOF cannot validate the unchanged supervised fallback", "B changes fallback; compare decision gains to same-pipeline argmax",
                              "negative sentiment recall is not clinical risk sensitivity", "no human adjudication or clinical efficacy evidence"]}
    save(output / "results.json", report)
    print(json.dumps({"historical": evaluations, "external": external_metrics, "seconds": report["seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.previous, args.output)
