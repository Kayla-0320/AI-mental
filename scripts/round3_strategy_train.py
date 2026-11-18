"""ESConv 历史上下文→下一支持策略的对话隔离实验，不做疗效推断。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm"))

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

from intervention.strategy_learning import context_text


def extract(dialogues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """逐轮截取过去信息；组ID按完整对话摘要确定，重复对话不会跨分割。"""
    examples = []
    for dialogue in dialogues:
        turns = dialogue["dialog"]
        group = hashlib.sha256(json.dumps(turns, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        history: list[dict[str, str]] = []
        for turn in turns:
            strategy = turn.get("annotation", {}).get("strategy")
            if turn["speaker"] == "supporter" and strategy and any(m["role"] == "user" for m in history):
                examples.append({"group": group, "context": context_text(history[-64:], True),
                                 "last_user": context_text(history[-64:], False), "label": strategy})
            history.append({"role": "user" if turn["speaker"] == "seeker" else "assistant", "content": turn["content"][:4000]})
    return examples


def probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """稳定softmax，温度仅由独立校准组拟合。"""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def metrics(y: np.ndarray, p: np.ndarray, classes: int) -> dict[str, float]:
    """固定完整类别集合报告描述性策略任务指标。"""
    confidence, pred = p.max(axis=1), p.argmax(axis=1)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= lo) & (confidence <= hi if hi == 1 else confidence < hi)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - (pred[mask] == y[mask]).mean())
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=list(range(classes)), average="macro", zero_division=0)),
            "nll": float(log_loss(y, p, labels=list(range(classes)))), "ece": float(ece)}


def run(source: Path, output: Path, artifact: Path) -> None:
    """在选择组比较两个上下文基线，校准组定温度，冻结后评测试组。"""
    raw = source.read_bytes()
    examples = extract(json.loads(raw))
    groups = sorted({r["group"] for r in examples})
    order = np.random.default_rng(42).permutation(groups)
    n = len(order)
    partitions = dict(zip(("train", "selection", "calibration", "test"),
                          np.split(order, [int(n * .65), int(n * .75), int(n * .85)])))
    labels = sorted({r["label"] for r in examples})
    label_ids = {name: i for i, name in enumerate(labels)}
    data = {split: [r for r in examples if r["group"] in set(ids)] for split, ids in partitions.items()}
    ys = {split: np.array([label_ids[r["label"]] for r in rows]) for split, rows in data.items()}
    if set(ys["train"]) != set(range(len(labels))):
        raise ValueError("训练集类别不完整")
    arms = {}
    options = []
    output.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    for key in ("last_user", "context"):
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=18000,
                                     min_df=3, sublinear_tf=True)
        x = {"train": vectorizer.fit_transform([r[key] for r in data["train"]])}
        x.update({s: vectorizer.transform([r[key] for r in rows]) for s, rows in data.items() if s != "train"})
        candidates = []
        for c in (.1, 1.0, 10.0):
            model = LogisticRegression(C=c, max_iter=350, random_state=42).fit(x["train"], ys["train"])
            score = metrics(ys["selection"], model.predict_proba(x["selection"]), len(labels))["macro_f1"]
            candidates.append((score, c, model))
        score, c, model = max(candidates, key=lambda candidate: candidate[0])
        calibration_logits = model.decision_function(x["calibration"])
        fitted = minimize_scalar(lambda t: log_loss(ys["calibration"], probabilities(calibration_logits, t),
                                                   labels=list(range(len(labels)))), bounds=(.1, 10), method="bounded")
        temperature = float(fitted.x)
        test_logits = model.decision_function(x["test"])
        arms[key] = {"selection_macro_f1": score, "C": c, "temperature": temperature,
                     "test_raw": metrics(ys["test"], probabilities(test_logits), len(labels)),
                     "test_calibrated": metrics(ys["test"], probabilities(test_logits, temperature), len(labels))}
        options.append((score, key, vectorizer, model, temperature))
        np.savez_compressed(output / f"{key}_test.npz", labels=ys["test"],
                            probabilities=probabilities(test_logits, temperature),
                            groups=np.array([r["group"] for r in data["test"]]))
        print(key, json.dumps(arms[key]), flush=True)
    _, key, vectorizer, model, temperature = max(options, key=lambda option: option[0])
    path = artifact / "weights.npz"
    np.savez_compressed(path, coef=model.coef_, intercept=model.intercept_, idf=vectorizer.idf_)
    metadata = {"schema": "esconv_strategy.v1", "language": "en", "labels": labels,
                "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
                "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(raw).hexdigest(),
                "use_context": key == "context", "selection_split": "selection", "calibration_split": "calibration",
                "temperature": temperature, "source": "https://github.com/thu-coai/Emotional-Support-Conversation",
                "scope": "descriptive_next_strategy_not_intervention_effect"}
    (artifact / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    majority = np.bincount(ys["train"], minlength=len(labels)).argmax()
    prior = (np.bincount(ys["train"], minlength=len(labels)) + 1) / (len(ys["train"]) + len(labels))
    report = {"seed": 42, "source_sha256": metadata["source_sha256"], "dialogues": len(json.loads(raw)),
              "unique_dialogue_groups": n, "examples": len(examples), "labels": labels,
              "partition_groups": {s: len(ids) for s, ids in partitions.items()},
              "partition_examples": {s: len(rows) for s, rows in data.items()},
              "class_counts": dict(Counter(r["label"] for r in examples)), "arms": arms,
              "selected_using_selection_only": key,
              "train_prior_baseline": metrics(ys["test"], np.tile(prior, (len(ys["test"]), 1)), len(labels)),
              "limitations": ["English-only", "strategy labels describe behavior not causal benefit", "not Chinese target population validation",
                              "no current response, survey scores or final outcomes used as features"]}
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=ROOT / "algorithm/artifacts/round3_strategy")
    a = parser.parse_args()
    run(a.source, a.output, a.artifact)
