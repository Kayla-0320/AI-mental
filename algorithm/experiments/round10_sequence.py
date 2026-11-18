"""真实EmotionTalk会话的因果状态序列预测；不冒充跨天纵向或原始媒体预测。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from experiments.round10_fusion import ROOT, OUTPUT, save

CLASSES = ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised")
ALIASES = {"anger": "angry", "disgust": "disgusted", "fear": "fearful", "surprise": "surprised",
           "happiness": "happy", "sadness": "sad"}


def label(value: str) -> int:
    """只统一同义拼写，不把情绪类别改造成极性标签。"""
    return CLASSES.index(ALIASES.get(value.lower(), value.lower()))


def load_sequences() -> dict[str, list[dict[str, Any]]]:
    """按官方对话、说话人分段；末段编号是待核验的相对话轮顺序。"""
    directory = ROOT / "round6" / "data"
    sources = {}
    for name in ("text", "video", "multimodal"):
        rows = [json.loads(line) for line in (directory / f"{name}_annotations.jsonl").read_text("utf-8").splitlines() if line]
        sources[name] = {row["id"]: row["annotation"] for row in rows}
    if not all(set(rows) == set(sources["text"]) for rows in sources.values()):
        raise ValueError("三种标注的ID不一致")
    result = defaultdict(list)
    for identifier, row in sources["text"].items():
        pieces = identifier.split("_")
        conversation = "_".join(pieces[:2])
        speaker = row["speaker_id"]
        if speaker != pieces[-2]:
            raise ValueError("说话人ID与路径不一致")
        values = [label(sources[name][identifier]["emotion_result"]) for name in sources]
        result[f"{conversation}_{speaker}"].append({"id": identifier, "turn": int(pieces[-1]),
            "group": pieces[0], "speaker": speaker, "labels": values})
    for records in result.values():
        records.sort(key=lambda row: row["turn"])
        turns = [row["turn"] for row in records]
        if len(set(turns)) != len(turns):
            raise ValueError("同一会话内话轮编号重复")
    return dict(result)


def windows(sequences: dict[str, list[dict[str, Any]]], context: int, horizon: int) -> dict[str, np.ndarray]:
    """目标前至少有5次已观测状态，所有输入编号严格早于目标编号。"""
    features, targets, groups, ids, previous, cutoff = [], [], [], [], [], []
    for rows in sequences.values():
        for t in range(5, len(rows) - horizon + 1):
            history = rows[max(0, t-context):t]
            target = rows[t+horizon-1]
            if not all(row["turn"] < target["turn"] for row in history):
                raise ValueError("未来泄漏")
            x = np.zeros((context, 3, 7), np.float32)
            for j, row in enumerate(history, context-len(history)):
                x[j, np.arange(3), row["labels"]] = 1
            # 同时给短期细节与历史频率，不输入来源ID或目标话轮文本。
            features.append(np.concatenate([x.reshape(-1), x.mean(0).reshape(-1)]))
            targets.append(target["labels"][-1])
            previous.append(history[-1]["labels"][-1])
            groups.append(target["group"])
            ids.append(target["id"])
            cutoff.append(history[-1]["turn"])
    return {"x": np.stack(features), "labels": np.array(targets), "groups": np.array(groups),
            "sample_ids": np.array(ids), "previous": np.array(previous), "cutoff_turn": np.array(cutoff)}


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """七类情绪单独报告，不与CH-SIMS三类macro-F1混算。"""
    return {"n": len(y), "macro_f1": float(f1_score(y, pred, labels=np.arange(7), average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)), "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(7)).tolist()}


def run(output: Path) -> None:
    """固定官方组留出，开发锁定后一次评测下一话轮及多步跨度。"""
    if (output / "results.json").exists():
        raise ValueError("拒绝覆盖序列评测")
    sequences = load_sequences()
    validation_groups = {"G00001", "G00012"}
    test_groups = {"G00003", "G00015"}
    options = [{"context": context, "C": c, "class_weight": weight}
               for context in (5, 10, 20) for c in (.01, .1, 1.) for weight in (None, "balanced")]
    save(output / "preregister.json", {"task": "next same-speaker utterance multimodal-emotion label from past annotated states",
         "classes": CLASSES, "candidates": options, "horizons": [1, 3, 5],
         "validation_groups": sorted(validation_groups), "test_groups": sorted(test_groups),
         "ordering": "numeric utterance filename index within official dialogue and speaker; no absolute timestamps",
         "limitations": ["past annotation states are oracle observations, not outputs of a deployed recognizer",
                         "utterance ID order is inferred; requires timestamp/media confirmation before temporal claims",
                         "short acted conversations, not cross-day psychological longitudinal data"],
         "sequence_count": len(sequences), "length_quantiles": np.quantile([len(r) for r in sequences.values()], [0, .5, .9, 1]).tolist()})
    report = {"protocol": "conditional on filename utterance ordering and past annotated states", "horizons": {}}
    for horizon in (1, 3, 5):
        candidates, cached = [], {}
        for config in options:
            context = config["context"]
            if context not in cached:
                cached[context] = windows(sequences, context, horizon)
            data = cached[context]
            va = np.isin(data["groups"], list(validation_groups))
            te = np.isin(data["groups"], list(test_groups))
            tr = ~va & ~te
            model = LogisticRegression(C=config["C"], class_weight=config["class_weight"],
                                       max_iter=500, random_state=42).fit(data["x"][tr], data["labels"][tr])
            validation = metrics(data["labels"][va], model.predict(data["x"][va]))
            candidates.append({"config": config, "validation": validation, "model": model})
        winner = max(candidates, key=lambda row: (row["validation"]["macro_f1"], row["validation"]["accuracy"]))
        save(output / f"horizon_{horizon}_selection_locked.json", {"winner": winner["config"],
             "development": [{k: v for k, v in row.items() if k != "model"} for row in candidates]})
        data = cached[winner["config"]["context"]]
        te = np.isin(data["groups"], list(test_groups))
        va = np.isin(data["groups"], list(validation_groups))
        tr = ~va & ~te
        prediction = winner["model"].predict(data["x"][te])
        report["horizons"][str(horizon)] = {"selected": winner["config"], "model": metrics(data["labels"][te], prediction),
            "persistence": metrics(data["labels"][te], data["previous"][te]),
            "train_n": int(tr.sum()), "validation_n": int(va.sum()), "test_n": int(te.sum())}
        np.savez_compressed(output / f"horizon_{horizon}_predictions.npz", labels=data["labels"][te], pred=prediction,
                            persistence=data["previous"][te], sample_ids=data["sample_ids"][te], cutoff_turn=data["cutoff_turn"][te])
    save(output / "results.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def archive_selected_models(output: Path) -> None:
    """按既有锁定配置补存模型，重放既有预测，不重新选型或覆盖原成绩。"""
    targets = [output / f"horizon_{horizon}_model.pkl" for horizon in (1, 3, 5)]
    if any(path.exists() for path in targets):
        raise ValueError("拒绝覆盖已保存序列模型")
    protocol = json.loads((output / "preregister.json").read_text("utf-8"))
    sequences = load_sequences()
    checks = []
    for horizon, target in zip((1, 3, 5), targets):
        lock = json.loads((output / f"horizon_{horizon}_selection_locked.json").read_text("utf-8"))["winner"]
        data = windows(sequences, lock["context"], horizon)
        validation = np.isin(data["groups"], protocol["validation_groups"])
        testing = np.isin(data["groups"], protocol["test_groups"])
        training = ~validation & ~testing
        model = LogisticRegression(C=lock["C"], class_weight=lock["class_weight"], max_iter=500, random_state=42)
        model.fit(data["x"][training], data["labels"][training])
        with np.load(output / f"horizon_{horizon}_predictions.npz") as old:
            assert np.array_equal(old["sample_ids"], data["sample_ids"][testing])
            assert np.array_equal(old["labels"], data["labels"][testing])
            assert np.array_equal(old["pred"], model.predict(data["x"][testing]))
        with target.open("wb") as handle:
            pickle.dump({"model": model, "selected": lock, "horizon": horizon, "classes": CLASSES,
                         "training_groups": np.unique(data["groups"][training]),
                         "input_condition": "past annotation states, filename utterance ordering; not raw-media future inference"}, handle)
        with target.open("rb") as handle:
            replay = pickle.load(handle)
        assert np.array_equal(replay["model"].predict(data["x"][testing]), model.predict(data["x"][testing]))
        checks.append({"horizon": horizon, "prediction_count": int(testing.sum()), "stored_predictions_unchanged": True,
                       "model_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    save(output / "model_archive_check.json", {"checks": checks, "scope": "locked model reconstruction and serialization replay"})
    print(json.dumps(checks), flush=True)


def main() -> None:
    """执行条件性真实状态序列实验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT / "sequence")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    if args.save_models:
        archive_selected_models(args.output)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
