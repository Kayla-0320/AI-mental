"""第十轮来源隔离的表征融合、分类决策和复现工件。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.round7_decision_oof import metric, probability_metrics

ROOT = Path(r"D:\Develop\AIC\_国赛优化")
OUTPUT = Path(os.environ.get("ROUND10_OUTPUT", str(ROOT / "round10")))
PACKS = {
    "textface": ("roberta_mean", "bge_cls", "face_summary"),
    "expression": ("roberta_mean", "bge_cls", "face_expression"),
    "full": ("roberta_mean", "bge_cls", "face_summary", "visual", "audio"),
    "robface": ("roberta_mean", "face_summary"),
    "bgeface": ("bge_cls", "face_summary"),
    "textonly": ("roberta_mean", "bge_cls"),
    "faceonly": ("face_expression",),
    "metadata": ("roberta_mean", "bge_cls", "face_metadata"),
}


def save(path: Path, value: Any) -> None:
    """原子写入UTF-8结构化工件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """读取无对象数组。"""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def load_data(splits: tuple[str, ...] = ("train", "selection")) -> dict[str, np.ndarray]:
    """按ID核对冻结文字及人脸表征，只加载指定划分。"""
    parts = []
    for split in splits:
        part = load_npz(ROOT / "round5" / "face_features" / f"{split}.npz")
        source = load_npz(ROOT / "round5" / "text_features" / f"{split}.npz")
        if not np.array_equal(part["sample_ids"], source["sample_ids"]):
            raise ValueError("表征ID未对齐")
        part["face_expression"] = part["face_summary"][:, :24]
        part["face_metadata"] = part["face_summary"][:, 24:]
        parts.append(part)
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def take(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    """按相同索引切片全部字段。"""
    return {name: values[indices] for name, values in data.items()}


def observed(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    """真实缺失人脸不可作为全零观测参与归一化。"""
    if name.startswith("face_"):
        return data["face_summary"][:, 24] > 0
    if name in ("audio", "visual"):
        j = ("text", "audio", "visual").index(name)
        return data["modality_mask"][:, j] & (data["quality"][:, j] > 0)
    return np.ones(len(data["labels"]), dtype=bool)


def features(data: dict[str, np.ndarray], config: dict[str, Any],
             scalers: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """仅用训练折有效观测拟合统计，各分支按维数平衡。"""
    fitting = scalers is None
    scalers = {} if fitting else scalers
    blocks = []
    for name in PACKS[config["pack"]]:
        mask = observed(data, name)
        values = data[name].astype(np.float32)
        if not np.isfinite(values[mask]).all():
            raise ValueError(f"{name} 有效观测含非有限值")
        if fitting:
            scalers[name] = StandardScaler().fit(values[mask])
        clean = np.where(mask[:, None], values, scalers[name].mean_)
        block = scalers[name].transform(clean) / np.sqrt(values.shape[1])
        if name.startswith("face_"):
            block *= config["face_weight"]
        blocks.extend([block, mask[:, None].astype(np.float32) * .1])
    return np.column_stack(blocks).astype(np.float64), scalers


def make_model(config: dict[str, Any]) -> Any:
    """固定模型族；只由开发侧选择超参数。"""
    if config["kind"] == "lr":
        return LogisticRegression(C=config["C"], class_weight=config["class_weight"],
                                  max_iter=500, random_state=42)
    return SVC(C=config["C"], class_weight=config["class_weight"], kernel="rbf",
               gamma="scale", random_state=42)


def score(prediction: np.ndarray, bias: list[float]) -> np.ndarray:
    """应用来源OOF锁定的统一类别偏置，不按测试来源调参。"""
    return prediction + np.asarray(bias)


def select_bias(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    """只用开发OOF标签选择小型全局偏置网格。"""
    candidates = []
    for negative, neutral in itertools.product((-.25, 0., .25), (-.5, -.25, 0., .25, .5)):
        bias = [negative, neutral, 0.]
        result = metric(labels, score(logits, bias).argmax(1))
        candidates.append({"bias": bias, "metrics": result})
    return max(candidates, key=lambda row: (row["metrics"]["macro_f1"],
                                           row["metrics"]["accuracy"],
                                           -sum(abs(v) for v in row["bias"])))


def configs() -> list[dict[str, Any]]:
    """在首次训练前公开候选，控制组不与主候选混报。"""
    result = []
    for pack in PACKS:
        controls = pack in ("textonly", "faceonly", "metadata")
        for fw in ((1.,) if controls else (.5, 1., 2.)):
            for kind, c in (("lr", .03), ("lr", .1), ("lr", 1.), ("svm", 1.), ("svm", 10.)):
                for cw in (None, "balanced"):
                    result.append({"id": f"{pack}_{fw}_{kind}_{c}_{cw}", "pack": pack,
                                   "face_weight": fw, "kind": kind, "C": c,
                                   "class_weight": cw, "control": controls})
    return result


def folds(data: dict[str, np.ndarray], n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """整来源视频不可跨训练和留出；不把同一视频窗口随机拆分。"""
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)
    result = list(splitter.split(data["labels"], data["labels"], data["source_groups"]))
    for train, valid in result:
        if set(data["source_groups"][train]) & set(data["source_groups"][valid]):
            raise ValueError("来源跨折")
    return result


def search(output: Path) -> None:
    """开发阶段不读取703或821的标签、特征或预测。"""
    if (output / "search_complete.json").exists():
        raise ValueError("拒绝覆盖完成的搜索")
    data = load_data()
    options = configs()
    plan = folds(data)
    assignment = np.full(len(data["labels"]), -1)
    for fold, (_, valid) in enumerate(plan):
        assignment[valid] = fold
    save(output / "preregister.json", {"baseline_macro_f1": .6388, "target": .6388 * 1.1,
         "selection": "4-fold source-group OOF on original train+selection; 703 and 821 excluded",
         "training_n": len(assignment), "training_groups": len(np.unique(data["source_groups"])),
         "configs": options, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "metric": "macro-F1; no additional negative-sentiment recall constraint",
         "limitations": "Selected OOF is a development score, not an unbiased generalization estimate"})
    output.mkdir(parents=True, exist_ok=True)
    predictions = np.full((len(options), len(assignment), 3), np.nan)
    results = []
    start = time.monotonic()
    # 同pack/权重下缓存折内标准化，避免重复拟合与数据泄漏。
    cache: dict[tuple, Any] = {}
    for index, config in enumerate(options):
        for fold, (train, valid) in enumerate(plan):
            key = (config["pack"], config["face_weight"], fold)
            if key not in cache:
                x, scalers = features(take(data, train), config)
                v, _ = features(take(data, valid), config, scalers)
                cache[key] = x, v
            x, v = cache[key]
            model = make_model(config).fit(x, data["labels"][train])
            predictions[index, valid] = model.decision_function(v)
        chosen = select_bias(data["labels"], predictions[index])
        raw = metric(data["labels"], predictions[index].argmax(1))
        results.append({"config": config, "argmax": raw, **chosen})
        if (index + 1) % 10 == 0:
            best = max(results, key=lambda row: row["metrics"]["macro_f1"])
            print(json.dumps({"candidates": index + 1, "total": len(options),
                  "best_oof": best["metrics"]["macro_f1"], "winner": best["config"]["id"],
                  "seconds": round(time.monotonic() - start, 1)}), flush=True)
            save(output / "progress.json", results)
    np.savez_compressed(output / "candidates_oof.npz", logits=predictions, labels=data["labels"],
                        sample_ids=data["sample_ids"], source_groups=data["source_groups"], fold=assignment)
    save(output / "search_complete.json", {"results": results, "elapsed_seconds": time.monotonic() - start})


def fit_final(output: Path, winner: dict[str, Any]) -> None:
    """先保存锁定选择，再拟合全开发集；本函数不加载测试集。"""
    save(output / "selection_locked.json", winner)
    data = load_data()
    x, scalers = features(data, winner["config"])
    model = make_model(winner["config"]).fit(x, data["labels"])
    with (output / "selected.pkl").open("wb") as handle:
        pickle.dump({"model": model, "scalers": scalers, "selection": winner}, handle)


def main() -> None:
    """搜索入口；历史评测由独立锁定阶段完成。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT / "linear")
    args = parser.parse_args()
    search(args.output)


if __name__ == "__main__":
    main()
