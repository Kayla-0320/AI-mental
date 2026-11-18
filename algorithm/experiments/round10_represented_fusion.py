"""目标标签无关的新文字/表情表征，来源折内降维和融合选型。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments.round10_fusion import OUTPUT, load_data, load_npz, folds, take, save, make_model, select_bias
from experiments.round7_decision_oof import metric

PACKS = {
    "aux_textface": ("aux_mean", "bge_cls", "face_summary"),
    "aux_face": ("aux_mean", "face_summary"),
    "aux_full": ("aux_mean", "bge_cls", "face_summary", "visual", "audio"),
    "deep_frozen": ("roberta_mean", "bge_cls", "deep_face"),
    "deep_aux": ("aux_mean", "bge_cls", "deep_face"),
    "deep_combined": ("aux_mean", "bge_cls", "deep_face", "face_summary"),
}


def data_for(splits: tuple[str, ...], deep: bool) -> dict[str, np.ndarray]:
    """新表征必须逐行ID对齐，不能拿监督微调的整训练缓存伪装OOF。"""
    data = load_data(splits)
    for kind in (["auxiliary", "deep_face"] if deep else ["auxiliary"]):
        parts = [load_npz(OUTPUT / f"{kind}_features" / f"{split}.npz") for split in splits]
        ids = np.concatenate([part["sample_ids"] for part in parts])
        if not np.array_equal(ids, data["sample_ids"]):
            raise ValueError("新表征ID未对齐")
        names = ("aux_mean", "aux_cls") if kind == "auxiliary" else ("deep_face", "coverage")
        for name in names:
            data[name] = np.concatenate([part[name] for part in parts])
    return data


def transform(data: dict[str, np.ndarray], config: dict[str, Any],
              scalers: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """标准化和PCA都只见当前训练折有效观测。"""
    fitting = scalers is None
    scalers = {} if fitting else scalers
    blocks = []
    for name in PACKS[config["pack"]]:
        if name == "deep_face":
            mask = data["coverage"] > 0
        elif name == "face_summary":
            mask = data["face_summary"][:, 24] > 0
        elif name in ("visual", "audio", "speech", "speech_emotion"):
            j = 1 if name.startswith("speech") else ("text", "audio", "visual").index(name)
            mask = data["modality_mask"][:, j] & (data["quality"][:, j] > 0)
            if name.startswith("speech"):
                mask &= data["speech_available"]
        else:
            mask = np.ones(len(data["labels"]), bool)
        values = data[name]
        if fitting:
            scaler = StandardScaler().fit(values[mask])
            pca = None
            if name == "deep_face":
                pca = PCA(n_components=config["components"], whiten=True, svd_solver="randomized", random_state=42).fit(scaler.transform(values[mask]))
            scalers[name] = {"scaler": scaler, "pca": pca}
        scaler, pca = scalers[name]["scaler"], scalers[name]["pca"]
        block = scaler.transform(np.where(mask[:, None], values, scaler.mean_))
        if pca is not None:
            block = pca.transform(block)
        block = np.where(mask[:, None], block, 0.) / np.sqrt(block.shape[1])
        if name in ("face_summary", "deep_face"):
            block *= config["face_weight"]
        blocks.extend([block, mask[:, None].astype(float) * .1])
    return np.column_stack(blocks), scalers


def candidates(deep: bool) -> list[dict[str, Any]]:
    """附加表征阶段只按开发OOF扩展，不读取历史测试分数。"""
    result = []
    for pack in PACKS:
        if pack.startswith("deep_") != deep:
            continue
        for components in ((32, 128) if deep else (0,)):
            for fw, c, weight in itertools.product((.5, 1., 2.), (.03, .1, 1.), (None, "balanced")):
                result.append({"id": f"{pack}_{components}_{fw}_{c}_{weight}", "pack": pack,
                    "components": components, "face_weight": fw, "kind": "lr", "C": c, "class_weight": weight})
    return result


def search(output: Path, deep: bool) -> None:
    """来源OOF开发；模型和预处理选择都在历史821解封前完成。"""
    if (output / "search_complete.json").exists():
        raise ValueError("拒绝覆盖搜索")
    data = data_for(("train", "selection"), deep)
    options = candidates(deep)
    save(output / "preregister.json", {"configs": options, "source": "target-independent auxiliary encoder and frozen face backbone",
         "selection": "source-group 4-fold OOF, no calibration703 or test821", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    predictions = np.full((len(options), len(data["labels"]), 3), np.nan)
    cache, result = {}, []
    start = time.monotonic()
    for i, config in enumerate(options):
        for fold, (tr, va) in enumerate(folds(data)):
            key = (config["pack"], config["components"], config["face_weight"], fold)
            if key not in cache:
                x, scalers = transform(take(data, tr), config)
                v, _ = transform(take(data, va), config, scalers)
                cache[key] = (x, v)
            x, v = cache[key]
            model = make_model(config).fit(x, data["labels"][tr])
            predictions[i, va] = model.decision_function(v)
        result.append({"config": config, "argmax": metric(data["labels"], predictions[i].argmax(1)), **select_bias(data["labels"], predictions[i])})
        if (i+1) % 6 == 0:
            best = max(result, key=lambda row: row["metrics"]["macro_f1"])
            print(json.dumps({"candidates": i+1, "total": len(options), "best_oof": best["metrics"]["macro_f1"],
                              "winner": best["config"]["id"], "seconds": round(time.monotonic()-start, 1)}), flush=True)
            save(output / "progress.json", result)
    np.savez_compressed(output / "candidates_oof.npz", logits=predictions, labels=data["labels"], sample_ids=data["sample_ids"], source_groups=data["source_groups"])
    save(output / "search_complete.json", {"results": result})


def main() -> None:
    """分离辅助文字和深层人脸两个开发阶段。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    search(OUTPUT / ("deep_fusion" if args.deep else "aux_fusion"), args.deep)


if __name__ == "__main__":
    main()
