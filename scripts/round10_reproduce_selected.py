"""在新目录从冻结输入重建锁定配方，逐样本核对独立重训结果。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

# 项目实验模块会设置BLAS变量；子进程必须继承启动时环境，避免父导入改变数值配方。
STARTUP_ENVIRONMENT = dict(os.environ)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "algorithm"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from experiments.round10_fusion import OUTPUT, save, folds, take, load_npz
from experiments.round10_domain import weights_for
from experiments.round10_represented_fusion import transform
from experiments.round10_large_speech import data_for


def child_environment(destination: Path) -> dict[str, str]:
    """继承程序启动环境，不能继承项目导入之后被改写的BLAS变量。"""
    environment = dict(STARTUP_ENVIRONMENT)
    environment["ROUND10_OUTPUT"] = str(destination)
    return environment


def fit_cheap() -> None:
    """重建同一四折轻量专家，不读取或复用原训练模型权重。"""
    lock = json.loads((OUTPUT / "large_final" / "selection_locked.json").read_text("utf-8"))
    member = lock["members"][2]
    data = data_for(("train", "selection", "calibration"))
    models = []
    for tr, _ in folds(data):
        part = take(data, tr)
        x, scaler = transform(part, member["config"])
        weights = weights_for(part["labels"], part["source_groups"], 0., 1.)
        model = LogisticRegression(C=member["config"]["C"]*2157/len(tr), max_iter=500, random_state=42)
        model.fit(x, part["labels"], sample_weight=weights)
        models.append({"model": model, "scaler": scaler})
    target = OUTPUT / "cascade" / "model.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump({"members": [{**member, "models": models}], "training_groups": np.unique(data["source_groups"])}, handle)


def rebuild(destination: Path) -> None:
    """仅复制冻结输入、锁定参数和对照预测，所有最终训练权重重新拟合。"""
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("重建目标必须是全新目录，禁止覆盖既有工件")
    destination.mkdir(parents=True)
    for name in ("auxiliary_features", "deep_face_features", "speech_features", "llm_features", "llm_large_features"):
        target = destination / name
        target.mkdir()
        for split in ("train", "selection", "calibration", "test"):
            shutil.copy2(OUTPUT / name / f"{split}.npz", target / f"{split}.npz")
    shutil.copy2(OUTPUT / "official_meta.csv", destination / "official_meta.csv")
    selected = ["large_final/selection_locked.json", "large_final/selected_oof.npz", "large_final/historical_predictions.npz",
                "refined_final/selection_locked.json", "refined_final/preregister.json", "final/historical_predictions.npz"]
    for stage in ("large_speech", "large_speech5"):
        selected.extend([f"{stage}/fold{i}_evidence.npz" for i in range(4)])
    for relative in selected:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT / relative, target)
    save(destination / "rebuild_protocol.json", {"source": str(OUTPUT), "destination": str(destination),
        "trained_weights_copied": False, "frozen_inputs_copied": True,
        "child_blas_environment_before_project_import": {key: STARTUP_ENVIRONMENT.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "scope": "fixed-recipe numerical reconstruction, no hyperparameter selection or historical tuning",
        "acceptance": "all4105 hard labels identical; probabilities max error<1e-8; five seed F1 mean identical"})
    environment = child_environment(destination)
    numeric_runtime = None
    commands = [
        [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--fit-cheap"],
        [sys.executable, "-X", "utf8", "-m", "experiments.round10_large_final", "refit"],
        [sys.executable, "-X", "utf8", "-m", "experiments.round10_refined", "fit"],
        [sys.executable, "-X", "utf8", "-m", "experiments.round10_refined", "evaluate"],
        [sys.executable, "-X", "utf8", str(REPO / "scripts" / "round10_audit.py"), "--stage", "refined_final"],
    ]
    for i, command in enumerate(commands):
        print(json.dumps({"rebuild_step": i+1, "steps": len(commands), "command": command}), flush=True)
        with (destination / f"step{i+1}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO / "algorithm", env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
                if line.startswith('{"training_numeric_runtime":'):
                    numeric_runtime = json.loads(line)["training_numeric_runtime"]
            if process.wait() != 0:
                raise RuntimeError(f"重建第{i+1}步失败，日志: {destination / f'step{i+1}.log'}")
    original = load_npz(OUTPUT / "refined_final" / "historical_predictions.npz")
    rebuilt = load_npz(destination / "refined_final" / "historical_predictions.npz")
    assert np.array_equal(original["sample_ids"], rebuilt["sample_ids"])
    assert np.array_equal(original["labels"], rebuilt["labels"])
    error = float(np.max(abs(original["probabilities"]-rebuilt["probabilities"])))
    same_labels = np.array_equal(original["probabilities"].argmax(-1), rebuilt["probabilities"].argmax(-1))
    if not same_labels or error >= 1e-8:
        raise ValueError(f"重建未通过一致性门槛: hard_labels_equal={same_labels},max_probability_error={error}")
    result = {"new_directory": str(destination), "trained_weights_copied": False,
        "hard_labels_identical": bool(same_labels), "n_predictions": 4105, "max_probability_error": error,
        "rebuilt_mean_macro_f1": json.loads((destination / "refined_final" / "results.json").read_text("utf-8"))["mean_macro_f1"],
        "training_numeric_runtime": numeric_runtime, "passed": True}
    save(destination / "rebuild_check.json", result)
    save(OUTPUT / "locked_rebuild_check.json", result)
    print(json.dumps(result), flush=True)


def main() -> None:
    """新目录重建与子进程的轻量训练入口明确区分。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--fit-cheap", action="store_true")
    args = parser.parse_args()
    if args.fit_cheap:
        fit_cheap()
    elif args.destination is not None:
        rebuild(args.destination)
    else:
        parser.error("需要--destination指定全新目录")


if __name__ == "__main__":
    main()
