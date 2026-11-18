"""从冻结源代码重训跨日小模型，在独立输出目录比对全部测试预测。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.round14_common import OUT, save, sha256
from experiments.round14_runtime import verify_artifacts


def reproduce(destination: Path) -> dict[str, Any]:
    """原测试结果只读；新文件不覆盖预注册、选型锁或原始预测。"""
    if destination.resolve() == OUT.resolve() or OUT.resolve() in destination.resolve().parents:
        raise ValueError("复算目录应独立于正式证据目录")
    if destination.exists() and any(destination.iterdir()): raise ValueError("复算目录须为空，防止混入历史结果")
    destination.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, snapshot in (("studentlife_calendar_days", "current"), ("adolescent_sampling_days_reused", "adolescent_locked")):
        original = OUT / "longitudinal" / name
        protocol = json.loads((original / "preregister.json").read_text(encoding="utf-8"))
        source = OUT / "source_locked" / snapshot / "round14_longitudinal.py"
        if sha256(source) != protocol["code_sha256"]: raise ValueError("冻结源代码摘要不一致")
        spec = importlib.util.spec_from_file_location("locked_" + name, source)
        if spec is None or spec.loader is None: raise ValueError("无法加载冻结代码")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.OUT = destination
        frame = pd.read_csv(original / "daily_observations.csv", dtype={"person": str}, float_precision="round_trip")
        module.run_dataset(name, frame, protocol["source"], protocol["scale"], protocol["meaningful_change_normalized"])
        comparisons = {}
        for mode in ("same_person_future", "unseen_person_future"):
            expected = np.load(original / f"{mode}_test_predictions.npz", allow_pickle=False)
            actual = np.load(destination / "longitudinal" / name / f"{mode}_test_predictions.npz", allow_pickle=False)
            if set(expected.files) != set(actual.files): raise ValueError("复算预测字段不一致")
            errors = []
            for key in expected.files:
                if expected[key].dtype.kind in "US":
                    if not np.array_equal(expected[key], actual[key]): raise ValueError("测试参与者改变")
                else:
                    error = float(np.max(np.abs(expected[key] - actual[key])))
                    if error > 1e-12: raise ValueError(f"预测不一致: {name}/{mode}/{key}")
                    errors.append(error)
            old_lock = json.loads((original / f"{mode}_selection_locked.json").read_text(encoding="utf-8"))
            new_lock = json.loads((destination / "longitudinal" / name / f"{mode}_selection_locked.json").read_text(encoding="utf-8"))
            if old_lock != new_lock: raise ValueError("选型锁改变")
            comparisons[mode] = {"prediction_arrays": len(expected.files), "n": len(expected["truth"]), "max_error": max(errors)}
        report[name] = comparisons
    result = {"training_reproduced": report, "runtime_replay": verify_artifacts(), "all_passed": True}
    save(destination / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(reproduce(parser.parse_args().output), ensure_ascii=False))
