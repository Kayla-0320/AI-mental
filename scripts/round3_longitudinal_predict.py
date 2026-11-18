"""云南中国青少年两波数据的可验证部分：结构审计与"基线→随访"预测。

**构念边界**：公开记录没有给出逐列题目字典，因此这里把 `g1..g9` 与
`h1..h7` 只当作两组 1–4 级自评条目，分别求和为"条目组合计"，不命名为
PHQ-9、GAD-7 或任何具体量表。分析只回答"基线自评能否预测同一组条目在
随访时的合计"，不产出临床风险结论。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SEED = 42
G_ITEMS = [f"g{i}" for i in range(1, 10)]
H_ITEMS = [f"h{i}" for i in range(1, 8)]
BASELINE_SINGLE = ["age", "race"]


def load_rows(source: Path) -> tuple[list[str], list[tuple[Any, ...]]]:
    """读取唯一工作表，返回表头与原始行；保持原文件只读。"""
    import openpyxl

    workbook = openpyxl.load_workbook(source, read_only=True, data_only=True)
    try:
        names = workbook.sheetnames
        if names != ["Sheet1"]:
            raise ValueError(f"工作表结构与预期不符: {names}")
        rows = list(workbook["Sheet1"].iter_rows(values_only=True))
    finally:
        workbook.close()
    header = [str(value) if value is not None else "" for value in rows[0]]
    keep = [position for position, name in enumerate(header) if name]
    columns = [header[position] for position in keep]
    if len(set(columns)) != len(columns):
        raise ValueError("表头存在重名列")
    body = [tuple(row[position] for position in keep) for row in rows[1:]]
    return columns, body


def quarantine_duplicates(rows: list[tuple[Any, ...]], header: list[str]) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
    """同一(cohort, code)出现多次时隔离全部相关行，不保留任何一条。"""
    cohort, code = header.index("cohort"), header.index("code")
    seen = Counter((row[cohort], row[code]) for row in rows)
    duplicated = {key for key, count in seen.items() if count > 1}
    if any(key[1] in (None, "") for key in seen):
        raise ValueError("存在缺失参与者编码")
    kept = [row for row in rows if (row[cohort], row[code]) not in duplicated]
    dropped = [{"cohort": key[0], "code": key[1], "rows": seen[key]} for key in sorted(duplicated, key=lambda k: (str(k[0]), str(k[1])))]
    return kept, dropped


def _number(value: Any) -> float | None:
    """只接受 1–4 的整数条目；空白与其它编码显式视为缺失。"""
    if isinstance(value, bool) or value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    if not math.isfinite(number) or number != int(number) or not 1 <= number <= 4:
        return None
    return number


def build_frame(rows: list[tuple[Any, ...]], header: list[str]) -> dict[str, Any]:
    """只保留四个条目组在两波都完整的参与者；缺失不插补。"""
    index = {name: position for position, name in enumerate(header)}
    required = [f"{item}_{wave}" for wave in (1, 2) for item in G_ITEMS + H_ITEMS]
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"缺少条目列: {missing}")
    participants: list[str] = []
    wave1_g: list[float] = []
    wave1_h: list[float] = []
    wave2_g: list[float] = []
    wave2_h: list[float] = []
    age: list[float] = []
    race: list[float] = []
    followed: list[int] = []
    for row in rows:
        g1 = [_number(row[index[f"{item}_1"]]) for item in G_ITEMS]
        h1 = [_number(row[index[f"{item}_1"]]) for item in H_ITEMS]
        g2 = [_number(row[index[f"{item}_2"]]) for item in G_ITEMS]
        h2 = [_number(row[index[f"{item}_2"]]) for item in H_ITEMS]
        if None in g1 or None in h1 or None in g2 or None in h2:
            continue
        participants.append(f"{row[index['cohort']]}-{row[index['code']]}")
        wave1_g.append(sum(g1))
        wave1_h.append(sum(h1))
        wave2_g.append(sum(g2))
        wave2_h.append(sum(h2))
        age.append(float(row[index["age"]]) if isinstance(row[index["age"]], (int, float)) else float("nan"))
        race.append(float(row[index["race"]]) if isinstance(row[index["race"]], (int, float)) else float("nan"))
        followed.append(int(row[index["wave2_followup"]]))
    if len({*participants}) != len(participants):
        raise ValueError("参与者编码仍不唯一，禁止进入切分")
    frame = {
        "participants": participants,
        "wave1_g": np.asarray(wave1_g, dtype=np.float64),
        "wave1_h": np.asarray(wave1_h, dtype=np.float64),
        "wave2_g": np.asarray(wave2_g, dtype=np.float64),
        "wave2_h": np.asarray(wave2_h, dtype=np.float64),
        "age": np.asarray(age, dtype=np.float64),
        "race": np.asarray(race, dtype=np.float64),
        "wave2_followup": np.asarray(followed, dtype=np.int64),
    }
    return frame


def split_participants(participants: list[str]) -> dict[str, np.ndarray]:
    """按参与者整体切分 60/20/20，禁止同一人跨切分。"""
    order = list(range(len(participants)))
    random.Random(SEED).shuffle(order)
    train_end = int(len(order) * .6)
    selection_end = int(len(order) * .8)
    split = {"train": np.asarray(order[:train_end]), "selection": np.asarray(order[train_end:selection_end]),
             "test": np.asarray(order[selection_end:])}
    for name, indices in split.items():
        if len(indices) == 0:
            raise ValueError(f"切分 {name} 为空")
    sets = {name: {participants[i] for i in indices} for name, indices in split.items()}
    if sets["train"] & sets["selection"] or sets["train"] & sets["test"] or sets["selection"] & sets["test"]:
        raise ValueError("参与者跨切分泄漏")
    return split


def features(frame: dict[str, Any], indices: np.ndarray, columns: list[str]) -> np.ndarray:
    """只允许基线列与第一波条目；调用方给出的列名在此白名单校验。"""
    allowed = {"wave1_g", "wave1_h", "age", "race"}
    if not set(columns) <= allowed:
        raise ValueError(f"非基线特征被拒绝: {set(columns) - allowed}")
    matrix = np.column_stack([frame[name][indices] for name in columns])
    if not np.isfinite(matrix).all():
        raise ValueError("特征包含非有限值")
    return matrix


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """MAE、RMSE、R² 与并列秩正确的Spearman；常数相关返回null。"""
    from scipy.stats import spearmanr

    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if truth.ndim != 1 or prediction.shape != truth.shape or not len(truth):
        raise ValueError("真值与预测必须是非空、等长的一维数组")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("真值与预测必须有限")
    error = prediction - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    variance = float(np.var(truth))
    r2 = float(1 - np.mean(error ** 2) / variance) if variance > 0 else None
    spearman = (float(spearmanr(truth, prediction).statistic)
                if len(truth) > 1 and np.ptp(truth) > 0 and np.ptp(prediction) > 0 else None)
    return {"n": int(len(truth)), "mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman}


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, selection_x: np.ndarray, selection_y: np.ndarray,
              grid: tuple[float, ...] = (.1, 1., 10., 100.)) -> tuple[Any, float]:
    """标准化后岭回归；alpha 只在选择集上挑选。"""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best = (None, float("inf"))
    for alpha in grid:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train_x, train_y)
        selection_mae = float(np.mean(np.abs(model.predict(selection_x) - selection_y)))
        if selection_mae < best[1] - 1e-12:
            best = (model, selection_mae)
    if best[0] is None:
        raise ValueError("岭回归未得到有效模型")
    return best[0], best[1]


def bootstrap_mae_delta(truth: np.ndarray, better: np.ndarray, reference: np.ndarray, repeats: int = 1000) -> list[float]:
    """按参与者重采样的配对 bootstrap，返回 MAE 差值的 95% 区间。"""
    generator = np.random.default_rng(SEED)
    indices = np.arange(len(truth))
    differences = []
    for _ in range(repeats):
        sample = generator.integers(0, len(truth), len(truth))
        differences.append(float(np.mean(np.abs(better[sample] - truth[sample])) - np.mean(np.abs(reference[sample] - truth[sample]))))
    return [float(value) for value in np.quantile(differences, [.025, .975])]


def evaluate_target(frame: dict[str, Any], split: dict[str, np.ndarray], target: str, baseline: str) -> dict[str, Any]:
    """三种基线/模型对照；测试只在选择完成后读取。"""
    columns = ["wave1_g", "wave1_h", "age", "race"]
    train_x = features(frame, split["train"], columns)
    selection_x = features(frame, split["selection"], columns)
    train_y, selection_y = frame[target][split["train"]], frame[target][split["selection"]]
    test_y = frame[target][split["test"]]
    model, selection_mae = fit_ridge(train_x, train_y, selection_x, selection_y)
    predictions = {
        "train_mean": float(np.mean(train_y)),
        "persistence_wave1": None,
        "ridge_baseline_features": model.predict(features(frame, split["test"], columns)),
    }
    predictions["persistence_wave1"] = frame[baseline][split["test"]]
    mean_prediction = np.full(len(test_y), predictions["train_mean"])
    results = {
        "train_mean": regression_metrics(test_y, mean_prediction),
        "persistence_wave1": regression_metrics(test_y, predictions["persistence_wave1"]),
        "ridge_baseline_features": regression_metrics(test_y, predictions["ridge_baseline_features"]),
    }
    results["selection"] = {
        "train_mean_mae": float(np.mean(np.abs(selection_y - np.mean(train_y)))),
        "persistence_mae": float(np.mean(np.abs(selection_y - frame[baseline][split["selection"]]))),
        "ridge_mae": selection_mae,
    }
    # 依赖训练残差的描述性 90% 区间，不是独立校准保证。
    residuals = train_y - model.predict(train_x)
    low, high = np.quantile(residuals, [.05, .95])
    interval_low = predictions["ridge_baseline_features"] + low
    interval_high = predictions["ridge_baseline_features"] + high
    results["ridge_descriptive_90pct_coverage"] = float(np.mean((test_y >= interval_low) & (test_y <= interval_high)))
    results["delta_vs_persistence"] = {
        "ridge_minus_persistence_mae": results["ridge_baseline_features"]["mae"] - results["persistence_wave1"]["mae"],
        "ci95": bootstrap_mae_delta(test_y, predictions["ridge_baseline_features"], predictions["persistence_wave1"]),
    }
    results["delta_vs_train_mean"] = {
        "persistence_minus_train_mean_mae": results["persistence_wave1"]["mae"] - results["train_mean"]["mae"],
        "ci95": bootstrap_mae_delta(test_y, predictions["persistence_wave1"], mean_prediction),
    }
    results["feature_columns"] = columns
    results["target_range"] = [float(np.min(test_y)), float(np.max(test_y))]
    return results


def run(source: Path, output_dir: Path) -> dict[str, Any]:
    """完成结构审计与两个条目组的随访预测，写出可复核 JSON。"""
    header, rows = load_rows(source)
    kept, duplicates = quarantine_duplicates(rows, header)
    frame = build_frame(kept, header)
    split = split_participants(frame["participants"])
    index = {name: position for position, name in enumerate(header)}
    item_values = Counter()
    for row in kept:
        for wave in (1, 2):
            for item in G_ITEMS + H_ITEMS:
                item_values[str(row[index[f"{item}_{wave}"]])] += 1
    audit = {
        "schema": "longitudinal_structure_audit.v2",
        "source_sha256": _sha256(source),
        "rows_raw": len(rows),
        "columns": len(header),
        "duplicates_quarantined": duplicates,
        "rows_after_quarantine": len(kept),
        "complete_both_waves": len(frame["participants"]),
        "followup_flag_consistency": {
            "complete_and_followup_1": int(np.sum(frame["wave2_followup"] == 1)),
            "complete_and_followup_0": int(np.sum(frame["wave2_followup"] == 0)),
        },
        "item_value_domain": {key: value for key, value in item_values.items()},
        "item_domain_clean": set(item_values) <= {"1", "2", "3", "4", "None", ""},
        "followup_interval": {
            "verified": False,
            "reason": "原表只有随访日期列 date2，没有基线日期列；无法从数据本身计算间隔",
            "date_columns": [name for name in header if "date" in name.lower()],
        },
        "item_to_scale_mapping": {
            "verified": False,
            "published_record_has_codebook": False,
            "checked_sources": ["Mendeley 记录描述与 method 字段", "公开文件列表（仅 1 个 xlsx）", "工作表与定义名称、批注"],
            "analysis_scope": "只使用 g1..g9 与 h1..h7 的条目合计，不命名为任何具体量表",
        },
    }
    targets = {}
    for target, baseline in (("wave2_g", "wave1_g"), ("wave2_h", "wave1_h")):
        targets[target] = evaluate_target(frame, split, target, baseline)
    forecast = {
        "schema": "two_wave_followup_prediction.v1",
        "source_sha256": audit["source_sha256"],
        "seed": SEED,
        "construct_scope": "g/h 条目组合计；题目字典未核实，不声称任何量表或临床含义",
        "participants": len(frame["participants"]),
        "split_sizes": {name: int(len(indices)) for name, indices in split.items()},
        "split_policy": "按参与者 60/20/20，seed 42，同一人不跨切分",
        "descriptives": {
            "wave1_g": {"mean": float(np.mean(frame["wave1_g"])), "sd": float(np.std(frame["wave1_g"], ddof=1))},
            "wave2_g": {"mean": float(np.mean(frame["wave2_g"])), "sd": float(np.std(frame["wave2_g"], ddof=1))},
            "wave1_h": {"mean": float(np.mean(frame["wave1_h"])), "sd": float(np.std(frame["wave1_h"], ddof=1))},
            "wave2_h": {"mean": float(np.mean(frame["wave2_h"])), "sd": float(np.std(frame["wave2_h"], ddof=1))},
            "wave1_g_wave2_g_pearson": float(np.corrcoef(frame["wave1_g"], frame["wave2_g"])[0, 1]),
            "wave1_h_wave2_h_pearson": float(np.corrcoef(frame["wave1_h"], frame["wave2_h"])[0, 1]),
            "age_range": [float(np.nanmin(frame["age"])), float(np.nanmax(frame["age"]))],
        },
        "targets": targets,
        "limitations": [
            "两波数据只支持随访预测，不构成日/周级长序列或提前预警",
            "题目字典与随访间隔未核实，因此不输出量表分数或临床阈值",
            "自评条目受共同方法偏差影响，不能证明自然接触或干预的因果作用",
            f"样本为云南在校学生，年龄 {int(np.nanmin(frame['age']))}–{int(np.nanmax(frame['age']))}，不代表全国或平台目标人群",
            "按参与者重采样的 1,000 次 bootstrap 只反映当前样本不确定性",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "longitudinal_structure_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "two_wave_prediction.json").write_text(json.dumps(forecast, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"audit": str(output_dir / "longitudinal_structure_audit.json"),
                      "prediction": str(output_dir / "two_wave_prediction.json"),
                      "participants": forecast["participants"]}, ensure_ascii=False))
    return forecast


def _sha256(path: Path) -> str:
    """流式摘要，避免把大表整体读入内存。"""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.source, arguments.output_dir)
