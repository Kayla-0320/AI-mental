"""两波数据结构审计与随访预测的边界测试，不用真实大表。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/round3_longitudinal_predict.py"
    spec = importlib.util.spec_from_file_location("round3_longitudinal_predict", path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def header() -> list[str]:
    names = ["cohort", "code", "wave2_followup", "age", "race"]
    names += [f"g{i}_1" for i in range(1, 10)] + [f"g{i}_2" for i in range(1, 10)]
    names += [f"h{i}_1" for i in range(1, 8)] + [f"h{i}_2" for i in range(1, 8)]
    return names


def row(cohort: int, code: int, g1: float, g2: float, h1: float = 2.0, h2: float = 2.0, followup: int = 1) -> tuple:
    values = [cohort, code, followup, 13, 1]
    values += [g1] * 9 + [g2] * 9
    values += [h1] * 7 + [h2] * 7
    return tuple(values)


def test_duplicate_participants_are_quarantined_entirely() -> None:
    predict = module()
    rows = [row(1, 100, 2, 2), row(1, 100, 3, 3), row(1, 200, 2, 2)]
    kept, dropped = predict.quarantine_duplicates(rows, header())
    assert [(item["cohort"], item["code"]) for item in dropped] == [(1, 100)]
    assert len(kept) == 1 and kept[0][1] == 200


def test_incomplete_or_out_of_domain_items_are_excluded() -> None:
    predict = module()
    rows = [row(1, 100, 2, 2), row(1, 200, 2, 5), row(1, 300, 2, 2)]
    rows[2] = rows[2][:10] + (None,) + rows[2][11:]
    frame = predict.build_frame(rows, header())
    assert frame["participants"] == ["1-100"]
    assert frame["wave1_g"][0] == 18.0 and frame["wave2_g"][0] == 18.0


def test_split_is_participant_level_and_deterministic() -> None:
    predict = module()
    participants = [f"1-{i}" for i in range(50)]
    first = predict.split_participants(participants)
    second = predict.split_participants(participants)
    for name in ("train", "selection", "test"):
        assert np.array_equal(first[name], second[name])
    sets = {name: set(participants[i] for i in indices) for name, indices in first.items()}
    assert not sets["train"] & sets["test"] and not sets["selection"] & sets["test"]
    assert sum(len(values) for values in first.values()) == 50


def test_features_reject_wave2_and_non_baseline_columns() -> None:
    predict = module()
    frame = {"wave1_g": np.array([1.0, 2.0]), "wave1_h": np.array([1.0, 2.0]),
             "age": np.array([13.0, 14.0]), "race": np.array([1.0, 2.0]), "wave2_g": np.array([9.0, 9.0])}
    indices = np.array([0, 1])
    assert predict.features(frame, indices, ["wave1_g", "age"]).shape == (2, 2)
    with pytest.raises(ValueError):
        predict.features(frame, indices, ["wave2_g"])
    with pytest.raises(ValueError):
        predict.features(frame, indices, ["wave1_g", "wave2_g"])


def test_regression_metrics_and_bootstrap_are_stable() -> None:
    predict = module()
    truth = np.array([10.0, 12.0, 14.0, 16.0])
    assert predict.regression_metrics(truth, truth)["mae"] == 0.0
    summary = predict.regression_metrics(truth, truth + 1.0)
    assert summary["mae"] == pytest.approx(1.0) and summary["rmse"] == pytest.approx(1.0)
    assert 0 < summary["r2"] < 1
    poor = predict.regression_metrics(truth, np.full_like(truth, 1e6))
    assert poor["r2"] < 0
    a = predict.bootstrap_mae_delta(truth, truth + 1.0, truth + 2.0, repeats=200)
    b = predict.bootstrap_mae_delta(truth, truth + 1.0, truth + 2.0, repeats=200)
    assert a == b and a[0] <= a[1]


def test_evaluate_target_never_reads_test_before_selection() -> None:
    predict = module()
    generator = np.random.default_rng(42)
    size = 300
    wave1 = generator.normal(13, 4, size)
    frame = {"participants": [f"1-{i}" for i in range(size)], "wave1_g": wave1, "wave1_h": generator.normal(9, 3, size),
             "wave2_g": wave1 * .55 + generator.normal(0, 3.4, size), "wave2_h": generator.normal(9, 3, size),
             "age": generator.integers(10, 18, size).astype(float), "race": generator.integers(1, 3, size).astype(float),
             "wave2_followup": np.ones(size, dtype=np.int64)}
    split = predict.split_participants(frame["participants"])
    result = predict.evaluate_target(frame, split, "wave2_g", "wave1_g")
    assert result["persistence_wave1"]["mae"] > 0
    assert result["ridge_baseline_features"]["n"] == len(split["test"])
    assert result["feature_columns"] == ["wave1_g", "wave1_h", "age", "race"]
    assert result["delta_vs_persistence"]["ci95"][0] <= result["delta_vs_persistence"]["ci95"][1]
