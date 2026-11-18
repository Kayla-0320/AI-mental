"""来源OOF、校准作用域和受约束决策的边界回归。"""
from __future__ import annotations

import numpy as np
from experiments.round7_decision_oof import availability, calibrate, decide, fit_pipeline, fold_plan, matrix, raw_predict, select_rules


def data(n: int = 36) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    values = {name: rng.normal(size=(n, 4)).astype(np.float32) for name in ["roberta_mean", "bge_cls", "visual", "audio"]}
    values.update(face_summary=np.zeros((n, 29), np.float32), modality_mask=np.ones((n, 3), bool), quality=np.ones((n, 3), np.float32),
                  labels=np.arange(n) % 3, raw_text=np.array([['天气非常糟糕', '今天是周二', '这个事情太好了'][i % 3] for i in range(n)]),
                  sample_ids=np.array([str(i) for i in range(n)]), source_groups=np.array([str(i // 3) for i in range(n)]))
    return values


def test_negative_guard_precedes_neutral_override_and_missing_abstains() -> None:
    p = np.array([[.6, .3, .1], [.45, .3, .25], [.1, .2, .7], [np.nan, np.nan, np.nan]])
    rule = {"kind": "band", "negative_guard": .55, "neutral_threshold": .25}
    assert decide(p, rule).tolist() == [0, 1, 2, -1]


def test_all_outer_and_inner_source_groups_stay_together() -> None:
    values = data()
    for train, valid in fold_plan(values, 4):
        assert not set(values["source_groups"][train]) & set(values["source_groups"][valid])
        inner = {key: value[train] for key, value in values.items()}
        for a, b in fold_plan(inner, 3):
            assert not set(inner["source_groups"][a]) & set(inner["source_groups"][b])


def test_calibrator_ignores_labels_outside_its_selection_scope() -> None:
    rng = np.random.default_rng(42)
    raw = {"scores": rng.normal(size=(90, 3)), "face": np.array([True] * 60 + [False] * 30)}
    labels = np.arange(90) % 3
    scope = raw["face"]
    a = calibrate(raw, labels, scope)
    labels[~scope] = (labels[~scope] + 1) % 3
    b = calibrate(raw, labels, scope)
    assert a == b and a["counts"]["fallback"] == 0


def test_training_scalers_are_not_refitted_and_missing_nan_media_is_zeroed() -> None:
    training = data()
    _, fitted = matrix(training)
    original = fitted["audio"].mean_.copy()
    validation = data()
    validation["audio"][:] = np.nan
    validation["modality_mask"][:, 1] = False
    transformed, _ = matrix(validation, fitted)
    assert np.isfinite(transformed).all() and np.array_equal(original, fitted["audio"].mean_)


def test_fold_pipeline_does_not_read_prediction_labels_and_abstains_without_text() -> None:
    training = data()
    source = [{"text": ['难受又失望', '昨天吃了面条', '开心又高兴'][i % 3], "label": i % 3} for i in range(12)]
    model = fit_pipeline(training, source)
    validation = data(6)
    a = raw_predict(model, validation)
    validation["labels"][:] = 2
    validation["source_groups"][:] = 'different'
    b = raw_predict(model, validation)
    assert np.array_equal(a["scores"], b["scores"])
    validation["modality_mask"][0, 0] = False
    validation["raw_text"][1] = ''
    missing = raw_predict(model, validation)
    assert np.isnan(missing["scores"][:2]).all()


def test_constraint_optimizer_returns_feasible_rule_for_every_cost_floor() -> None:
    p = np.array([[.4, .35, .25], [.2, .4, .4], [.2, .5, .3], [.1, .2, .7]])
    selected = select_rules(p, np.array([0, 0, 1, 2]))["selected"]
    for floor, row in selected.items():
        assert row["metrics"]["per_class_recall"][0] >= float(floor)
