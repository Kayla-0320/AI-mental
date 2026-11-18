"""验证过去窗口、任务边界、产物篡改与研究 API 隔离。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.research import router
from assessment.forecasting import ForecastPredictor, history_features, validate_history
from assessment.forecasting.predictor import TARGETS, SCHEMA
from shared import EmotionObservation


def make_history() -> list[EmotionObservation]:
    return [EmotionObservation(index, 20 + index * 2, 70 - index) for index in range(1, 9)]


def artifact(tmp_path: Path, method: str = "persistence") -> Path:
    weights = tmp_path / "weights.npz"
    np.savez(weights, stub=np.ones(1))
    metadata = {"schema": SCHEMA, "targets": TARGETS, "context": 8, "scale": [0, 100],
                "fit_split": "train", "selection_split": "validation", "weights_file": weights.name,
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "model_id": "test-fixture",
                "horizons": {"1": {"selected_method": method, "validation_absolute_error_quantile": [0.1, 0.2]}}}
    path = tmp_path / "forecast.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", ["short", "duplicate", "reverse", "gap", "nan", "range", "source", "boolean_step"])
def test_reject_invalid_or_mismatched_observations(kind: str) -> None:
    points = make_history()
    if kind == "short":
        points.pop()
    elif kind == "duplicate":
        points[-1].step = points[-2].step
    elif kind == "reverse":
        points.reverse()
    elif kind == "gap":
        points[-1].step = 10
    elif kind == "nan":
        points[1].positive = float("nan")
    elif kind == "range":
        points[1].negative = 101
    elif kind == "source":
        points[1].source = "classifier_probability"
    elif kind == "boolean_step":
        points[0].step = True
    with pytest.raises(ValueError):
        validate_history(points)


def test_prediction_and_bands_are_not_clinical(tmp_path: Path) -> None:
    prediction = ForecastPredictor(artifact(tmp_path)).predict(make_history())
    assert prediction.positive == 36
    assert prediction.negative == 62
    assert prediction.target_step == 9
    assert prediction.interval[TARGETS[0]] == [26, 46]
    assert "research_only_not_clinical_forecast" in prediction.warnings


def test_artifact_digest_and_task_are_enforced(tmp_path: Path) -> None:
    path = artifact(tmp_path)
    (tmp_path / "weights.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        ForecastPredictor(path)
    path = artifact(tmp_path)
    value = json.loads(path.read_text())
    value["targets"] = ["depression", "crisis"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="task/schema"):
        ForecastPredictor(path)


def test_history_features_only_consume_supplied_past() -> None:
    full = np.arange(24, dtype=float).reshape(12, 2) / 100
    before = history_features(full[:8][None])
    full[8:] = 99
    np.testing.assert_array_equal(before, history_features(full[:8][None]))
    with pytest.raises(ValueError):
        history_features(full[None])


def test_forecast_api_disabled_auth_and_valid_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    body = {"history": [{"step": p.step, "positive": p.positive, "negative": p.negative} for p in make_history()]}
    monkeypatch.delenv("ROUND2_RESEARCH_TOKEN", raising=False)
    assert client.post("/research/forecast", json=body).status_code == 503
    monkeypatch.setenv("ROUND2_RESEARCH_TOKEN", "local-test-only")
    assert client.post("/research/forecast", json=body).status_code == 401
    monkeypatch.setenv("ROUND2_FORECAST_ARTIFACT", str(artifact(tmp_path)))
    response = client.post("/research/forecast", json=body, headers={"Authorization": "Bearer local-test-only"})
    assert response.status_code == 200
    assert response.json()["target_step"] == 9
    assert client.post("/research/forecast", json={**body, "horizon": 2},
                       headers={"Authorization": "Bearer local-test-only"}).status_code == 422
    assert client.post("/research/forecast", json={**body, "unknown": 1},
                       headers={"Authorization": "Bearer local-test-only"}).status_code == 422


def test_training_windows_exclude_future_and_keep_participants_separate() -> None:
    path = Path(__file__).resolve().parents[2] / "scripts/round2_forecast.py"
    spec = importlib.util.spec_from_file_location("round2_forecast", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = {1: {step: [step / 100, 0.2] for step in range(1, 12)}}
    x, y, _, origins = module.windows(source, [1], 3)
    assert origins.tolist() == [8]
    assert x[0, -1, 0] == 0.08 and y[0, 0] == 0.11
    source[1][11][0] = 0.99
    np.testing.assert_array_equal(x, module.windows(source, [1], 3)[0])
    split = module.participant_split(list(range(58)))
    assert not set(split["train"]) & set(split["validation"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["validation"]) & set(split["test"])
