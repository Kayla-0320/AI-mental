"""独立复核：非法自评不应变成有效观测，坏产物应作为服务不可用处理。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from api.research import router
from assessment.forecasting import validate_history
from assessment.forecasting.predictor import SCHEMA, TARGETS
from shared import EmotionObservation


def write_artifact(directory: Path, fault: str = "none") -> Path:
    """构造任务身份正确但不同结构故障的产物，避免依赖训练。"""
    arrays = {"stub": np.ones(1)}
    settings: dict[str, Any] = {"selected_method": "persistence", "validation_absolute_error_quantile": [0.1, 0.2]}
    if fault in {"missing_ridge_weight", "wrong_ridge_shape"}:
        settings["selected_method"] = "ridge"
        if fault == "wrong_ridge_shape":
            arrays = {"h1_coef": np.ones((2, 3)), "h1_intercept": np.zeros(2)}
    elif fault == "missing_gru_weight":
        settings["selected_method"] = "gru"
    elif fault == "unsupported_method":
        settings["selected_method"] = "arbitrary"
    elif fault == "invalid_error_band":
        settings["validation_absolute_error_quantile"] = [0.1]
    weights = directory / "weights.npz"
    np.savez(weights, **arrays)
    metadata: Any = {"schema": SCHEMA, "targets": TARGETS, "context": 8, "scale": [0, 100],
                    "fit_split": "train", "selection_split": "validation", "weights_file": weights.name,
                    "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "model_id": "independent-review-fixture", "horizons": {"1": settings}}
    if fault == "missing_model_id":
        del metadata["model_id"]
    elif fault == "missing_horizons":
        del metadata["horizons"]
    elif fault == "null_horizons":
        metadata["horizons"] = None
    elif fault == "list_instead_of_metadata":
        metadata = []
    path = directory / "forecast.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def request_body() -> dict[str, Any]:
    """使用同一有效过去窗口以定位产物错误而非用户输入错误。"""
    return {"history": [{"step": step, "positive": 35, "negative": 20} for step in range(1, 9)]}


def client_for(path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """为每个测试隔离启用研究接口，关闭服务异常直抛以检查 HTTP 契约。"""
    monkeypatch.setenv("ROUND2_RESEARCH_TOKEN", "isolated-review-token")
    monkeypatch.setenv("ROUND2_FORECAST_ARTIFACT", str(path))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("value", [True, False, "invalid"])
def test_library_rejects_non_numeric_self_reports_as_value_error(value: object) -> None:
    history = [EmotionObservation(step, 35, 20) for step in range(1, 9)]
    history[-1].positive = value
    with pytest.raises(ValueError):
        validate_history(history)


def test_http_rejects_boolean_self_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = client_for(write_artifact(tmp_path), monkeypatch)
    body = request_body()
    body["history"][-1]["positive"] = True
    response = client.post("/research/forecast", json=body, headers={"Authorization": "Bearer isolated-review-token"})
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("fault", ["missing_ridge_weight", "wrong_ridge_shape", "missing_gru_weight",
                                 "unsupported_method", "invalid_error_band", "missing_model_id",
                                 "missing_horizons", "null_horizons", "list_instead_of_metadata"])
def test_bad_artifact_is_unavailable_not_client_error_or_server_crash(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = client_for(write_artifact(tmp_path, fault), monkeypatch)
    response = client.post("/research/forecast", json=request_body(), headers={"Authorization": "Bearer isolated-review-token"})
    assert response.status_code == 503, (fault, response.status_code, response.text)


def test_integer_scale_values_still_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = client_for(write_artifact(tmp_path), monkeypatch)
    response = client.post("/research/forecast", json=request_body(), headers={"Authorization": "Bearer isolated-review-token"})
    assert response.status_code == 200, response.text
    assert response.json()["positive"] == 35


def test_non_ascii_authorization_is_rejected_without_crashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = client_for(write_artifact(tmp_path), monkeypatch)
    response = client.post("/research/forecast", json=request_body(), headers=[(b"Authorization", b"Bearer \xff")])
    assert response.status_code == 401, response.text


@pytest.mark.parametrize("horizon", [0, -1, True])
def test_window_builder_cannot_be_used_with_non_future_horizon(horizon: int) -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "round2_forecast.py"
    spec = importlib.util.spec_from_file_location("forecast_independent_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = {1: {step: [step / 100, 0.2] for step in range(1, 12)}}
    with pytest.raises(ValueError):
        module.windows(source, [1], horizon)
