"""第三轮独立接口复核：空证据、产物错误、出处及缺失模式校准边界。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from api.research import router
from intervention.strategy_learning import SupportStrategyModel
from perception.knowledge_moe.observation import infer_observation
from shared import MultimodalPrediction


def strategy_artifact(directory: Path, fault: str = "none") -> Path:
    """构造有合法摘要的最小策略工件，不训练或使用真人记录。"""
    weights = directory / "weights.npz"
    np.savez(weights, coef=np.array([[0.1], [-0.1]]), intercept=np.zeros(2), idf=np.ones(1))
    m: Any = {"schema": "esconv_strategy.v1", "language": "en", "selection_split": "selection",
              "calibration_split": "calibration", "labels": ["Question", "Reflection"],
              "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "vocabulary": {"abc": 0}, "temperature": 1.0, "use_context": True}
    if fault == "missing_use_context":
        del m["use_context"]
    elif fault == "string_use_context":
        m["use_context"] = "false"
    elif fault == "blank_label":
        m["labels"][0] = "  "
    elif fault == "list_metadata":
        m = []
    (directory / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return directory


def research_client(directory: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """隔离令牌及工件目录，按真实HTTP序列化检查状态码。"""
    monkeypatch.setenv("ROUND2_RESEARCH_TOKEN", "review-only-token")
    monkeypatch.setenv("ROUND3_STRATEGY_ARTIFACT", str(directory))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("content", ["", "   ", "???", "123456"])
def test_role_markers_do_not_become_english_evidence(content: str) -> None:
    directory = Path(__file__).resolve().parents[1] / "artifacts" / "round3_strategy"
    result = SupportStrategyModel(directory).predict([{"role": "user", "content": content}], "en")
    assert not result.available
    assert result.probabilities == []


@pytest.mark.parametrize("fault", ["missing_use_context", "string_use_context", "blank_label", "list_metadata", "malformed_json"])
def test_bad_strategy_artifact_is_service_unavailable(fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = strategy_artifact(tmp_path, fault)
    if fault == "malformed_json":
        (directory / "manifest.json").write_text("{broken JSON", encoding="utf-8")
    client = research_client(directory, monkeypatch)
    response = client.post("/research/strategy", json={"messages": [{"role": "user", "content": "abcdef"}], "language": "en"},
                           headers={"Authorization": "Bearer review-only-token"})
    assert response.status_code == 503, response.text


def test_unknown_strategy_language_is_not_silently_english(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = research_client(strategy_artifact(tmp_path), monkeypatch)
    response = client.post("/research/strategy", json={"messages": [{"role": "user", "content": "abcdef"}], "language": "fr"},
                           headers={"Authorization": "Bearer review-only-token"})
    assert response.status_code == 422


def test_forged_source_and_blank_id_are_rejected_before_model_loading() -> None:
    text = "我担心考试。"
    stamp = datetime.now(timezone.utc).isoformat()
    with pytest.raises(ValueError):
        infer_observation(text, "message1", "0" * 64, stamp)
    with pytest.raises(ValueError):
        infer_observation(text, "   ", hashlib.sha256(text.encode()).hexdigest(), stamp)


@pytest.mark.parametrize("calibration_mode", ["absent", "matching", "wrong_condition", "wrong_fit_split", "invalid_n", "nonfinite_temperature"])
def test_full_modality_temperature_does_not_claim_text_only_calibration(
    calibration_mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用冻结编码器替身检查输出口径，不制造真实VLM性能或校准证据。"""
    import perception.knowledge_moe.features as features
    import perception.knowledge_moe.runtime as runtime
    import perception.knowledge_moe.observation as observation

    artifact_dir, encoder_dir = tmp_path / "artifact", tmp_path / "encoder"
    artifact_dir.mkdir()
    encoder_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "weights.pt").write_bytes(b"contract fixture")
    (encoder_dir / "pytorch_model.bin").write_bytes(b"contract fixture")

    class FakeEncoder:
        """只生成有界测试向量，不加载任何模型。"""

        fingerprint = "fixture_encoder"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def encode_text(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 2), dtype=np.float32)

    class FakeRuntime:
        """模拟完整模态产物的校准状态，检验文本-only接口不夸大适用范围。"""

        meta = {"class_names": ["negative", "neutral", "positive"], "dimensions": {"text": 2, "audio": 2, "visual": 2},
                "live_text_encoder_fingerprint": "fixture_encoder", "task": "fixture_sentiment3",
                "encoder_fingerprints": {"text": "fixture_encoder", "audio": "fixture_audio", "visual": "fixture_visual"},
                "calibration": {"status": "calibrated", "fit_split": "calibration", "modality_pattern": "full"}}
        temperature = 2.0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def predict(self, batch: Any) -> MultimodalPrediction:
            assert batch.observed == {"text": [True], "audio": [False], "visual": [False]}
            return MultimodalPrediction(batch.task, batch.class_names, [[0.6, 0.2, 0.2]], [[1.0]], ["text"], batch.sample_ids,
                                        "calibrated", "fixture_model", abstained=[False])

    if calibration_mode != "absent":
        details = {"status": "calibrated", "fit_split": "calibration", "condition": "text_present_audio_visual_missing", "n": 10, "temperature": 1.0}
        if calibration_mode == "wrong_condition":
            details["condition"] = "full_modalities"
        elif calibration_mode == "wrong_fit_split":
            details["fit_split"] = "test"
        elif calibration_mode == "invalid_n":
            details["n"] = True
        elif calibration_mode == "nonfinite_temperature":
            details["temperature"] = float("nan")
        FakeRuntime.meta["text_only_calibration"] = details

    monkeypatch.setattr(features, "FrozenCLIPEncoder", FakeEncoder)
    monkeypatch.setattr(runtime, "FusionRuntime", FakeRuntime)
    monkeypatch.setattr(observation, "_CACHE", {})
    monkeypatch.setenv("ROUND3_FUSION_ARTIFACT", str(artifact_dir))
    monkeypatch.setenv("ROUND3_VLM_ENCODER", str(encoder_dir))
    text = "我担心考试。"
    result = infer_observation(text, "message1", hashlib.sha256(text.encode()).hexdigest(), datetime.now(timezone.utc).isoformat())
    assert result.available
    if calibration_mode == "matching":
        assert result.calibration["status"] == "calibrated"
        assert result.probabilities == pytest.approx([9 / 11, 1 / 11, 1 / 11])
    else:
        assert result.calibration["status"] != "calibrated", result.calibration
        assert result.probabilities == [0.6, 0.2, 0.2]
