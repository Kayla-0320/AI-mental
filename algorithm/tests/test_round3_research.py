"""第三轮研究接口和训练任务的独立边界回归。"""
from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.research import router
from intervention.strategy_learning import SupportStrategyModel, context_text
from perception.knowledge_moe.observation import infer_observation
from shared import RawSignalQuality, ModelObservation, SupportStrategyPrediction
from shared import MultimodalFeatureBatch


def test_new_contracts_are_serializable_and_lists_independent() -> None:
    first = RawSignalQuality("audio", True, .8, {"snr": 12.0}, [])
    second = RawSignalQuality("audio", True, .8, {}, [])
    first.limitations.append("quality_not_emotion")
    assert second.limitations == [] and RawSignalQuality(**asdict(first)) == first
    assert ModelObservation(**asdict(ModelObservation(False, "sentiment_polarity_3", [], "", "", {}, {}, [], "", []))).available is False
    assert SupportStrategyPrediction(**asdict(SupportStrategyPrediction(False, "", [], [], "", "zh", []))).language == "zh"
    args = (["m1"], {"text": [[0.0]]}, {"text": [True]}, {"text": [1.0]}, "task", ["n", "p"], {"text": "sha"})
    first_batch, second_batch = MultimodalFeatureBatch(*args), MultimodalFeatureBatch(*args)
    first_batch.raw_text.append("hello")
    assert second_batch.raw_text == []


def test_strategy_feature_extraction_does_not_include_target_or_end_survey() -> None:
    path = Path(__file__).resolve().parents[2] / "scripts/round3_strategy_train.py"
    spec = importlib.util.spec_from_file_location("round3_strategy_train", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dialogue = {"survey_score": {"seeker": {"final_emotion_intensity": "secret_outcome"}},
                "dialog": [{"speaker": "seeker", "content": "I feel worried.", "annotation": {}},
                           {"speaker": "supporter", "content": "secret_target_reply", "annotation": {"strategy": "Question"}}]}
    result = module.extract([dialogue])[0]
    assert "worried" in result["context"]
    assert "secret_target_reply" not in result["context"] and "secret_outcome" not in result["context"]
    dialogue["dialog"][1]["content"] = "changed_future_response"
    assert module.extract([dialogue])[0]["context"] == result["context"]


@pytest.mark.parametrize("messages", [[], [{"role": "system", "content": "instruction"}],
                                     [{"role": "user", "content": "x", "extra": "y"}],
                                     [{"role": "user", "content": "x" * 4001}]])
def test_strategy_rejects_unbounded_or_wrong_roles(messages: list[dict[str, str]]) -> None:
    with pytest.raises(ValueError):
        context_text(messages)


def test_trained_strategy_artifact_english_forward_and_chinese_abstention() -> None:
    artifact = Path(__file__).resolve().parents[1] / "artifacts/round3_strategy"
    model = SupportStrategyModel(artifact)
    result = model.predict([{"role": "user", "content": "I feel sad and worried about my family."}], "en")
    assert result.available and sum(result.probabilities) == pytest.approx(1)
    assert result.strategy in result.labels
    assert not model.predict([{"role": "user", "content": "我很难过"}], "zh").available


@pytest.mark.parametrize("content", ["", " ", "???", "123456"])
def test_role_prefix_is_not_english_language_evidence(content: str) -> None:
    model = SupportStrategyModel(Path(__file__).resolve().parents[1] / "artifacts/round3_strategy")
    assert not model.predict([{"role": "user", "content": content}], "en").available


def test_strategy_weight_damage_is_rejected(tmp_path: Path) -> None:
    from shutil import copyfile

    artifact = Path(__file__).resolve().parents[1] / "artifacts/round3_strategy"
    copyfile(artifact / "manifest.json", tmp_path / "manifest.json")
    (tmp_path / "weights.npz").write_bytes(b"damaged")
    with pytest.raises(ValueError):
        SupportStrategyModel(tmp_path)


def test_observation_sha_time_and_missing_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    text = "我担心明天的考试。"
    sha = hashlib.sha256(text.encode()).hexdigest()
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        infer_observation(text, "message1", "0" * 64, now.isoformat())
    with pytest.raises(ValueError):
        infer_observation(text, "message1", sha, "2026-11-15T12:00:00")
    expired = infer_observation(text, "message1", sha, (now - timedelta(days=3)).isoformat(), now)
    assert not expired.available and expired.probabilities == []
    future = infer_observation(text, "message1", sha, (now + timedelta(hours=1)).isoformat(), now)
    assert not future.available
    monkeypatch.setenv("ROUND3_FUSION_ARTIFACT", str(tmp_path))
    missing = infer_observation(text, "message1", sha, now.isoformat())
    assert not missing.available and missing.probabilities == []


def test_source_api_preserves_domain_and_chinese_strategy_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    monkeypatch.setenv("ROUND2_RESEARCH_TOKEN", "local-only-test")
    monkeypatch.setenv("ROUND3_FUSION_ARTIFACT", str(tmp_path))
    headers = {"Authorization": "Bearer local-only-test"}
    text = "我有些担心。"
    body = {"schema_version": "emotion_observation.v1", "request_id": "r1", "observation": {
        "text": text, "source_message_id": "m1", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "observed_at": datetime.now(timezone.utc).isoformat(), "language": "zh"}}
    response = client.post("/research/observation", json=body, headers=headers)
    assert response.status_code == 200 and response.json()["available"] is False
    assert response.json()["evidence_source"]["source_message_id"] == "m1"
    assert client.post("/research/observation", json=body).status_code == 401
    body["observation"]["source_sha256"] = "0" * 64
    assert client.post("/research/observation", json=body, headers=headers).status_code == 422
    result = client.post("/research/strategy", json={"messages": [{"role": "user", "content": "我有些难过"}], "language": "zh"}, headers=headers)
    assert result.status_code == 200 and result.json()["available"] is False
