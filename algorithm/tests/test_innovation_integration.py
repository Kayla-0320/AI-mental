"""业务接入回归：真实像素代理、缺失降级、模型出处与业务路由。"""
from dataclasses import asdict
from datetime import datetime, timezone
import base64
import hashlib
import io
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from perception import innovation
from shared import InnovationObservation, MultimodalPrediction


def frames(black: bool = False) -> list[dict]:
    rng = np.random.default_rng(42)
    now = time.time() * 1000
    result = []
    for i in range(3):
        pixels = np.zeros((224, 224, 3), np.uint8) if black else rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format="JPEG")
        result.append({"sampledAt": now - (3 - i) * 100, "jpeg": "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()})
    return result


@pytest.fixture
def models(monkeypatch):
    monkeypatch.delenv("INNOVATION_ENABLED", raising=False)
    calls = []

    class Encoder:
        def encode_text(self, texts):
            return np.ones((len(texts), 4), np.float32)
        def encode(self, images, texts):
            calls.append("visual")
            return {"visual": np.ones((len(images), 4), np.float32)}

    class Runtime:
        meta = {"dimensions": {"text": 4, "audio": 2, "visual": 4}, "task": "sentiment",
                "encoder_fingerprints": {}, "text_only_calibration": {"status": "calibrated", "temperature": 1}}
        temperature = 1
        knowledge = None
        def predict(self, batch):
            calls.append(batch)
            return MultimodalPrediction("sentiment", ["negative", "neutral", "positive"], [[.2, .3, .5]],
                [[1.0]], ["text"], batch.sample_ids, "calibrated", "a" * 64, [False])

    monkeypatch.setattr(innovation, "_models", lambda: (Runtime(), Encoder()))
    return calls


def test_shared_observation_serializes_without_media():
    first, second = InnovationObservation(), InnovationObservation()
    first.quality["visual"] = {"quality_score": .2}
    assert second.quality == {} and not second.available and second.probabilities == []
    assert "visualFrames" not in asdict(first)


def test_real_pixel_quality_matches_round17():
    from experiments.round17_pixels import quality_from_frames
    rng = np.random.default_rng(42)
    original = rng.integers(0, 256, (3, 128, 128, 3), dtype=np.uint8)
    blocked = original.copy(); blocked[:, 24:104, 24:104] = 127
    for value in (original, blocked, np.zeros_like(original)):
        assert asdict(innovation.visual_quality(value)) == asdict(quality_from_frames(value))
    assert innovation.visual_quality(blocked).quality_score <= .25


@pytest.mark.parametrize("kind", ["old", "future", "duplicate", "too_many", "invalid_base64", "oversized"])
def test_frame_contract_rejects_invalid_payload(kind):
    payload = frames(); now = time.time() * 1000
    if kind == "old": payload[0]["sampledAt"] = now - 11000
    if kind == "future": payload[-1]["sampledAt"] = now + 10000
    if kind == "duplicate": payload[1]["sampledAt"] = payload[0]["sampledAt"]
    if kind == "too_many": payload.append(payload[-1])
    if kind == "invalid_base64": payload[0]["jpeg"] = "data:image/jpeg;base64,????"
    if kind == "oversized": payload[0]["jpeg"] += "a" * 100001
    with pytest.raises(ValueError): innovation.decode_frames(payload, now)


def test_valid_frames_execute_visual_and_never_audio(models):
    result = innovation.observe("我想把这件事说清楚", frames())
    assert result.available and result.observed_modalities == ["text", "visual"]
    assert "visual" in models
    assert models[-1].observed["audio"] == [False]
    assert "jpeg" not in str(asdict(result))


def test_black_frames_fall_back_before_visual_encoder(models):
    result = innovation.observe("我担心考试", frames(black=True))
    assert result.available and result.observed_modalities == ["text"]
    assert "visual" not in models and models[-1].observed["visual"] == [False]
    assert "visual_quality_fallback_to_text" in result.warnings


def test_missing_artifact_returns_unavailable_without_fake_probabilities(monkeypatch):
    monkeypatch.setattr(innovation, "_models", lambda: (_ for _ in ()).throw(ValueError("mismatch")))
    result = innovation.observe("今天还好")
    assert not result.available and result.probabilities == [] and innovation.prompt_reference(result) == ""


def test_quality_head_shadow_records_mos_without_dropping_visual(models, monkeypatch):
    """默认影子模式：记录模型 MOS，但不改变本轮专家选择。"""
    monkeypatch.setenv("INNOVATION_QUALITY_GUARD", "shadow")

    def fake(frames_arg, visuals, result):
        result.quality["visual"]["metrics"].update({"quality_mos_worst": 3.4, "quality_mos_mean": 3.5, "quality_mos_low": 0.0})
        return {"low": False, "worst": 3.4, "threshold": 2.5}

    monkeypatch.setattr(innovation, "frame_quality", fake)
    result = innovation.observe("我想把这件事说清楚", frames())
    assert result.observed_modalities == ["text", "visual"]
    assert result.quality["visual"]["metrics"]["quality_mos_worst"] == 3.4


def test_quality_head_enforce_falls_back_only_when_explicitly_enabled(models, monkeypatch):
    """只有显式 enforce 才让画质头决定退回文字；这是配置项，不是默认行为。"""
    monkeypatch.setenv("INNOVATION_QUALITY_GUARD", "enforce")
    monkeypatch.setattr(innovation, "frame_quality", lambda frames_arg, visuals, result: {"low": True, "worst": 1.2, "threshold": 2.5})
    result = innovation.observe("我担心考试", frames())
    assert result.observed_modalities == ["text"]
    assert "quality_model_fallback_to_text" in result.warnings


@pytest.mark.parametrize("text,available", [("好", True), ("不", True), ("我不难过", True), ("！！！", False), ("https://example.org", False)])
def test_text_quality_keeps_short_and_negated_expression(models, text, available):
    assert innovation.observe(text).available == available


def test_production_observation_route_without_research_token(monkeypatch):
    import api.innovation as api
    from perception.knowledge_moe.observation import unavailable
    monkeypatch.delenv("ROUND2_RESEARCH_TOKEN", raising=False)
    monkeypatch.setattr(api, "infer_observation", lambda text, sid, digest, stamp: unavailable(sid, digest, stamp, "model_missing"))
    app = FastAPI(); app.include_router(api.router, prefix="/api/v1")
    body = {"text": "今天还好", "source_message_id": "source-1", "source_sha256": hashlib.sha256("今天还好".encode()).hexdigest(),
            "observed_at": datetime.now(timezone.utc).isoformat()}
    response = TestClient(app).post("/api/v1/innovation/observation", json=body)
    assert response.status_code == 200 and response.json()["reason"] == "model_missing"
    assert response.json()["evidence_source"]["source_message_id"] == "source-1"
