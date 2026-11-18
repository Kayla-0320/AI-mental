"""真实 HTTP / SSE 路由回归；LLM、感知与长期记忆为可控替身，不模拟用户研究。"""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.intervention as iv


@pytest.fixture
def harness(monkeypatch):
    captured = []
    emotion = lambda: SimpleNamespace(text_emotion_probs=[.1, .05, .02, .03, .8],
                                     confidence=.8, evidence=[], crisis_keywords=[])
    monkeypatch.setattr(iv, "get_perception_service", lambda: SimpleNamespace(analyze_text=lambda _: emotion()))
    monkeypatch.setattr(iv, "map_all", lambda _: {})
    monkeypatch.setattr(iv, "build_memory_context", lambda **_: "")
    monkeypatch.setattr(iv, "MemoryExtractor", lambda: SimpleNamespace(extract_and_store=lambda **_: None))
    monkeypatch.setattr(iv, "predict_distortion_safe", lambda _: None)
    monkeypatch.setattr(iv, "build_scene_context", lambda *_, **__: "")
    monkeypatch.setattr(iv, "_get_tamer", lambda _: SimpleNamespace(process=lambda **_: None))
    monkeypatch.setattr(iv, "_safety_loops", {})
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")

    class Response:
        ok, status_code = True, 200
        def json(self):
            return {"choices": [{"message": {"content": "嗯，我在听。"}}]}
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def iter_lines(self, **_):
            yield ("data: " + json.dumps({"choices": [{"delta": {"content": "嗯，我在听。"}}]}, ensure_ascii=False)).encode()
            yield b"data: [DONE]"

    def post(*_, **kwargs):
        captured.append(kwargs["json"])
        return Response()
    monkeypatch.setattr(iv.requests, "post", post)
    app = FastAPI()
    app.include_router(iv.router, prefix="/api/v1")
    return TestClient(app), captured


def payload(**updates):
    return {"user_id": "support-http", "session_id": "conv", "message_id": "user-1",
            "message": "今天朋友没有回消息，我觉得他不想理我，有点难过。", **updates}


def get_state(client):
    response = client.post("/api/v1/intervention/smart-chat", json=payload())
    assert response.status_code == 200 and not response.json()["fallback"]
    return response.json()["support_state"]


def test_correction_then_next_http_prompt_uses_saved_meaning(harness):
    client, captured = harness
    state = get_state(client)
    thought = next(item for item in state["items"] if item["slot"] == "thought")
    req = {"session_id": "conv", "state": state, "feedback_id": "correction",
           "expected_revision": state["revision"], "action": "correct", "item_id": thought["id"],
           "text": "我担心自己被忽视，不是觉得他讨厌我"}
    corrected = client.post("/api/v1/intervention/support-feedback", json=req)
    assert corrected.status_code == 200
    new_state = corrected.json()["support_state"]
    result = client.post("/api/v1/intervention/smart-chat", json=payload(
        message_id="user-2", message="我想一起理清这件事", support_state=new_state,
        conversation_history=[{"role": "user", "content": payload()["message"], "id": "user-1"}]))
    assert not result.json()["fallback"]
    assert result.json()["support_state"]["goal"] == "understand"
    assert any(item["quote"] == req["text"] for item in result.json()["support_state"]["items"])
    system = captured[-1]["messages"][0]["content"]
    assert req["text"] in system and "不得再把被否认的旧理解当事实" in system
    assert "不证明对外部事实的判断" in system
    assert state["items"] != new_state["items"]


def test_sse_done_and_nonstream_share_support_contract(harness):
    client, captured = harness
    state = get_state(client)
    response = client.post("/api/v1/intervention/smart-chat/stream", json=payload())
    frames = response.text.split("\n\n")
    done = next(json.loads(line[6:]) for frame in frames if "event: done" in frame
                for line in frame.splitlines() if line.startswith("data: "))
    assert done["support_state"] == state and done["text"] == "嗯，我在听。"
    assert "本人可纠正的理解" in captured[-1]["messages"][0]["content"]


def test_business_innovation_in_both_chat_paths(harness, monkeypatch):
    from shared import InnovationObservation
    client, captured = harness
    service = iv.get_perception_service()
    service.analyze_innovation = lambda *args: InnovationObservation(available=True, probabilities=[.1, .2, .7])
    monkeypatch.setattr(iv, "get_perception_service", lambda: service)
    response = client.post("/api/v1/intervention/smart-chat", json=payload()).json()
    assert response["innovation"]["available"] and response["risk_level"] == "low"
    assert "本轮辅助情感参考" in captured[-1]["messages"][0]["content"]
    stream = client.post("/api/v1/intervention/smart-chat/stream", json=payload()).text
    done = next(json.loads(line[6:]) for frame in stream.split("\n\n") if "event: done" in frame
                for line in frame.splitlines() if line.startswith("data: "))
    assert done["innovation"] == response["innovation"]


def test_high_risk_does_not_call_auxiliary_experts(harness, monkeypatch):
    client, _ = harness
    service = iv.get_perception_service()
    service.analyze_innovation = lambda *args: pytest.fail("高风险必须先走原有升级")
    monkeypatch.setattr(iv, "get_perception_service", lambda: service)
    response = client.post("/api/v1/intervention/smart-chat", json=payload(message="我想自杀")).json()
    assert response["risk_level"] in ("high", "crisis")
    assert response["requires_escalation"] and response["innovation"] is None


@pytest.mark.parametrize("change,status", [({"session_id": "foreign"}, 409),
    ({"expected_revision": 999}, 409), ({"expected_revision": "1"}, 422), ({"action": "diagnose"}, 409)])
def test_http_rejects_foreign_stale_and_invalid_feedback(harness, change, status):
    client, _ = harness
    state = get_state(client)
    response = client.post("/api/v1/intervention/support-feedback", json={
        "session_id": "conv", "state": state, "feedback_id": "x", "expected_revision": state["revision"],
        "action": "set_goal", "goal": "listen", **change})
    assert response.status_code == status


def test_invalid_previous_state_cannot_enter_prompt(harness):
    client, captured = harness
    state = get_state(client)
    state["session_id"] = "foreign"
    response = client.post("/api/v1/intervention/smart-chat", json=payload(support_state=state))
    assert response.status_code == 200 and response.json()["support_state"] is None
    assert "本人可纠正的理解" not in captured[-1]["messages"][0]["content"]


@pytest.mark.parametrize("risk", ["high", "crisis"])
def test_support_never_overrides_existing_high_risk_escalation(harness, monkeypatch, risk):
    # ⚠️ 比赛 Demo 为模拟实现，生产环境需接入真实危机干预流程
    client, captured = harness
    import escalation.crisis as crisis
    escalations = []
    def escalate(alert):
        escalations.append(alert)
        return SimpleNamespace(status=SimpleNamespace(value="test-recorded"), alert_id="test")
    monkeypatch.setattr(crisis, "escalate", escalate)
    if risk == "high":
        monkeypatch.setattr(iv, "map_all", lambda _: {"test": SimpleNamespace(severity="重度")})
    else:
        monkeypatch.setattr(iv, "_detect_crisis", lambda _: True)
    response = client.post("/api/v1/intervention/smart-chat", json=payload())
    body = response.json()
    assert body["risk_level"] == risk and body["requires_escalation"] and not body["fallback"]
    assert body["support_state"] is None and escalations
    assert "本人可纠正的理解" not in captured[-1]["messages"][0]["content"]
