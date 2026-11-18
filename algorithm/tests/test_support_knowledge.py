"""知识图谱只读接口回归：默认拒绝待审、研究显式开启、场景隔离与出处完整。"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from perception.support_knowledge import capabilities, load


def test_manifest_matches_round19_artifact() -> None:
    info = capabilities()
    assert info["graph_schema"] == "aic.support_concepts.v2"
    assert info["nodes"] == 63 and info["edges"] == 56
    assert info["pending_items"] == 119
    assert info["human_expert_reviews"] == 0
    assert info["default_unreviewed_allowed"] is False
    assert info["production_connected"] is False


def test_default_request_abstains_on_pending_items() -> None:
    result = load().evidence("锚定当下是什么", "understand")
    assert result["status"] == "abstain"
    assert result["reason"] == "pending_review"
    assert result["evidence"] == []
    assert result["not_personalized_advice"] is True


def test_research_opt_in_returns_concept_with_citations() -> None:
    result = load().evidence("锚定当下是什么", "understand", research=True, include_unreviewed=True)
    assert result["status"] == "evidence"
    block = result["evidence"][0]
    assert block["review_status"] == "pending" and block["not_a_user_fact"] is True
    assert block["facts"]
    assert all(fact["url"].startswith("https://") and fact["locator"] and fact["source_sha256"] for fact in block["facts"])
    assert 0 < result["encoded_evidence_chars"] <= 4000


def test_parent_child_scope_is_not_returned_in_general_setting() -> None:
    index = load()
    general = index.evidence("认真倾听与问我需要什么", "listen", research=True, include_unreviewed=True)
    assert general["status"] == "abstain"
    parent = index.evidence("认真倾听与问我需要什么", "listen", setting="parent_child",
                            research=True, include_unreviewed=True)
    assert parent["status"] == "evidence"
    assert {block["concept"] for block in parent["evidence"]} & {"主动倾听", "询问回应偏好"}


def test_pause_negation_and_out_of_scope_abstain() -> None:
    index = load()
    assert index.evidence("锚定当下", "pause", research=True, include_unreviewed=True)["reason"] == "explicit_goal"
    assert index.evidence("我不想锚定当下", "understand", research=True, include_unreviewed=True)["status"] == "abstain"
    assert index.evidence("我是不是抑郁症，要不要吃药", "understand",
                          research=True, include_unreviewed=True)["reason"] == "outside_concept_scope"


def test_budget_never_truncates_a_block() -> None:
    result = load().evidence("锚定当下", "understand", research=True, include_unreviewed=True, budget=100)
    assert result["status"] == "abstain" or result["encoded_evidence_chars"] <= 100


def test_http_route_is_disabled_by_default(monkeypatch) -> None:
    import api.knowledge as api

    monkeypatch.delenv("SUPPORT_KNOWLEDGE_ENABLED", raising=False)
    app = FastAPI(); app.include_router(api.router, prefix="/api/v1")
    client = TestClient(app)
    assert client.get("/api/v1/knowledge/capabilities").json()["enabled"] is False
    body = {"query": "锚定当下", "goal": "understand", "research": True, "include_unreviewed": True}
    assert client.post("/api/v1/knowledge/evidence", json=body).status_code == 503


def test_http_route_returns_evidence_when_enabled(monkeypatch) -> None:
    import api.knowledge as api

    monkeypatch.setenv("SUPPORT_KNOWLEDGE_ENABLED", "1")
    app = FastAPI(); app.include_router(api.router, prefix="/api/v1")
    body = {"query": "锚定当下", "goal": "understand", "research": True, "include_unreviewed": True}
    response = TestClient(app).post("/api/v1/knowledge/evidence", json=body)
    assert response.status_code == 200
    assert response.json()["status"] == "evidence"
    assert response.json()["production_connected"] is False
