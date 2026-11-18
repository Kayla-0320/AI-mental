"""知识出处、范围、人工审核绑定、泄漏防护和真实标注对照的回归。"""
from __future__ import annotations
import copy
import json
from typing import Any
import numpy as np
import pytest
from experiments.round19_prepare import OUT
from experiments.round19_graph import GraphIndex,validate,fingerprint,conflicts,apply_reviews
from experiments.round19_encode import make_tasks,normalized,knowledge_channels
from experiments.round19_evaluate import ARMS
from experiments.round19_runtime import evidence_packet,predict_strategy


def graph() -> dict[str,Any]:
    """读本轮全部待审图谱，不生成真人记录。"""
    return json.loads((OUT/"knowledge_graph/graph.json").read_text(encoding="utf-8"))


def test_provenance_scope_and_default_pending() -> None:
    g=graph();assert not validate(g) and len(g["nodes"])==63 and len(g["edges"])==56
    assert {"WHO2020","UNICEF_TALK","UNICEF_11_13","ACL2021"}<=set(g["sources"])
    for e in g["edges"]:
        assert e["locator"] and e["scope"] and e["source_sha256"]==g["sources"][e["source_id"]]["document_sha256"]
    assert all(x["review_status"]=="pending" for x in g["nodes"]+g["edges"])


@pytest.mark.parametrize("kwargs",[{},{"research":True},{"include_unreviewed":True}])
def test_pending_requires_two_explicit_research_flags(kwargs: dict[str,bool]) -> None:
    assert not GraphIndex(graph()).retrieve("锚定当下","understand",**kwargs)["facts"]


def test_research_results_keep_evidence_separate_from_user_facts() -> None:
    g=graph();before=copy.deepcopy(g);r=GraphIndex(g).retrieve("锚定当下","understand",research=True,include_unreviewed=True)
    assert any(x["node_id"]=="grounding" for x in r["facts"])
    assert all(not x["applied_to_user"] and not x["human_reviewed"] for x in r["facts"])
    assert g==before


def test_parent_child_guidance_not_silently_treated_as_AI_efficacy() -> None:
    index=GraphIndex(graph());kwargs={"research":True,"include_unreviewed":True}
    assert not index.retrieve("认真倾听与问我需要什么","listen",setting="general_concept",**kwargs)["facts"]
    results=index.retrieve("认真倾听与问我需要什么","listen",setting="parent_child",**kwargs)["facts"]
    assert results and all(x["source_setting"]=="parent_child" for x in results)
    assert all(e["scope"]["setting"]=="parent_child" for x in results for e in x["facts"])


@pytest.mark.parametrize("query,goal",[("锚定当下","pause"),("锚定当下","unknown"),("我不想锚定当下","act"),("他说“锚定当下”","understand"),("朋友说锚定当下","act"),("锚定当下能诊断吗","act")])
def test_quoted_negated_and_paused_queries_do_not_activate_grounding(query: str,goal: str) -> None:
    r=GraphIndex(graph()).retrieve(query,goal,research=True,include_unreviewed=True)
    assert "grounding" not in {x["node_id"] for x in r["facts"]}


def test_latest_direct_negation_overrides_prior_positive_mention() -> None:
    r=GraphIndex(graph()).retrieve("我想了解锚定当下，我现在不想锚定当下","act",research=True,include_unreviewed=True)
    assert "grounding" not in {x["node_id"] for x in r["facts"]}


@pytest.mark.parametrize("corruption",["source","content","dangling","review"])
def test_mutated_or_unbound_graph_rejected(corruption: str) -> None:
    g=graph()
    if corruption=="source":g["edges"][0]["source_sha256"]="0"*64
    if corruption=="content":g["nodes"][0]["definition"]="changed"
    if corruption=="dangling":g["edges"][0]["object"]="missing"
    if corruption=="review":g["nodes"][0].update(review_status="approved",reviewer_id="fixture",reviewed_at="2026-11-17",review_method="model")
    assert validate(g)
    with pytest.raises(ValueError):GraphIndex(g)


@pytest.mark.parametrize("change",["model","stale","blank","duplicate"])
def test_review_import_rejects_model_or_unbound_records(change: str) -> None:
    g=graph();item=g["nodes"][0];row={"item_id":item["id"],"content_sha256":item["content_sha256"],"decision":"approved","review_method":"human","reviewer_id":"fixture_not_a_real_review","reviewed_at":"2026-11-17T00:00:00Z","qualification_basis":"synthetic_test_only"}
    if change=="model":row["review_method"]="model"
    if change=="stale":row["content_sha256"]="0"*64
    if change=="blank":row["reviewer_id"]=""
    rows=[row,row] if change=="duplicate" else [row]
    with pytest.raises(ValueError):apply_reviews(g,rows)


def test_conflicting_same_scope_claims_are_withheld_without_majority_voting() -> None:
    g=graph();edge=copy.deepcopy(next(e for e in g["edges"] if e["subject"]=="grounding"));edge.update(id="TEST_ONLY_OPPOSITE",polarity="negative");edge["content_sha256"]=fingerprint(edge);g["edges"].append(edge)
    assert not validate(g);bad={x for c in conflicts(g) for x in c};assert edge["id"] in bad
    r=GraphIndex(g).retrieve("锚定当下","understand",research=True,include_unreviewed=True)
    assert all(e["id"] not in bad for x in r["facts"] for e in x["facts"])
    edge["scope"]={"setting":"different_source_setting","population":"source_only"};edge["content_sha256"]=fingerprint(edge);assert not conflicts(g)


def test_budget_preserves_whole_citations_and_default_abstention() -> None:
    assert evidence_packet("锚定当下","understand")["status"]=="abstain"
    r=evidence_packet("锚定当下","understand",research=True,include_unreviewed=True,budget=4000)
    assert r["evidence"] and r["encoded_evidence_chars"]<=4000
    for block in r["evidence"]:
        assert block["not_a_user_fact"] and all(c["edge_id"] and c["url"] and c["locator"] and len(c["source_sha256"])==64 for c in block["citations"])


def test_next_strategy_input_excludes_target_and_annotation() -> None:
    d={"dialog":[{"speaker":"seeker","content":"past user text","annotation":{}},{"speaker":"supporter","content":"FUTURE_TARGET_SENTINEL","annotation":{"strategy":"Question"}}]}
    tasks,_=make_tasks([d]);rows=sum(tasks["next_strategy"].values(),[])
    assert len(rows)==1 and "FUTURE_TARGET_SENTINEL" not in rows[0]["text"] and "Question" not in rows[0]["text"]
    assert rows[0]["input_max_index"]<rows[0]["target_index"]


@pytest.mark.parametrize("task",["attribution","next_strategy"])
def test_actual_dialogue_and_input_splits_disjoint(task: str) -> None:
    folder=OUT/"benchmark"/task;sets={};groups={}
    for split in ("train","validation","test"):
        rows=json.loads((folder/(split+".json")).read_text(encoding="utf-8"));sets[split]={normalized(r["text"]) for r in rows};groups[split]={r["dialogue"] for r in rows}
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        assert not sets[a]&sets[b] and not groups[a]&groups[b]


def test_graph_channels_are_rebuilt_without_labels_and_keep_control_capacity() -> None:
    g=graph();v=np.load(OUT/"benchmark/graph_vectors.npz",allow_pickle=False);z=np.load(OUT/"benchmark/attribution/test.npz",allow_pickle=False)
    channels=knowledge_channels(z["embedding"],v["vectors"],g,v["node_ids"].tolist())
    for arm in ARMS:np.testing.assert_allclose(channels[arm],z[arm],atol=1e-6)
    assert np.all(channels["no_graph"]==0)
    assert not np.allclose(channels["true_graph"],channels["definitions_only"])
    assert not np.allclose(channels["true_graph"],channels["shuffled_graph"])


@pytest.mark.parametrize("task",["attribution","next_strategy"])
def test_selection_capacity_and_runtime_exact(task: str) -> None:
    folder=OUT/"benchmark"/task;lock=json.loads((folder/"selection_lock.json").read_text(encoding="utf-8"))
    assert not lock["test_labels_used"] and len({r["features"] for r in lock["arms"].values()})==1
    assert all(len(r["candidates"])==2 for r in lock["arms"].values())
    z=np.load(folder/"test.npz",allow_pickle=False);stored=np.load(folder/"predictions.npz",allow_pickle=False);rows=json.loads((folder/"test.json").read_text(encoding="utf-8"))
    for arm in ARMS:
        np.testing.assert_allclose(predict_strategy([r["text"] for r in rows],z["embedding"],z[arm],folder,arm),stored[arm],atol=1e-12)


def test_no_model_generated_human_ratings_or_production_changes() -> None:
    r=json.loads((OUT/"verification.json").read_text(encoding="utf-8"))
    assert r["expert_reviews"]==r["project_human_blind_ratings"]==0
    assert not r["production_connected"] and r["parent_round18_graph_unchanged"] and all(r["protected_files_unchanged"].values())


def test_jsonld_reifies_all_relationships_in_real_RDF_parser() -> None:
    from rdflib import Graph,URIRef
    data=(OUT/"knowledge_graph/graph.jsonld").read_text(encoding="utf-8");rdf=Graph().parse(data=data,format="json-ld")
    assert sum(1 for _ in rdf.triples((None,URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"),None)))==56
