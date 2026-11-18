"""图谱边界、来源指纹、原文件保护和策略模型逐行回放的独立核验。"""
from __future__ import annotations
import json
import hashlib
import html
import re
from pathlib import Path
from typing import Any
import numpy as np
from experiments.round14_common import ROOT,REPO,save,sha256
from experiments.round19_prepare import OUT,DATA
from experiments.round19_graph import validate,GraphIndex,conflicts
from experiments.round19_encode import knowledge_channels,normalized
from experiments.round19_evaluate import ARMS,metrics
from experiments.round19_runtime import predict_strategy,evidence_packet


def verify() -> dict[str,Any]:
    """只读回放所有测试行并验证知识通道重建，不改变锁型、图谱和评测参数。"""
    graphpath=OUT/"knowledge_graph/graph.json";graph=json.loads(graphpath.read_text(encoding="utf-8"));assert not validate(graph)
    index=GraphIndex(graph);assert not index.retrieve("锚定当下","understand")["facts"]
    assert not conflicts(graph);vectors=np.load(OUT/"benchmark/graph_vectors.npz",allow_pickle=False);replays=[]
    for task in ("attribution","next_strategy"):
        folder=OUT/"benchmark"/task;input_sets={};all_groups={}
        for split in ("train","validation","test"):
            rows=json.loads((folder/(split+".json")).read_text(encoding="utf-8"));input_sets[split]={normalized(r["text"]) for r in rows};all_groups[split]={r["dialogue"] for r in rows}
            if task=="next_strategy":assert all(r["input_max_index"]<r["target_index"] for r in rows)
        assert not (input_sets["train"]&input_sets["test"] or input_sets["validation"]&input_sets["test"] or input_sets["train"]&input_sets["validation"])
        assert not (all_groups["train"]&all_groups["test"] or all_groups["validation"]&all_groups["test"] or all_groups["train"]&all_groups["validation"])
        z=np.load(folder/"test.npz",allow_pickle=False);stored=np.load(folder/"predictions.npz",allow_pickle=False);rows=json.loads((folder/"test.json").read_text(encoding="utf-8"));channels=knowledge_channels(z["embedding"],vectors["vectors"],graph,vectors["node_ids"].tolist())
        for arm in ARMS:
            assert np.max(np.abs(channels[arm]-z[arm]))<1e-6
            score=predict_strategy([r["text"] for r in rows],z["embedding"],z[arm],folder,arm);error=float(np.max(np.abs(score-stored[arm])));assert error<1e-12
            replays.append({"task":task,"arm":arm,"rows":len(rows),"max_error":error,"metrics":metrics(z["labels"],score)})
    boundary_queries=[("锚定当下是什么","understand","general_concept"),("我不要锚定当下","act","general_concept"),("他说“锚定当下”","act","general_concept"),("我想先停一下","pause","general_concept"),("认真倾听与问我需要什么","listen","parent_child"),("认真倾听与问我需要什么","listen","general_concept")]
    boundary=[{"query":q,"goal":goal,"setting":setting,"retrieved_ids":[e["node_id"] for e in index.retrieve(q,goal,research=True,include_unreviewed=True,setting=setting)["facts"]]} for q,goal,setting in boundary_queries]
    source_status=json.loads((OUT/"workspace_baseline.json").read_text(encoding="utf-8"))["protected_sha256"]
    unchanged={p:sha256(REPO/p)==digest for p,digest in source_status.items()};assert all(unchanged.values())
    old_manifest=json.loads((ROOT/"round18/manifest.json").read_text(encoding="utf-8"));parent_files=["knowledge_graph/graph.json","knowledge_graph/graph.jsonld","knowledge_graph/expert_review_pending.csv"]
    assert all(sha256(ROOT/"round18"/p)==old_manifest[p]["sha256"] for p in parent_files)
    packet=evidence_packet("锚定当下的概念","understand",research=True,include_unreviewed=True,budget=4000);assert packet["encoded_evidence_chars"]<=4000
    anchors={"UNICEF_TALK":["listen more than you talk","ask their permission"],"UNICEF_11_13":["dismiss or minimize","respect and encourage","offer solutions"]};source_checks=[]
    for source,fragments in anchors.items():
        raw=(DATA/(source+".html")).read_text(encoding="utf-8");text=html.unescape(re.sub(r"<[^>]+>"," ",raw));text=re.sub(r"\s+"," ",text).casefold()
        assert sha256(DATA/(source+".html"))==graph["sources"][source]["document_sha256"]
        for fragment in fragments:
            assert fragment in text
            source_checks.append({"source":source,"locator_fragment_sha256":hashlib.sha256(fragment.encode()).hexdigest(),"found":True,"not_expert_semantic_review":True})
    from rdflib import Graph,URIRef
    rdf=Graph().parse(data=(OUT/"knowledge_graph/graph.jsonld").read_text(encoding="utf-8"),format="json-ld")
    relation_count=sum(1 for _ in rdf.triples((None,URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"),None)));assert relation_count==len(graph["edges"])
    result={"replays":replays,"graph_nodes":len(graph["nodes"]),"graph_edges":len(graph["edges"]),"rdf_relationships":relation_count,"source_anchor_checks":source_checks,"default_pending_abstains":True,"parent_round18_graph_unchanged":True,"protected_files_unchanged":unchanged,"boundary_functional_probes_not_human_gold":boundary,"expert_reviews":0,"project_human_blind_ratings":0,"production_connected":False}
    save(OUT/"verification.json",result);return result


if __name__=="__main__":print(json.dumps(verify()),flush=True)
