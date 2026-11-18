"""可追溯的压力管理概念图谱初版；非专家背书、非诊断图谱、离线研究专用。"""
from __future__ import annotations
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import numpy as np
import requests
from experiments.round14_common import ROOT,save,sha256

OUT=ROOT/"round18/knowledge_graph"
WHO="https://cdn.who.int/media/docs/default-source/thailand/doing-what-matters-in-times-of-stress-eng.pdf"
SKOS="https://www.w3.org/TR/skos-reference/"


def build() -> dict[str,Any]:
    """区分指南概念关系与工程契约，每条边携带来源、定位、限制和待审核状态。"""
    nodes=[];edges=[]
    def node(identifier: str,label: str,kind: str,aliases: list[str]) -> None:
        nodes.append({"id":identifier,"label":label,"kind":kind,"aliases":aliases,"review_status":"pending","reviewer_id":None,"reviewed_at":None})
    node("guide","压力管理概念","concept_scheme",["压力管理"])
    skills=[("grounding","锚定当下",["锚定","grounding","回到当下"]), ("unhooking","与想法解钩",["解钩","unhooking"]),
        ("values","按重视的方向行动",["价值观","重视的方向","values"]),("kindness","善待自己与他人",["善待","友善","kindness"]),("making_room","给困难感受留出空间",["留出空间","making room"])]
    for identifier,label,aliases in skills:node(identifier,label,"public_guideline_concept",aliases)
    details=[("notice","觉察",["觉察","注意到"]),("body","身体感受",["身体感受"]),("engage","关注当前活动",["当前活动"]),
        ("name","给体验命名",["命名"]),("thoughts","想法",["想法"]),("feelings","感受",["感受"]),("small_action","小行动",["小行动"]),
        ("self","自己",["自己"]),("others","他人",["他人"]),("space","容纳体验",["容纳体验"])]
    for identifier,label,aliases in details:node(identifier,label,"public_guideline_step",aliases)
    for identifier,label in (("listen","倾听目标"),("understand","梳理目标"),("act","行动目标"),("pause","暂停目标"),
        ("user_quote","用户原话"),("correction","本人修正"),("confirmed","本人确认"),("unknown","尚未核实"),
        ("negated","被否定的陈述"),("quoted","引述"),("hypothetical","假设"),("current","当前语境")):
        node(identifier,label,"project_contract",[label])
    def edge(subject: str,predicate: str,obj: str,source: str,locator: str,kind: str) -> None:
        edges.append({"id":f"E{len(edges)+1:03d}","subject":subject,"predicate":predicate,"object":obj,"source_id":source,"locator":locator,
            "evidence_type":kind,"review_status":"pending","reviewer_id":None,"reviewed_at":None,
            "scope":"concept description only; not evidence about this user or treatment efficacy","limitations":["no_diagnosis","no_individual_causal_inference","no_expert_review_yet"]})
    for i,(identifier,_,_) in enumerate(skills):edge(identifier,"part_of","guide","WHO2020",f"Tool{i+1}, printed p{122+i}, PDF zero-based index{123+i}","public_guideline")
    relationships=[("grounding","notice",122),("grounding","body",122),("grounding","engage",122),("unhooking","notice",123),("unhooking","name",123),
        ("unhooking","engage",123),("values","small_action",124),("kindness","self",125),("kindness","others",125),("making_room","feelings",126),("making_room","space",126)]
    for subject,obj,page in relationships:edge(subject,"has_step",obj,"WHO2020",f"Tool summary, printed p{page}, PDF zero-based index{page+1}","public_guideline")
    for subject,obj in (("notice","thoughts"),("notice","feelings")):edge(subject,"related",obj,"WHO2020","Tool1, printed p122; Tool2, printed p123","public_guideline")
    for skill,_,_ in skills:edge(skill,"requires_explicit_goal","act","PROJECT","offline action eligibility contract v1; not a WHO claim","engineering_contract")
    for subject,obj in (("correction","user_quote"),("confirmed","user_quote"),("unknown","user_quote"),("negated","current"),("quoted","current"),("hypothetical","current")):
        edge(subject,"grounded_in",obj,"PROJECT","source-bound evidence contract; uncertainty is retained","engineering_contract")
    return {"schema":"aic.support_concepts.v1","version":"0.1.0","status":"literature_sourced_engineering_draft_pending_expert_review","sources":{
        "WHO2020":{"url":WHO,"title":"Doing what matters in times of stress: an illustrated guide","publisher":"WHO","year":2020,"license":"CC BY-NC-SA3.0IGO","endorsement":False},
        "PROJECT":{"url":"project://source-bound-support-contract-v1","title":"工程支持目标与出处约束","license":"project research contract","not_external_evidence":True},
        "SKOS":{"url":SKOS,"title":"SKOS Reference; concept labels and associative relations only"}},
        "nodes":nodes,"edges":edges,"graph_data_license":"CC BY-NC-SA3.0IGO for guideline-derived concept adaptation",
        "translation_notice":"中文术语为工程整理，非WHO官方译本；以英文原版为准；WHO不为本项目背书。","platform_connected":False}


def validate(graph: dict[str,Any]) -> list[str]:
    """检查引用、来源和审核字段；不存在诊断、个体因果或疗效推断关系。"""
    errors=[];ids=[n["id"] for n in graph["nodes"]]
    if len(ids)!=len(set(ids)):errors.append("duplicate_node")
    edge_ids=[e["id"] for e in graph["edges"]]
    if len(edge_ids)!=len(set(edge_ids)):errors.append("duplicate_edge")
    for item in graph["nodes"]+graph["edges"]:
        if item.get("review_status") not in {"pending","approved","rejected"}:errors.append("invalid_review_status")
        if item.get("review_status")=="approved" and (not item.get("reviewer_id") or not item.get("reviewed_at")):errors.append("false_approval")
    permitted={"part_of","has_step","related","requires_explicit_goal","grounded_in"}
    for edge in graph["edges"]:
        if edge["subject"] not in ids or edge["object"] not in ids:errors.append("dangling_edge")
        if edge["predicate"] not in permitted:errors.append("unsupported_relation")
        if edge["source_id"] not in graph["sources"] or not edge.get("locator"):errors.append("missing_provenance")
        if edge["review_status"]=="approved" and (not edge.get("reviewer_id") or not edge.get("reviewed_at")):errors.append("false_approval")
    return sorted(set(errors))


def retrieve(graph: dict[str,Any],query: str,goal: str,*,research: bool=False,include_unreviewed: bool=False) -> dict[str,Any]:
    """默认拒绝未审核内容；研究模式也只返回有出处概念，不生成个性化支持结论。"""
    if validate(graph):raise ValueError("图谱结构或来源不合法")
    if goal in {"pause","listen"}:return {"status":"abstain","reason":"respect_explicit_goal","facts":[]}
    if goal not in {"act","understand"}:return {"status":"abstain","reason":"goal_unknown","facts":[]}
    if re.search(r"诊断|确诊|治愈|治好|药物|用药|药量",query):return {"status":"abstain","reason":"outside_concept_graph_scope","facts":[]}
    clean=re.sub(r'“[^”]*”|「[^」]*」|"[^"\n]*"','',query)
    facts=[];by_id={n["id"]:n for n in graph["nodes"]}
    for node in graph["nodes"]:
        if node["kind"]!="public_guideline_concept":continue
        matches=[m for alias in node["aliases"] for m in re.finditer(re.escape(alias),clean,re.I)]
        direct=[m for m in matches if not re.search(r"不想|不要|不用|拒绝|朋友说|他说|她说",clean[max(0,m.start()-10):m.start()])]
        if not direct:continue
        approved=node["review_status"]=="approved"
        if node["review_status"]=="rejected" or (not approved and not (research and include_unreviewed)):continue
        for edge in graph["edges"]:
            if edge["subject"]!=node["id"] or edge["evidence_type"]!="public_guideline":continue
            if edge["review_status"]=="rejected":continue
            if edge["review_status"]!="approved" and not (research and include_unreviewed):continue
            target=by_id[edge["object"]]
            if target["review_status"]=="rejected" or (target["review_status"]!="approved" and not (research and include_unreviewed)):continue
            facts.append({**edge,"concept_label":node["label"],"source_url":graph["sources"][edge["source_id"]]["url"],"applied_to_user":False,"human_reviewed":approved and edge["review_status"]=="approved"})
    return {"status":"research_concept_candidates" if facts else "abstain","facts":facts,"not_personalized_advice":True,"clinical_inference":False}


def jsonld(graph: dict[str,Any]) -> dict[str,Any]:
    """导出具有明确RDF谓词映射的带出处关系，不让JSON-LD处理器丢弃未知键。"""
    return {"@context":{"skos":"http://www.w3.org/2004/02/skos/core#","rdf":"http://www.w3.org/1999/02/22-rdf-syntax-ns#","dct":"http://purl.org/dc/terms/","aic":"urn:aic:kg:",
        "label":"skos:prefLabel","subject":{"@id":"rdf:subject","@type":"@id"},"predicate":{"@id":"rdf:predicate","@type":"@id"},"object":{"@id":"rdf:object","@type":"@id"},
        "source":{"@id":"dct:source","@type":"@id"},"source_url":{"@id":"aic:sourceURL","@type":"@id"},"locator":"aic:sourceLocator","review_status":"aic:reviewStatus"},
        "@graph":[{"@id":"urn:aic:kg:"+n["id"],"@type":"skos:ConceptScheme" if n["kind"]=="concept_scheme" else "skos:Concept","label":{"@value":n["label"],"@language":"zh"},"review_status":n["review_status"]} for n in graph["nodes"]]+[
            {"@id":"urn:aic:edge:"+e["id"],"@type":"rdf:Statement","subject":"urn:aic:kg:"+e["subject"],"predicate":"urn:aic:relation:"+e["predicate"],"object":"urn:aic:kg:"+e["object"],"source":"urn:aic:source:"+e["source_id"],"locator":e["locator"],"review_status":e["review_status"]} for e in graph["edges"]]+[
            {"@id":"urn:aic:source:"+k,"label":s["title"],"source_url":s["url"]} for k,s in graph["sources"].items()]}


def run() -> None:
    """写出JSON/JSON-LD/关系表及空审核表，执行关系恢复负对照而非真人质量评分。"""
    OUT.mkdir(parents=True,exist_ok=True);graph=build();assert not validate(graph)
    response=requests.get(WHO,timeout=(20,120));response.raise_for_status()
    graph["sources"]["WHO2020"]["download_sha256"]=hashlib.sha256(response.content).hexdigest();graph["sources"]["WHO2020"]["bytes"]=len(response.content)
    save(OUT/"graph.json",graph)
    save(OUT/"graph.jsonld",jsonld(graph))
    for filename,rows in (("edges.csv",graph["edges"]),("nodes.csv",graph["nodes"])):
        with (OUT/filename).open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/"expert_review_pending.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        fields=["item_id","item_type","review_status","human_expert_id","reviewed_at","qualification_basis","comments"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({"item_id":item["id"],"item_type":kind,"review_status":"pending"} for kind,rows in (("node",graph["nodes"]),("edge",graph["edges"])) for item in rows)
    control=copy.deepcopy(graph);targets=[e["object"] for e in control["edges"]];order=np.random.default_rng(42).permutation(len(targets))
    for i,e in enumerate(control["edges"]):e["object"]=targets[int(order[i])]
    queries=[("锚定当下有哪些概念步骤","grounding"),("解钩包含什么","unhooking"),("价值观与行动的关系","values"),("善待涉及谁","kindness"),("留出空间的概念","making_room")]
    result=[]
    for query,subject in queries:
        gold={(e["predicate"],e["object"]) for e in graph["edges"] if e["subject"]==subject and e["evidence_type"]=="public_guideline"}
        true=retrieve(graph,query,"understand",research=True,include_unreviewed=True)["facts"]
        shuffled=retrieve(control,query,"understand",research=True,include_unreviewed=True)["facts"]
        result.append({"query":query,"true_relation_hits":len(gold&{(e["predicate"],e["object"]) for e in true}),"shuffled_relation_hits":len(gold&{(e["predicate"],e["object"]) for e in shuffled}),"no_graph_relation_hits":0,"expected_relations":len(gold)})
    save(OUT/"retrieval_controls.json",{"rows":result,"scope":"author-defined relationship recovery and intentional graph corruption; not independent QA accuracy, model benefit or human evaluation","shuffled_graph_served":False})
    save(OUT/"status.json",{"nodes":len(graph["nodes"]),"edges":len(graph["edges"]),"source_backed_edges":sum(e["evidence_type"]=="public_guideline" for e in graph["edges"]),
        "engineering_edges":sum(e["evidence_type"]=="engineering_contract" for e in graph["edges"]),"human_expert_reviews":0,"default_serving_unreviewed":False,"production_connected":False})
    print(json.dumps({"nodes":len(graph["nodes"]),"edges":len(graph["edges"]),"expert_reviews":0}),flush=True)


if __name__=="__main__":run()
