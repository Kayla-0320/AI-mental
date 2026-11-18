"""多来源、带范围和内容指纹的概念图谱；只供离线研究，不承诺专业审核或个体效果。"""
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
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round18_graph import jsonld
from experiments.round19_prepare import OUT,DATA

STRATEGIES=[
    ("question","Question","询问","Ask about the situation or feelings with an open or specific question.",["询问","提问","了解情况"],["situation details","open question"]),
    ("paraphrase","Restatement or Paraphrasing","复述","Restate the seeker's situation concisely in different words.",["复述","换句话说","澄清事实"],["concise restatement","rephrased situation"]),
    ("reflection","Reflection of feelings","感受反映","Name or reflect feelings expressed by the seeker.",["感受反映","情感反映","表达理解"],["named emotion","reflected feeling"]),
    ("disclosure","Self-disclosure","自我披露","Share a relevant personal experience or similar feeling.",["自我披露","个人经历"],["personal experience","shared experience"]),
    ("affirmation","Affirmation and Reassurance","肯定与鼓励","Recognize strengths and offer reassurance about coping.",["肯定","鼓励","安慰"],["recognized strengths","encouraging reassurance"]),
    ("suggestions","Providing Suggestions","提供可选建议","Offer an optional action concerning the described problem.",["提供建议","可选建议","小步骤"],["optional action","practical suggestion"]),
    ("information","Information","信息支持","Provide relevant facts, resources or information.",["信息支持","资源信息","事实信息"],["relevant facts","available resources"]),
    ("others","Others","其他交谈功能","Use greetings, thanks, closings or other dialogue functions.",["寒暄","感谢","结束交谈"],["greeting thanks","conversation closing"]),
]
PERMITTED={"part_of","has_step","related","requires_explicit_goal","grounded_in","has_facet","discusses","recommends","discourages","source_context"}
MUTABLE={"review_status","reviewer_id","reviewed_at","review_method","review_attestation","content_sha256"}


def fingerprint(item: dict[str,Any]) -> str:
    """审核只绑定当前内容；来源、关系或范围变化都会使旧审核指纹失效。"""
    return hashlib.sha256(json.dumps({k:v for k,v in item.items() if k not in MUTABLE},sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def conflicts(graph: dict[str,Any]) -> list[list[str]]:
    """只有同一范围、同一命题正反冲突才报冲突，不把不同年龄/场景差异直接当矛盾。"""
    buckets={}
    for e in graph["edges"]:
        if e["review_status"]=="rejected":continue
        key=json.dumps([e["subject"],e["predicate"],e["object"],e["scope"]],sort_keys=True,ensure_ascii=False)
        buckets.setdefault(key,[]).append(e)
    return [[e["id"] for e in rows] for rows in buckets.values() if len({e["polarity"] for e in rows})>1]


def validate(graph: dict[str,Any]) -> list[str]:
    """核验结构、证据、审核绑定和版本，不将软件检查叫作专家审核。"""
    errors=[]
    if graph.get("schema")!="aic.support_concepts.v2":errors.append("schema")
    ids=[n["id"] for n in graph["nodes"]];eids=[e["id"] for e in graph["edges"]]
    if len(ids)!=len(set(ids)) or len(eids)!=len(set(eids)):errors.append("duplicate_id")
    for n in graph["nodes"]:
        if not n.get("aliases") or any(not isinstance(a,str) or not a.strip() for a in n["aliases"]):errors.append("alias")
    for e in graph["edges"]:
        if e["subject"] not in ids or e["object"] not in ids:errors.append("dangling")
        if e["predicate"] not in PERMITTED or e["polarity"] not in {"positive","negative"}:errors.append("relation")
        source=graph["sources"].get(e.get("source_id"))
        if not source or not e.get("locator") or not e.get("scope") or not e.get("source_sha256"):errors.append("provenance")
        elif source["document_sha256"]!=e["source_sha256"]:errors.append("source_changed")
    for item in graph["nodes"]+graph["edges"]:
        if item.get("review_status") not in {"pending","approved","rejected"}:errors.append("review_state")
        if item.get("content_sha256")!=fingerprint(item):errors.append("content_changed")
        if item.get("review_status")=="approved":
            att=item.get("review_attestation",{})
            if not item.get("reviewer_id") or not item.get("reviewed_at") or item.get("review_method")!="human" or att.get("content_sha256")!=fingerprint(item) or not att.get("qualification_basis"):errors.append("unbound_or_nonhuman_review")
    return sorted(set(errors))


def build() -> dict[str,Any]:
    """继承十八轮并补充家长沟通原则和八类策略定义，所有新旧内容仍为待审。"""
    base=json.loads((ROOT/"round18/knowledge_graph/graph.json").read_text(encoding="utf-8"));locks=json.loads((OUT/"source_lock.json").read_text(encoding="utf-8"))
    g=copy.deepcopy(base);g.update(schema="aic.support_concepts.v2",version="0.2.0",parent_sha256=sha256(ROOT/"round18/knowledge_graph/graph.json"),status="multi_source_draft_pending_human_expert_review")
    for key,s in g["sources"].items():s["document_sha256"]=s.get("download_sha256",hashlib.sha256(s["url"].encode()).hexdigest())
    for key in ("UNICEF_TALK","UNICEF_11_13"):
        g["sources"][key]={"url":locks[key]["url"],"document_sha256":locks[key]["sha256"],"title":key,"publisher":"UNICEF","license":"copyright UNICEF; brief engineering paraphrases and links only","setting":"parent_child","population":"11_13" if key.endswith("11_13") else "teen_family","clinical_efficacy_evidence":False}
    paper=DATA/"ACL2021.pdf"
    if not paper.exists():
        r=requests.get("https://aclanthology.org/2021.acl-long.269.pdf",timeout=(15,60));r.raise_for_status();paper.write_bytes(r.content)
    g["sources"]["ACL2021"]={"url":"https://aclanthology.org/2021.acl-long.269.pdf","document_sha256":sha256(paper),"title":"Towards Emotional Support Dialog Systems","publisher":"ACL","year":2021,"license":"original ACL terms; brief definition paraphrases only","setting":"English_crowd_support","population":"published_crowdworkers","clinical_efficacy_evidence":False}
    for n in g["nodes"]:
        n.update(definition=n["label"],source_scope="general_concept",requires_goal="act" if n["kind"]=="public_guideline_concept" else "any",not_evidence_about_user=True)
    for e in g["edges"]:
        e.update(polarity="positive",scope={"setting":"engineering" if e["source_id"]=="PROJECT" else "general_stress_concept","population":"source_only"},source_sha256=g["sources"][e["source_id"]]["document_sha256"],derivation="inherited_round18",version="0.2.0")
    def node(identifier: str,label: str,kind: str,aliases: list[str],definition: str,scope: str="general_concept",goal: str="any") -> None:
        g["nodes"].append({"id":identifier,"label":label,"kind":kind,"aliases":aliases,"definition":definition,"source_scope":scope,"requires_goal":goal,"not_evidence_about_user":True,"review_status":"pending","reviewer_id":None,"reviewed_at":None})
    def edge(subject: str,predicate: str,obj: str,source: str,locator: str,scope: str,derivation: str="source_paraphrase") -> None:
        g["edges"].append({"id":f"R19E{len(g['edges'])+1:03d}","subject":subject,"predicate":predicate,"object":obj,"source_id":source,"source_sha256":g["sources"][source]["document_sha256"],"locator":locator,"scope":{"setting":scope,"population":g["sources"][source].get("population","source_only")},"polarity":"positive","derivation":derivation,"evidence_type":"engineering_decomposition" if derivation.startswith("engineering") else "source_concept","version":"0.2.0","review_status":"pending","reviewer_id":None,"reviewed_at":None,"limitations":["not_personalized_advice","no_treatment_efficacy_claim"]})
    for identifier,label,aliases,text,goal in (
        ("active_listening","主动倾听",["认真听","倾听","active listening"],"以倾听了解孩子的表达，避免急于主导交谈。","any"),
        ("ask_preference","询问回应偏好",["问我需要什么","先问意愿","回应偏好","ask permission"],"提供建议之前先了解对方希望得到何种回应。","any"),
        ("acknowledge","承认感受",["承认感受","认可感受","acknowledge feelings"],"在亲子沟通中承认孩子表达的感受。","any"),
        ("minimize","淡化感受",["小题大做","没什么大不了","minimize feelings"],"将对方的感受说成不重要。","any"),
        ("compare","比较孩子",["别人家的孩子","拿别人比较","compare children"],"用其他孩子的表现压过当前表达。","any"),
        ("collaborate","共同探索选择",["共同想办法","一起梳理","collaborative choices"],"先了解孩子的想法，再讨论可以怎样支持。","any"),
        ("conversation_timing","沟通时机",["不想现在聊","换个时间聊","conversation timing"],"考虑对方是否愿意以及当下是否适合交谈。","any"),
    ):node(identifier,label,"communication_principle",aliases,text,"parent_child",goal)
    node("parent_child_context","亲子沟通语境","source_context",["亲子沟通"],"来源为家长与孩子之间的沟通，未验证为AI对青少年的干预效果。","parent_child")
    for identifier in ("active_listening","ask_preference","conversation_timing"):
        edge("parent_child_context","recommends",identifier,"UNICEF_TALK","Transcript, paragraphs starting Next / Lastly / You could","parent_child")
    for identifier in ("active_listening","acknowledge","collaborate"):
        edge("parent_child_context","recommends",identifier,"UNICEF_11_13","Do: Listen / Acknowledge / Offer solutions","parent_child")
    for identifier in ("minimize","compare"):
        edge("parent_child_context","discourages",identifier,"UNICEF_11_13","Don't: Dismiss or minimize / Compare","parent_child")
    for key,label,zh,definition,aliases,facets in STRATEGIES:
        identifier="strategy_"+key;node(identifier,zh,"support_strategy",aliases+[label],definition,"English_crowd_support")
        for i,text in enumerate(facets):
            fid=f"facet_{key}_{i}";node(fid,text,"recognition_facet",[text],text,"English_crowd_support")
            edge(identifier,"has_facet",fid,"ACL2021","Appendix B, printed p3480, PDF zero-based index11","English_crowd_support","engineering_decomposition_of_definition")
    for key,label,text in (("exploration","探索阶段","Explore the situation without assuming hidden causes."),("comforting","安慰阶段","Offer empathy and understanding."),("action","行动阶段","Discuss optional ways forward.")):
        node("stage_"+key,label,"support_stage",[label,key],text,"English_crowd_support")
    for key in ("exploration","comforting","action"):edge("stage_"+key,"related","guide","ACL2021","Section3.2: flexible stages, not mandatory sequence","English_crowd_support","engineering_cross_schema_link_not_original_WHO_claim")
    # 修正跨文献连接的关系：来源阶段本身可追溯，连接到项目总方案仅为工程组织。
    for n in g["nodes"]:n["content_sha256"]=fingerprint(n)
    for e in g["edges"]:e["content_sha256"]=fingerprint(e)
    g["limitations"]=["draft_not_expert_reviewed","parent_child_guidance_not_AI_efficacy","strategy_labels_not_optimal_response","no_user_facts_stored","no_automatic_resolution_of_conflicting_claims"]
    assert not validate(g);return g


def apply_reviews(graph: dict[str,Any],rows: list[dict[str,str]]) -> dict[str,Any]:
    """导入实际人工审核记录；不验证真实身份或资格，模型记录和过期内容一律拒绝。"""
    result=copy.deepcopy(graph);items={n["id"]:n for n in result["nodes"]+result["edges"]};seen=set()
    for row in rows:
        identifier=row["item_id"]
        if identifier in seen or identifier not in items:raise ValueError("重复或未知审核对象")
        seen.add(identifier);item=items[identifier]
        if row.get("review_method")!="human" or not all(row.get(k,"").strip() for k in ("reviewer_id","reviewed_at","qualification_basis")):raise ValueError("需要真实人工审核记录，不接受模型代审")
        if row.get("content_sha256")!=fingerprint(item):raise ValueError("审核未绑定当前内容")
        if row.get("decision") not in {"approved","rejected"}:raise ValueError("审核决定无效")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T.+",row["reviewed_at"]):raise ValueError("审核时间须为ISO格式")
        item.update(review_status=row["decision"],reviewer_id=row["reviewer_id"],reviewed_at=row["reviewed_at"],review_method="human",review_attestation={"content_sha256":row["content_sha256"],"qualification_basis":row["qualification_basis"],"identity_independently_verified":False})
    if validate(result):raise ValueError("审核后图谱不合法")
    return result


class GraphIndex:
    """中文和英文别名/字串检索，保留词序边界、否定、引述、目标和来源场景。"""

    def __init__(self,graph: dict[str,Any]) -> None:
        if validate(graph):raise ValueError("图谱结构或审核指纹不合法")
        self.graph=graph;self.by_id={n["id"]:n for n in graph["nodes"]};self.nodes=[n for n in graph["nodes"] if n["kind"] in {"public_guideline_concept","communication_principle","support_strategy"}]
        self.vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(2,4),lowercase=True)
        self.matrix=self.vectorizer.fit_transform([" ".join([n["label"],n["definition"],*n["aliases"]]) for n in self.nodes])
        self.conflicted={e for group in conflicts(graph) for e in group}

    def retrieve(self,query: str,goal: str,*,research: bool=False,include_unreviewed: bool=False,setting: str="general_concept",limit: int=4) -> dict[str,Any]:
        """返回出处概念及边，不把输入说法写入图谱、不生成具体个人支持建议。"""
        if goal=="pause" or goal not in {"listen","understand","act"}:return {"status":"abstain","facts":[],"reason":"explicit_goal"}
        if not isinstance(query,str) or not query.strip() or not 1<=limit<=10:return {"status":"abstain","facts":[],"reason":"empty_or_invalid"}
        if re.search(r"诊断|确诊|用药|药量|治愈|diagnos|dosage|medication",query,re.I):return {"status":"abstain","facts":[],"reason":"outside_concept_scope"}
        clean=re.sub(r'“[^”]*”|「[^」]*」|"[^"\n]*"',lambda m:" "*len(m.group()),query)
        parts=list(re.finditer(r"[^。！？!?；;，,\n]+",clean));scores=np.asarray(self.matrix@self.vectorizer.transform([clean]).T.toarray()).ravel();rank=[]
        for i,n in enumerate(self.nodes):
            if n["review_status"]=="rejected" or (n["review_status"]!="approved" and not (research and include_unreviewed)):continue
            if n["source_scope"] not in {"general_concept",setting}:continue
            if n["requires_goal"]=="act" and goal!="act" and goal!="understand":continue
            blocked=False;direct=False
            matches=[]
            for part in parts:
                for alias in n["aliases"]:
                    pattern=r"\b"+re.escape(alias)+r"\b" if re.fullmatch(r"[A-Za-z -]+",alias) else re.escape(alias)
                    for m in re.finditer(pattern,part.group(),re.I):matches.append((part.start()+m.start(),part.group()[:m.start()]))
            if matches:
                _,prefix=max(matches,key=lambda p:p[0]);blocked=bool(re.search(r"不想|不要|不用|不需要|拒绝|朋友说|他说|她说|假设|假如|don't want|do not want|friend said",prefix,re.I));direct=not blocked
            if blocked:continue
            score=float(scores[i])+.8*direct
            if score<.12:continue
            incident=[]
            for e in self.graph["edges"]:
                if n["id"] not in {e["subject"],e["object"]} or e["id"] in self.conflicted or e["review_status"]=="rejected":continue
                if e["scope"]["setting"] not in {"general_stress_concept",setting}:continue
                if e["review_status"]!="approved" and not (research and include_unreviewed):continue
                endpoints=[self.by_id[e[k]] for k in ("subject","object")]
                if any(x["review_status"]=="rejected" or (x["review_status"]!="approved" and not (research and include_unreviewed)) for x in endpoints):continue
                incident.append({**e,"source_url":self.graph["sources"][e["source_id"]]["url"]})
            if incident:rank.append({"node_id":n["id"],"label":n["label"],"score":score,"definition":n["definition"],"facts":incident,"applied_to_user":False,"human_reviewed":n["review_status"]=="approved","source_setting":n["source_scope"]})
        rank.sort(key=lambda x:(-x["score"],x["node_id"]))
        return {"status":"research_concept_candidates" if rank else "abstain","facts":rank[:limit],"conflicts_withheld":len(self.conflicted),"not_personalized_advice":True}


def run() -> None:
    """版本化导出及空人工审核表，既有十八轮工件保持不动。"""
    folder=OUT/"knowledge_graph";folder.mkdir(parents=True,exist_ok=True)
    freeze(folder/"build_protocol.json",{"source_sha256":sha256(Path(__file__)),"parent_sha256":sha256(ROOT/"round18/knowledge_graph/graph.json"),"schema":"aic.support_concepts.v2","expert_review_count":0})
    g=build();save(folder/"graph.json",g);save(folder/"graph.jsonld",jsonld(g))
    for kind,rows in (("nodes",g["nodes"]),("edges",g["edges"])):
        fields=sorted(set().union(*(r.keys() for r in rows)))
        with (folder/(kind+".csv")).open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    fields=["item_id","item_type","content_sha256","decision","review_method","reviewer_id","reviewed_at","qualification_basis","comments"]
    with (folder/"expert_review_pending.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({"item_id":r["id"],"item_type":kind,"content_sha256":r["content_sha256"]} for kind,rows in (("node",g["nodes"]),("edge",g["edges"])) for r in rows)
    save(folder/"status.json",{"nodes":len(g["nodes"]),"edges":len(g["edges"]),"external_content_sources":["WHO2020","UNICEF_TALK","UNICEF_11_13","ACL2021"],"human_expert_reviews":0,"conflict_groups":conflicts(g),"production_connected":False,"default_unreviewed_allowed":False,"source_scope_preserved":True})
    print(json.dumps({"nodes":len(g["nodes"]),"edges":len(g["edges"]),"reviews":0}),flush=True)


if __name__=="__main__":run()
