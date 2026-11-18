"""知识证据包和策略识别的离线接口；不接入对话、用户画像或风险升级。"""
from __future__ import annotations
import json
from pathlib import Path
import pickle
from typing import Any
import numpy as np
from experiments.round14_common import sha256
from experiments.round19_prepare import OUT
from experiments.round19_graph import GraphIndex
from experiments.round19_evaluate import ARMS,matrix


def evidence_packet(query: str,goal: str,*,research: bool=False,include_unreviewed: bool=False,setting: str="general_concept",budget: int=1600) -> dict[str,Any]:
    """仅返回完整出处概念块；预算不足时整块舍弃，不截断引文，也不生成个人建议。"""
    if type(budget) is not int or not 100<=budget<=8000:raise ValueError("证据预算无效")
    path=OUT/"knowledge_graph/graph.json";graph=json.loads(path.read_text(encoding="utf-8"))
    retrieved=GraphIndex(graph).retrieve(query,goal,research=research,include_unreviewed=include_unreviewed,setting=setting)
    blocks=[];size=0
    for entry in retrieved["facts"]:
        facts=entry["facts"];block={"concept":entry["label"],"definition":entry["definition"],"source_setting":entry["source_setting"],"review_status":"pending" if not entry["human_reviewed"] else "human_review_recorded",
            "citations":[{"edge_id":e["id"],"url":e["source_url"],"locator":e["locator"],"source_sha256":e["source_sha256"],"scope":e["scope"],"derivation":e["derivation"]} for e in facts],"not_a_user_fact":True}
        length=len(json.dumps(block,ensure_ascii=False))
        if size+length>budget:continue
        blocks.append(block);size+=length
    return {"status":"research_evidence" if blocks else "abstain","evidence":blocks,"encoded_evidence_chars":size,"graph_sha256":sha256(path),"not_personalized_advice":True,"clinical_inference":False,"platform_connected":False}


def predict_strategy(texts: list[str],embedding: np.ndarray,channels: np.ndarray,folder: Path,arm: str) -> np.ndarray:
    """预测与公开策略标注的一致性；分数不是帮助程度、真人评级或最优行动概率。"""
    n=len(texts)
    if arm not in ARMS or embedding.shape!=(n,384) or channels.shape!=(n,8):raise ValueError("策略输入形状或对照臂无效")
    if any(not isinstance(t,str) for t in texts) or not np.isfinite(embedding).all() or not np.isfinite(channels).all():raise ValueError("策略输入不合法")
    if arm=="no_graph" and np.any(channels!=0):raise ValueError("无知识对照必须为零通道")
    lock=json.loads((folder/"selection_lock.json").read_text(encoding="utf-8"))
    if sha256(folder/"vectorizer.pkl")!=lock["vectorizer_sha256"] or sha256(folder/(arm+".pkl"))!=lock["arms"][arm]["model_sha256"]:raise ValueError("冻结策略模型摘要不符")
    with (folder/"vectorizer.pkl").open("rb") as stream:vectorizer=pickle.load(stream)
    with (folder/(arm+".pkl")).open("rb") as stream:model=pickle.load(stream)
    x=matrix(vectorizer.transform(texts),{"embedding":embedding,arm:channels},arm)
    if x.shape[1]!=lock["arms"][arm]["features"]:raise ValueError("特征模式不匹配")
    return model.decision_function(x)
