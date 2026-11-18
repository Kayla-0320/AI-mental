"""实际稀疏分派和用户背书知识的可观察行为，不用随机样例证明性能。"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.round3_moe_train import attested_graph
from perception.knowledge_moe.model import QualityAwareMoE
from perception.knowledge_moe.sparse import SparseQualityAwareMoE
from perception.knowledge_moe.knowledge import KnowledgeIndex
from perception.knowledge_moe.features import FrozenCLIPEncoder,save_json,sha256_file
from perception.knowledge_moe.runtime import FusionRuntime
from shared import MultimodalFeatureBatch


def values() -> tuple:
    torch.manual_seed(42)
    return {n:torch.randn(8,d) for n,d in {"text":4,"audio":3,"visual":2}.items()},torch.ones(8,3,dtype=torch.bool),torch.ones(8,3)


def test_sparse_executes_only_selected_rows_and_matches_dense_masked_reference() -> None:
    x,m,q=values();sparse=SparseQualityAwareMoE({"text":4,"audio":3,"visual":2},3,top_k=2).eval()
    dense=QualityAwareMoE({"text":4,"audio":3,"visual":2},3,top_k=2).eval();dense.load_state_dict(sparse.state_dict())
    actual=[0]*len(sparse.experts)
    hooks=[]
    for j,expert in enumerate(sparse.experts):
        hooks.append(expert.register_forward_hook(lambda module,inputs,output,index=j:actual.__setitem__(index,len(inputs[0]))))
    a,g,_=sparse(x,m,q);b,h,_=dense(x,m,q)
    for hook in hooks:hook.remove()
    assert torch.allclose(a,b,atol=1e-6) and torch.equal(g,h)
    assert actual==sparse.last_dispatch_counts and sum(actual)==16
    assert sum(actual)<8*len(sparse.experts)


def test_sparse_training_updates_router_and_selected_experts_and_masks_missing_nan() -> None:
    x,m,q=values();model=SparseQualityAwareMoE({"text":4,"audio":3,"visual":2},3)
    m[:,2]=False;x["visual"][:]=float("nan")
    logits,gates,balance=model(x,m,q)
    (torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,2,0,1,2,0,1]))+.01*balance).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.router.parameters())
    assert sum(p.grad.abs().sum() for p in model.router.parameters())>0
    assert not gates[:,[i for i,n in enumerate(model.expert_names) if "visual" in n]].any()


def test_sparse_only_one_available_expert_and_all_missing_abstention() -> None:
    x,m,q=values();model=SparseQualityAwareMoE({"text":4,"audio":3,"visual":2},3).eval();m[:,1:]=False
    _,g,_=model(x,m,q);assert torch.all(g[:,0]==1) and sum(model.last_dispatch_counts)==8
    m[:]=False
    with pytest.raises(ValueError):model(x,m,q)


def test_encode_text_avoids_broken_upstream_text_features_and_normalizes() -> None:
    """上游4.57的get_text_features在本模型返回None；文字入口必须走自身投影。"""
    class Batch(dict):
        def to(self, device:str) -> "Batch":return self
    class Tokenizer:
        def __call__(self, texts:list[str], **kwargs:object) -> Batch:return Batch(input_ids=[[1]]*len(texts))
    class TextOutputs:
        pooler_output=None
        last_hidden_state=torch.tensor([[[3.,4.]],[[1.,0.]]])
    class Model:
        config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=512))
        text_model=staticmethod(lambda **kwargs:TextOutputs())
        text_projection=staticmethod(lambda pooled:pooled)
        def get_text_features(self, **kwargs:object) -> torch.Tensor:
            raise AssertionError("不得调用已损坏的上游get_text_features")
    encoder=FrozenCLIPEncoder.__new__(FrozenCLIPEncoder)
    encoder.processor=SimpleNamespace(tokenizer=Tokenizer());encoder.model=Model();encoder.device="cpu"
    output=encoder.encode_text(["a","b"])
    assert np.allclose(output,[[.6,.8],[1.,0.]],atol=1e-6)
    assert np.allclose(np.linalg.norm(output,axis=1),1.,atol=1e-6)


def test_user_attestation_is_explicit_default_off_and_hash_bound(tmp_path:Path) -> None:
    graph=attested_graph(tmp_path)
    assert KnowledgeIndex(graph).retrieve("我很开心")==[]
    index=KnowledgeIndex(graph,allow_user_attested=True)
    hits=index.retrieve("我很开心");assert hits and hits[0]["review_status"]=="user_attested"
    data=json.loads(graph.read_text(encoding="utf-8"));data["nodes"][0]["text"]="modified"
    save_json(graph,data)
    with pytest.raises(ValueError,match="未绑定"):KnowledgeIndex(graph,allow_user_attested=True)


def test_hybrid_knowledge_uses_frozen_node_semantics_and_keeps_absence_explicit(tmp_path:Path) -> None:
    path=attested_graph(tmp_path);index=KnowledgeIndex(path,allow_user_attested=True)
    vectors={e["id"]:np.arange(1,9,dtype=np.float32)+i for i,e in enumerate(index.entries)}
    index.attach_semantic_vectors(vectors)
    encoded,present=index.encode(["我很开心","完全无关的描述"])
    assert encoded.shape==(2,40) and present.tolist()==[True,False]
    assert np.linalg.norm(encoded[0,:32])==pytest.approx(2**-.5,abs=1e-6)
    assert np.linalg.norm(encoded[0,32:])==pytest.approx(2**-.5,abs=1e-6)
    assert not encoded[1].any()
    invalid=dict(vectors);invalid[next(iter(invalid))]=np.full(8,np.nan)
    with pytest.raises(ValueError):KnowledgeIndex(path,allow_user_attested=True).attach_semantic_vectors(invalid)


def test_attested_runtime_requires_aligned_raw_text_and_never_forges_expert_identity(tmp_path:Path) -> None:
    graph=attested_graph(tmp_path);graph.replace(tmp_path/"knowledge.json");graph=tmp_path/"knowledge.json"
    kd=32;torch.manual_seed(42);model=SparseQualityAwareMoE({"text":4},3,knowledge_dim=kd)
    torch.save(model.state_dict(),tmp_path/"weights.pt")
    save_json(tmp_path/"manifest.json",{"schema_version":"knowledge_moe.v1","task":"sentiment","class_names":["negative","neutral","positive"],
        "dimensions":{"text":4},"hidden":32,"architecture":"sparse_moe","top_k":2,"knowledge_dim":kd,
        "knowledge_review_mode":"user_attested_only","knowledge_sha256":sha256_file(graph),"checkpoint_sha256":sha256_file(tmp_path/"weights.pt"),
        "encoder_fingerprints":{"text":"fixture"},"standardizer":{"text":{"mean":[0.]*4,"scale":[1.]*4}},"temperature":1.,"calibration":{"status":"uncalibrated"}})
    runtime=FusionRuntime(tmp_path)
    b=MultimodalFeatureBatch(sample_ids=["a"],modalities={"text":[[1.]*4]},observed={"text":[True]},quality={"text":[1.]},task="sentiment",class_names=["negative","neutral","positive"],encoder_fingerprints={"text":"fixture"})
    with pytest.raises(ValueError,match="原文"):runtime.predict(b)
    b.raw_text=["我很开心"]
    result=runtime.predict(b);assert np.isfinite(result.probabilities).all()
    data=json.loads(graph.read_text(encoding="utf-8"));proof=data["review_attestation"]
    assert proof["expert_name"] is None and proof["expert_review_date"] is None and proof["source_statement"]=="专家审核已经过关"
