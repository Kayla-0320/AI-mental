"""文字缺失时的明确退化路径，开发来源选权重，不修改干净输入决策。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import pickle
from typing import Any
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
from scipy.special import softmax
from experiments.round11_common import OUTPUT,digest,save
from experiments.round11_features import load_data
from experiments.round11_evidence import expert_evidence
from experiments.round10_fusion import take
from experiments.round7_decision_oof import metric


def route(raw: np.ndarray,evidence: np.ndarray,missing: np.ndarray,weight: float,bias: list[float],prior: np.ndarray) -> np.ndarray:
    """缺字时只合并确实存在的A/V证据；全缺失返回训练先验。"""
    result=raw.copy()
    a=evidence[:,6:9];v=evidence[:,9:12]
    activea=a.sum(1)>0;activev=v.sum(1)>0
    weightsa=np.where(activea,np.where(activev,weight,1.),0.)
    weightsv=np.where(activev,np.where(activea,1-weight,1.),0.)
    p=weightsa[:,None]*a+weightsv[:,None]*v
    # 端点权重为0但另一模态不存在时仍不能人为删除唯一证据。
    p[~(activea|activev)]=prior
    p=softmax(np.log(np.maximum(p,1e-12))+bias,axis=1)
    if raw.ndim==3:result[:,missing]=p[missing][None]
    else:result[missing]=p[missing]
    return result


def main() -> None:
    """五档全局权重只用来源OOF；T完整时与原模型逐位一致。"""
    output=OUTPUT/"missing_route"
    if (output/"selection_locked.json").exists():return
    save(output/"preregister.json",{"condition":"T is absent;clean/T-present predictions untouched","weights_A":[0.,.25,.5,.75,1.],
        "labels":"original development M3;no labels for generated counterfactuals",
        "policy":"existing held-out A/V experts;global same final class bias;all absent training prior",
        "code_sha256":digest(Path(__file__))})
    data=load_data("pts");ids=data["sample_ids"];e=np.full((len(ids),44),np.nan);p=np.full((len(ids),3),np.nan)
    bias=json.loads((OUTPUT/"blend/selection_locked.json").read_text("utf-8"))["winner"]["bias"]
    for f in range(4):
        folder=OUTPUT/"evidence_deterministic/family10"
        with np.load(folder/f"fold{f}.npz") as z:held=set(z["valid_ids"])
        ix=np.array([i for i,k in enumerate(ids) if k in held]);part=take(data,ix)
        part["modality_mask"]=part["modality_mask"].copy();part["quality"]=part["quality"].copy()
        part["modality_mask"][:,0]=False;part["quality"][:,0]=0
        with (folder/f"fold{f}_models.pkl").open("rb") as handle:bundle=pickle.load(handle)
        value=expert_evidence(bundle["experts"],part)
        if f==0:e=np.full((len(ids),value.shape[1]),np.nan)
        e[ix]=value;p[ix]=softmax(np.log(np.maximum(bundle["meta_models"][0].predict_proba(value),1e-12))+bias,axis=1)
    y=data["labels"];prior=np.bincount(y,minlength=3)/len(y);rows=[]
    for weight in (0.,.25,.5,.75,1.):
        value=route(p,e,np.ones(len(ids),bool),weight,bias,prior)
        rows.append({"audio_weight":weight,"metrics":metric(y,value.argmax(1))})
    winner=max(rows,key=lambda r:(r["metrics"]["macro_f1"],-abs(r["audio_weight"]-.5)))
    plain=metric(y,p.argmax(1));promote=winner["metrics"]["macro_f1"]>plain["macro_f1"]
    save(output/"selection_locked.json",{"winner":winner,"candidates":rows,"plain_meta_missing_T":plain,
        "promoted":promote,"clean_unchanged_by_construction":True,"development_only":True})
    print(json.dumps({"missing_T_plain":plain["macro_f1"],"fallback":winner["metrics"]["macro_f1"],"promoted":promote}),flush=True)


if __name__=="__main__":main()
