"""只评文字/视觉专家互补；复用冻结来源隔离专家，不读取新音频或训练音频分支。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import pickle
from typing import Any
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round10_fusion import folds,take
from experiments.round13_train import ch_data
from experiments.round13_models import matrix,probabilities

OUT=ROOT/"round16/moe"
OLD=ROOT/"round13"
PATTERNS=(1,4,5)


def gate_features(bank: np.ndarray,mask: np.ndarray) -> np.ndarray:
    """门控仅使用文字/视觉专家概率和可用性，不读标签和来源身份。"""
    p=np.maximum(bank,1e-8)
    return np.column_stack((np.log(p).reshape(len(p),-1),-(p*np.log(p)).sum(2),mask[:,[0,2]].astype(float)))


def mix(bank: np.ndarray,mask: np.ndarray,gate: Any,mode: str,prior: np.ndarray) -> np.ndarray:
    """资格由真实模态决定；全缺失用训练先验；该门控先计算专家概率，不声称稀疏。"""
    code=mask.astype(int)@np.array([1,2,4]);eligible=np.column_stack([(code&p)==p for p in PATTERNS])
    if mode=="uniform":weights=eligible.astype(float)
    else:
        weights=np.zeros((len(mask),3));weights[:,gate.classes_.astype(int)]=gate.predict_proba(gate_features(bank,mask))
        weights*=eligible
        if mode=="anchored":weights=.5*weights+.5*np.column_stack([np.zeros(len(mask)),np.zeros(len(mask)),eligible[:,2]])
    weights/=np.maximum(weights.sum(1,keepdims=True),1e-12)
    result=(weights[:,:,None]*bank).sum(1);result[~eligible.any(1)]=prior
    return result


def run() -> None:
    """四折已曝光开发诊断，普通融合与各门控在相同样本、专家和条件对照。"""
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/"protocol.json",{"seed":42,"experts":PATTERNS,"audio":"discarded and permanently unavailable; no audio head executed",
        "conditions":["TV","T_only","V_only"],"methods":["ordinary_TV","uniform","learned","anchored"],
        "gate":"logistic C1 trained on source-inner-OOF best expert, no in-sample meta labels",
        "status":"previously exposed CH development source folds; diagnostic not fresh external evidence","source_sha256":sha256(Path(__file__))})
    data=ch_data();data["A"]=np.zeros((len(data["labels"]),1),np.float32);data["mask"][:,1]=False
    outputs={c:{k:np.zeros((len(data["labels"]),3)) for k in ("ordinary_TV","uniform","learned","anchored")} for c in ("TV","T_only","V_only")}
    audits=[]
    for fold,(tr,va) in enumerate(folds(data)):
        teacher_audit=json.loads((OLD/f"neural/ch/fold{fold}/teacher_source_audit.json").read_text(encoding="utf-8"))
        held_groups=set(data["source_groups"][va].tolist())
        for inner in teacher_audit["folds"]:
            assert not held_groups&set(inner["train_groups"]) and not held_groups&set(inner["valid_groups"])
        cache=np.load(OLD/f"neural/ch/fold{fold}/inner_oof.npz",allow_pickle=False)
        bank=cache["missing_A"][:,np.array(PATTERNS)-1,:]
        if len(bank)!=len(tr):raise ValueError("旧OOF缓存顺序或来源划分不匹配")
        train=take(data,tr);mask=train["mask"].copy()
        # 不可见专家不可成为教师目标。
        eligibility=np.column_stack([((mask.astype(int)@np.array([1,2,4]))&p)==p for p in PATTERNS])
        losses=-np.log(np.maximum(bank[np.arange(len(tr))[:,None],np.arange(3)[None,:],train["labels"][:,None]],1e-9))
        losses[~eligibility]=np.inf;targets=losses.argmin(1);active=eligibility.any(1)
        gate=LogisticRegression(C=1.,max_iter=1000,random_state=42).fit(gate_features(bank,mask)[active],targets[active])
        with (OLD/f"training/ch/fold{fold}/projection.pkl").open("rb") as stream:projection=pickle.load(stream)
        with (OLD/f"training/ch/linear_1/fold{fold}/model.pkl").open("rb") as stream:bundle=pickle.load(stream)
        with (OUT/f"gate_fold{fold}.pkl").open("wb") as stream:pickle.dump(gate,stream)
        for condition in outputs:
            part=take(data,va)
            if condition=="T_only":part["mask"][:,2]=False;part["V"][:]=0
            if condition=="V_only":part["mask"][:,0]=False;part["T"][:]=0
            projected=projection.transform(part)
            expert=np.stack([probabilities(bundle["models"][p],matrix(projected,p,None)) for p in PATTERNS],axis=1)
            outputs[condition]["ordinary_TV"][va]=expert[:,2]
            for mode in ("uniform","learned","anchored"):outputs[condition][mode][va]=mix(expert,part["mask"],gate,mode,bundle["prior"])
        audits.append({"fold":fold,"train_groups":sorted(set(data["source_groups"][tr].tolist())),"held_groups":sorted(held_groups),"gate_classes":gate.classes_.tolist(),
            "projection_sha256":sha256(OLD/f"training/ch/fold{fold}/projection.pkl"),"heads_sha256":sha256(OLD/f"training/ch/linear_1/fold{fold}/model.pkl")})
        print(json.dumps({"fold":fold,"completed":True}),flush=True)
    y=data["labels"];result={}
    for c,methods in outputs.items():
        result[c]={}
        for name,p in methods.items():
            assert np.isfinite(p).all() and np.allclose(p.sum(1),1.,atol=1e-6)
            result[c][name]={"macro_f1":float(f1_score(y,p.argmax(1),labels=[0,1,2],average="macro",zero_division=0)),"accuracy":float(accuracy_score(y,p.argmax(1)))}
        np.savez_compressed(OUT/(c+"_predictions.npz"),truth=y,groups=data["source_groups"],sample_ids=data["sample_ids"],**methods)
    oracle={}
    for c,methods in outputs.items():
        oracle[c]="Not reported as deployable performance; choosing with test labels is prohibited"
    save(OUT/"fold_audits.json",audits);save(OUT/"results.json",{"metrics":result,"samples":len(y),"oracle":oracle,
        "cost":"gates use already-computed T,V,TV predictions; up to3 heads versus1, no encoder speedup claim","production_changed":False})
    print(json.dumps(result),flush=True)


if __name__=="__main__":run()
