"""面部表情表征与文字/原模态组合，固定开发选择后评测历史留出。"""
from __future__ import annotations
import argparse
import json
import os
import pickle
from pathlib import Path
import sys
from typing import Any
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from experiments.round2_moe_train import load_split,grouped_delta_ci
from experiments.round4_accuracy import merge
from experiments.round5_frozen_search import summary
from experiments.round5_ensemble import select_bias,normalized
from perception.knowledge_moe.features import save_json,sha256_file


def features(data: dict[str,np.ndarray],names: list[str],scalers: dict[str,Any]|None=None,face_weight: float=1.) -> tuple[np.ndarray,dict]:
    """各模态按训练统计标准化并按维数平衡，避免29维表情被千维语义淹没。"""
    fitting=scalers is None;scalers={} if scalers is None else scalers
    parts=[]
    for name in names:
        x=data[name].copy()
        if name in ["visual","audio"]:
            j=["text","audio","visual"].index(name);x*=data["modality_mask"][:,j:j+1]*data["quality"][:,j:j+1]
        if fitting:scalers[name]=StandardScaler().fit(x)
        value=scalers[name].transform(x)/np.sqrt(x.shape[1])
        if name=="face_summary":value*=face_weight
        parts.append(value)
    return np.column_stack(parts),scalers


def run(bundle: Path,previous: Path,output: Path) -> dict:
    """公开候选及开发成绩，禁止按历史结果改选。"""
    output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():raise ValueError("拒绝覆盖结果")
    train=merge([load_split(bundle/(s+".npz")) for s in ["train","selection"]]);valid=load_split(bundle/"calibration.npz")
    namesets=[["face_summary"],["roberta_mean","face_summary"],["roberta_mean","bge_cls","visual","audio","face_summary"],
              ["text","visual","audio","face_summary"]]
    configs=[{"id":f"{'+'.join(names)}_{kind}_{c}_{weight}_{fw}","names":names,"kind":kind,"C":c,"weight":weight,"face_weight":fw}
             for names in namesets for kind,cs in [("lr",[.1,1.,10.]),("svm",[1.,10.])] for c in cs for weight in [None,"balanced"] for fw in ([1.] if len(names)==1 else [.5,1.,2.])]
    protocol={"seed":42,"configs":configs,"selection":"703 development only","historical":"fixed821 never used for candidate selection",
              "face_identity":"largest detected face, speaker unverified","features_sha256":sha256_file(bundle/"manifest.json"),"code_sha256":sha256_file(Path(__file__))}
    save_json(output/"preregister.json",protocol);best=None;results={};selected=None
    for i,config in enumerate(configs):
        x,scalers=features(train,config["names"],face_weight=config["face_weight"]);v,_=features(valid,config["names"],scalers,config["face_weight"])
        model=(LogisticRegression(C=config["C"],class_weight=config["weight"],max_iter=700,random_state=42) if config["kind"]=="lr" else
               SVC(C=config["C"],class_weight=config["weight"],kernel="rbf",gamma="scale",random_state=42))
        model.fit(x,train["labels"]);scores=normalized(model.decision_function(v));bias,metric=select_bias(valid["labels"],scores)
        results[config["id"]]={"validation":metric,"bias":bias};key=(metric["macro_f1"],metric["accuracy"])
        if best is None or key>best:
            best=key;selected={"model":model,"scalers":scalers,"config":config,"bias":bias}
        if (i+1)%20==0:print(json.dumps({"face_candidates":i+1,"best":best}),flush=True)
    save_json(output/"selection_locked.json",{"winner":selected["config"]["id"],"development":results})
    test=load_split(bundle/"test.npz");x,_=features(test,selected["config"]["names"],selected["scalers"],selected["config"]["face_weight"])
    scores=normalized(selected["model"].decision_function(x));scores[:,1]+=selected["bias"]
    historical=summary(test["labels"],scores)
    with np.load(previous/"concat/test_predictions.npz",allow_pickle=False) as old:reference=old["logits"].argmax(1)
    ci=grouped_delta_ci(test["labels"],scores.argmax(1),reference,test["source_groups"],3)
    np.savez_compressed(output/"historical_predictions.npz",scores=scores,labels=test["labels"],sample_ids=test["sample_ids"])
    with (output/"selected.pkl").open("wb") as handle:pickle.dump(selected,handle)
    report={"protocol":protocol,"winner":selected["config"]["id"],"development":results,"historical":historical,
        "relative_f1_improvement":historical["macro_f1"]/.5657375344573995-1,"macro_f1_delta_ci95":ci,"deployed":False}
    save_json(output/"results.json",report);print(json.dumps({"winner":report["winner"],"historical":historical,"relative":report["relative_f1_improvement"],"ci":ci}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["bundle","previous","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.previous,args.output)
