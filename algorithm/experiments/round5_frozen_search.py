"""新文字表征、核分类器与多模态融合；开发锁定后评测原821条历史留出。"""
from __future__ import annotations
import argparse
import json
import os
import pickle
from pathlib import Path
import sys
from typing import Any
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,recall_score
from experiments.round2_moe_train import load_split,grouped_delta_ci
from experiments.round4_accuracy import merge
from perception.knowledge_moe.features import save_json,sha256_file


def summary(y: np.ndarray,scores: np.ndarray) -> dict[str,Any]:
    """只报告分类指标，不把SVM margin冒充校准概率。"""
    pred=scores.argmax(1)
    return {"n":len(y),"accuracy":float(accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,labels=[0,1,2],average="macro",zero_division=0)),
            "per_class_recall":recall_score(y,pred,labels=[0,1,2],average=None,zero_division=0).tolist(),
            "confusion_matrix":confusion_matrix(y,pred,labels=[0,1,2]).tolist()}


def matrix(data: dict[str,np.ndarray],names: list[str]) -> np.ndarray:
    """拼接固定表征，屏蔽明确不可用的音视频；不追加任何标签字段。"""
    parts=[]
    for name in names:
        value=data[name].copy()
        if name in ["visual","audio"]:
            j=["text","audio","visual"].index(name)
            value*=data["modality_mask"][:,j:j+1]*data["quality"][:,j:j+1]
        parts.append(value)
    return np.column_stack(parts)


def run(bundle: Path,previous: Path,output: Path) -> dict[str,Any]:
    """固定搜索网格，保留所有候选的开发成绩，历史测试不参与选型。"""
    output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():raise ValueError("拒绝覆盖已完成结果")
    train=merge([load_split(bundle/(name+".npz")) for name in ["train","selection"]])
    valid=load_split(bundle/"calibration.npz")
    namesets=[[name,*extra] for name in ["roberta_cls","roberta_mean","bge_cls","bge_mean"] for extra in [[],["visual","audio"]]]
    namesets += [["roberta_mean","bge_cls","visual","audio"]]
    configs=[{"id":f"{'+'.join(names)}_{kind}_{c}_{weight}","names":names,"kind":kind,"C":c,"weight":weight}
             for names in namesets for kind,cs in [("lr",[.1,1.,10.]),("svm",[1.,10.])] for c in cs for weight in [None,"balanced"]]
    protocol={"seed":42,"configs":configs,"training":"round3 train+selection, 2876 samples",
        "selection":"round3 calibration now repurposed as development selection, 703 samples; no calibrated probability claim",
        "test":"same original821 already examined historical samples, never used to select candidates",
        "target":{"macro_f1":.5657375344573995*1.1,"relative_improvement":.1},"neutral_bias_grid":[-.5,0.,.5,1.],
        "input_sha256":sha256_file(bundle/"manifest.json"),"code_sha256":sha256_file(Path(__file__))}
    save_json(output/"preregister.json",protocol)
    best=None;results={};fitted={};valid_scores={}
    for config in configs:
        scaler=StandardScaler();x=scaler.fit_transform(matrix(train,config["names"]))
        v=scaler.transform(matrix(valid,config["names"]))
        model=(LogisticRegression(C=config["C"],class_weight=config["weight"],max_iter=700,random_state=42)
               if config["kind"]=="lr" else SVC(C=config["C"],kernel="rbf",gamma="scale",class_weight=config["weight"],decision_function_shape="ovr",random_state=42))
        model.fit(x,train["labels"]);scores=model.decision_function(v)
        scores=(scores-scores.mean(1,keepdims=True))/np.maximum(scores.std(1,keepdims=True),1e-6)
        raw=summary(valid["labels"],scores);choices=[]
        for offset in protocol["neutral_bias_grid"]:
            modified=scores.copy();modified[:,1]+=offset
            metric=summary(valid["labels"],modified)
            if metric["accuracy"]>=raw["accuracy"]-.02:choices.append((metric["macro_f1"],metric["accuracy"],-abs(offset),offset,metric))
        choice=max(choices,key=lambda item:item[:3]);selected=scores.copy();selected[:,1]+=choice[3]
        results[config["id"]]={"raw":raw,"selected":choice[4],"neutral_bias":choice[3]}
        valid_scores[config["id"]]=selected
        key=(choice[0],choice[1]);fitted[config["id"]]={"model":model,"scaler":scaler,"config":config,"offset":choice[3]}
        if best is None or key>best[:2]:best=(*key,config["id"])
        if len(results)%10==0:print(json.dumps({"done":len(results),"best":best}),flush=True)
    winner=best[2]
    save_json(output/"selection_locked.json",{"winner":winner,"development":results})
    test=load_split(bundle/"test.npz");f=fitted[winner]
    scores=f["model"].decision_function(f["scaler"].transform(matrix(test,f["config"]["names"])))
    scores=(scores-scores.mean(1,keepdims=True))/np.maximum(scores.std(1,keepdims=True),1e-6);scores[:,1]+=f["offset"]
    score=summary(test["labels"],scores)
    with np.load(previous/"concat/test_predictions.npz",allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"],test["sample_ids"]):raise ValueError("历史样本ID错位")
        old=saved["logits"].argmax(1)
    ci=grouped_delta_ci(test["labels"],scores.argmax(1),old,test["source_groups"],3)
    report={"protocol":protocol,"winner":winner,"development":results,"historical":score,
            "relative_f1_improvement":score["macro_f1"]/.5657375344573995-1,"delta_ci95":ci,"external_validation":False,"deployed":False}
    save_json(output/"results.json",report)
    np.savez_compressed(output/"historical_predictions.npz",scores=scores,labels=test["labels"],sample_ids=test["sample_ids"])
    np.savez_compressed(output/"development_predictions.npz",labels=valid["labels"],**valid_scores)
    with (output/"selected.pkl").open("wb") as handle:pickle.dump(f,handle)
    print(json.dumps({"winner":winner,"historical":score,"relative":report["relative_f1_improvement"],"ci":ci}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["bundle","previous","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.previous,args.output)
