"""同一训练/开发规模下重训原拼接，并移除最终方案的人脸特征作对照。"""
from __future__ import annotations
import argparse
import copy
import json
import os
import pickle
from pathlib import Path
import sys
from typing import Any
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from experiments.round2_moe_train import load_split,seed_all,tensors,forward,grouped_delta_ci
from experiments.round4_accuracy import merge
from experiments.round5_face_search import features
from experiments.round5_frozen_search import summary
from experiments.round5_ensemble import select_bias,normalized
from perception.knowledge_moe.features import FeatureStandardizer,save_json,sha256_file
from perception.knowledge_moe.knowledge import KnowledgeIndex
from perception.knowledge_moe.model import ConcatenationHead


def run(bundle: Path,face_result: Path,previous: Path,output: Path) -> dict[str,Any]:
    """对照只按开发结果选择；不会更改已锁定最终模型。"""
    output.mkdir(parents=True,exist_ok=True)
    train=merge([load_split(bundle/(s+".npz")) for s in ["train","selection"]]);valid=load_split(bundle/"calibration.npz")
    protocol={"seed":42,"training":2876,"development":703,"historical":821,"original_concat_hidden":[32,64],"epochs_max":80,"patience":12,
        "policy":"control candidates chosen on development only; does not reselect final deployed/research winner","code_sha256":sha256_file(Path(__file__))}
    save_json(output/"preregister.json",protocol)
    names=["text","audio","visual"]
    scaler=FeatureStandardizer().fit({name:train[name] for name in names},train["modality_mask"])
    index=KnowledgeIndex(Path(__file__).resolve().parents[1]/"perception/knowledge_moe/knowledge_seed.json")
    tr=tensors(train,names,scaler,index);va=tensors(valid,names,scaler,index)
    candidates=[]
    for hidden in [32,64]:
        seed_all();model=ConcatenationHead({name:train[name].shape[1] for name in names},3,hidden)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        counts=torch.bincount(tr[-1],minlength=3).float();criterion=nn.CrossEntropyLoss(weight=len(train["labels"])/(3*counts))
        best=None;state=None;stale=0;best_epoch=0
        for epoch in range(80):
            model.train();order=torch.randperm(len(train["labels"]))
            for start in range(0,len(order),64):
                ix=order[start:start+64];logits,_,_,mask=forward(model,tr,ix,augment=True)
                loss=criterion(logits,tr[-1][ix][mask]);optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            model.eval()
            with torch.inference_mode():scores,_,_,_=forward(model,va,torch.arange(len(valid["labels"])))
            metric=summary(valid["labels"],scores.numpy());key=(metric["macro_f1"],metric["accuracy"])
            if best is None or key>best:best=key;state=copy.deepcopy(model.state_dict());stale=0;best_epoch=epoch+1
            else:stale+=1
            if stale>=12:break
        model.load_state_dict(state);model.eval();candidates.append((best,hidden,best_epoch,model))
    chosen=max(candidates,key=lambda row:row[0])
    with (face_result/"selected.pkl").open("rb") as handle:selected=pickle.load(handle)
    reduced=[name for name in selected["config"]["names"] if name!="face_summary"]
    x,scalers=features(train,reduced);v,_=features(valid,reduced,scalers)
    linear=LogisticRegression(C=selected["config"]["C"],class_weight=selected["config"]["weight"],max_iter=700,random_state=42).fit(x,train["labels"])
    bias,metric=select_bias(valid["labels"],normalized(linear.decision_function(v)))
    save_json(output/"selection_locked.json",{"concat":{"hidden":chosen[1],"epoch":chosen[2],"development_f1":chosen[0][0]},"without_face":{"development":metric,"bias":bias}})
    test=load_split(bundle/"test.npz");te=tensors(test,names,scaler,index)
    with torch.inference_mode():concat,_,_,_=forward(chosen[3],te,torch.arange(len(test["labels"])))
    x,_=features(test,reduced,scalers);reduced_scores=normalized(linear.decision_function(x));reduced_scores[:,1]+=bias
    with np.load(face_result/"historical_predictions.npz",allow_pickle=False) as final:face_scores=final["scores"]
    results={"retrained_original_concat":summary(test["labels"],concat.numpy()),"new_features_without_face":summary(test["labels"],reduced_scores),"final_with_face":summary(test["labels"],face_scores)}
    comparisons={}
    for name,scores in [("retrained_original_concat",concat.numpy()),("new_features_without_face",reduced_scores)]:
        comparisons["final_minus_"+name]={"delta_macro_f1":results["final_with_face"]["macro_f1"]-results[name]["macro_f1"],
            "ci95":grouped_delta_ci(test["labels"],face_scores.argmax(1),scores.argmax(1),test["source_groups"],3)}
        np.savez_compressed(output/(name+".npz"),scores=scores,labels=test["labels"],sample_ids=test["sample_ids"])
    report={"protocol":protocol,"historical":results,"comparisons":comparisons,"limitations":["different pretrained representations and classifiers remain factors","historical repeated diagnostics, not new independent confirmation"]}
    save_json(output/"results.json",report);print(json.dumps(report,ensure_ascii=False),flush=True);return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["bundle","face-result","previous","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.face_result,args.previous,args.output)
