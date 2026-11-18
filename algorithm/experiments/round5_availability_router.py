"""开发集选择可用性路由；没有充分视觉/人脸观测时回退独立文字适配模型。"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Any
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.round2_moe_train import load_split,grouped_delta_ci
from experiments.round5_face_search import features
from experiments.round5_frozen_search import summary
from experiments.round5_ensemble import normalized
from perception.knowledge_moe.features import save_json,sha256_file


def route_scores(data: dict[str,np.ndarray],face_scores: np.ndarray,fallback_scores: np.ndarray,minimum_coverage: float) -> tuple[np.ndarray,np.ndarray]:
    """路由只读可观测掩码和人脸覆盖，不接触标签、来源名称或未来结果。"""
    text_available=data["modality_mask"][:,0] & (data["quality"][:,0]>0)
    use_face=(text_available & data["modality_mask"][:,2] & (data["quality"][:,2]>0)
              & (data["face_summary"][:,24]>=minimum_coverage))
    scores=np.where(use_face[:,None],face_scores,fallback_scores)
    # 当前两分支都使用真实文字表征；无文字时不能用空串编码冒充有效观测。
    scores[~text_available]=np.nan
    return scores,use_face


def run(root: Path,previous: Path) -> dict[str,Any]:
    """严格按703开发样本选择门槛，历史821只作固定诊断。"""
    output=root/"availability_router";output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():raise ValueError("拒绝覆盖已完成结果")
    valid=load_split(root/"face_features/calibration.npz")
    with (root/"face_search/selected.pkl").open("rb") as handle:face=pickle.load(handle)
    ft=json.loads((root/"finetune/results.json").read_text(encoding="utf-8"));name=ft["winner"]
    x,_=features(valid,face["config"]["names"],face["scalers"],face["config"]["face_weight"])
    face_values=normalized(face["model"].decision_function(x));face_values[:,1]+=face["bias"]
    with np.load(root/"finetune"/(name+"_validation.npz"),allow_pickle=False) as saved:fallback=saved["logits"].copy()
    fallback[:,1]+=ft["development"][name]["bias"]
    protocol={"seed":42,"minimum_face_coverage_grid":[.2,.4,.6,.8],"criterion":"maximum703 development macro-F1, accuracy breaks ties",
        "routing_inputs":["visual availability","visual quality>0","fraction of sampled frames with detected face"],
        "no_label_or_source_routing":True,"fallback_checkpoint":name,"code_sha256":sha256_file(Path(__file__)),
        "motivation":"external2021 diagnostic revealed absent-media domain failure; do not call that dataset independent confirmation for this new router",
        "confirmation":"separate previously unused ECISA2019 official test will be evaluated after this lock"}
    save_json(output/"preregister.json",protocol);choices={}
    for threshold in protocol["minimum_face_coverage_grid"]:
        scores,mask=route_scores(valid,face_values,fallback,threshold)
        choices[str(threshold)]={"development":summary(valid["labels"],scores),"face_count":int(mask.sum()),"fallback_count":int((~mask).sum())}
    selected=max(choices,key=lambda key:(choices[key]["development"]["macro_f1"],choices[key]["development"]["accuracy"]))
    save_json(output/"selection_locked.json",{"minimum_face_coverage":float(selected),"choices":choices})
    test=load_split(root/"face_features/test.npz")
    with np.load(root/"face_search/historical_predictions.npz",allow_pickle=False) as saved:face_test=saved["scores"].copy()
    with np.load(root/"ensemble"/(name+"_historical.npz"),allow_pickle=False) as saved:fallback_test=saved["logits"].copy()
    fallback_test[:,1]+=ft["development"][name]["bias"]
    scores,mask=route_scores(test,face_test,fallback_test,float(selected));historical=summary(test["labels"],scores)
    with np.load(previous/"concat/test_predictions.npz",allow_pickle=False) as old:reference=old["logits"].argmax(1)
    ci=grouped_delta_ci(test["labels"],scores.argmax(1),reference,test["source_groups"],3)
    np.savez_compressed(output/"historical_predictions.npz",scores=scores,labels=test["labels"],sample_ids=test["sample_ids"],used_face=mask)
    report={"protocol":protocol,"minimum_face_coverage":float(selected),"development":choices,"historical":historical,
        "routing_counts":{"face":int(mask.sum()),"fallback":int((~mask).sum())},"relative_f1_improvement":historical["macro_f1"]/.5657375344573995-1,
        "macro_f1_delta_ci95":ci,"deployed":False}
    save_json(output/"results.json",report);print(json.dumps(report,ensure_ascii=False),flush=True);return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--previous",type=Path,required=True)
    args=parser.parse_args();run(args.root,args.previous)
