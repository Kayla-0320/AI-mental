"""既定可观测质量阈值的分层诊断；原总体和全部样本预测保持不变。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Any
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
from experiments.round11_common import OUTPUT,digest,save
from experiments.round7_decision_oof import metric


def strata(y: np.ndarray,old: np.ndarray,new: np.ndarray,q: np.ndarray,coverage: np.ndarray) -> list[dict[str,Any]]:
    """按真实观测分层，不能用预测是否错误来定义困难样本。"""
    masks={"all":np.ones(len(y),bool),"audio_quality_below_0.5":q[:,1]<.5,
        "visual_quality_below_0.5_or_face_coverage_below_0.6":(q[:,2]<.5)|(coverage<.6),
        "high_observed_quality":(q[:,1]>=.5)&(q[:,2]>=.5)&(coverage>=.6)}
    return [{"stratum":name,"n":int(mask.sum()),"old":metric(y[mask],old[mask].argmax(1)),
        "new":metric(y[mask],new[mask].argmax(1))} for name,mask in masks.items() if mask.any()]


def main() -> None:
    """先落盘诊断阈值，再在完成的固定预测上计算，不改变选型。"""
    parser=argparse.ArgumentParser();parser.add_argument("--plan",action="store_true");args=parser.parse_args()
    plan={"quality_threshold":.5,"face_coverage_threshold":.6,"no_sample_deletion":True,
        "selection_used":False,"scope":"postselection fixed observable strata;not independent new model selection",
        "code_sha256":digest(Path(__file__))}
    path=OUTPUT/"quality_strata_plan.json"
    if path.exists():
        if json.loads(path.read_text("utf-8"))!=plan:raise ValueError("质量分层阈值或代码改变")
    else:save(path,plan)
    if args.plan:return
    output=OUTPUT/"quality_strata.json"
    if output.exists():return
    from experiments.round11_features import load_data
    data=load_data("pts")
    with np.load(OUTPUT/"robustness/clean.npz",allow_pickle=False) as z:
        if not np.array_equal(z["sample_ids"],data["sample_ids"]):raise ValueError("开发质量分层ID不符")
        development=strata(z["labels"],z["old"],z["new"],data["quality"],data["coverage"])
    with np.load(OUTPUT/"emotiontalk_features/signal_features/test_inputs.npz",allow_pickle=False) as z:q=z["quality"];ids=z["sample_ids"]
    with np.load(OUTPUT/"emotiontalk_features/face_features/test_inputs.npz",allow_pickle=False) as z:
        if not np.array_equal(ids,z["sample_ids"]):raise ValueError("外部质量输入ID不符")
        coverage=z["coverage"]
    with np.load(OUTPUT/"external_m3/test_inputs_scored_predictions.npz",allow_pickle=False) as z:
        if not np.array_equal(ids,z["sample_ids"]):raise ValueError("外部质量分层预测ID不符")
        external=strata(z["labels"],z["old"].mean(0),z["new"].mean(0),q,coverage)
    save(output,{"development":development,"external_test":external,"scope":plan["scope"],"plan_sha256":digest(path)})


if __name__=="__main__":main()
