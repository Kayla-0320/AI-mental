"""按逐样本预测复算优化结果，验证留出身份、路由和外部数据隔离。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"algorithm"))
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
from experiments.round5_availability_router import route_scores
from experiments.round2_moe_train import load_split
from perception.knowledge_moe.features import save_json,sha256_file


def compare(saved: Path,expected: dict,key: str) -> int:
    """重新计算accuracy和macro-F1，验证样本ID和输出有限。"""
    with np.load(saved,allow_pickle=False) as data:
        y=data["labels"];scores=data[key];ids=data["sample_ids"]
        assert len(set(ids))==len(y) and scores.shape==(len(y),3) and np.isfinite(scores).all()
        prediction=scores.argmax(1)
        assert abs(accuracy_score(y,prediction)-expected["accuracy"])<1e-12
        assert abs(f1_score(y,prediction,labels=[0,1,2],average="macro",zero_division=0)-expected["macro_f1"])<1e-12
    return 3


def verify(root: Path,bundle: Path,previous: Path) -> dict:
    """不重训、不改选，只复算已锁定工件。"""
    checks=0;details=[]
    histories={}
    for component,key in [("frozen_search","scores"),("finetune","logits"),("ensemble","scores"),("face_search","scores"),("llm_search","scores"),("availability_router","scores")]:
        report=json.loads((root/component/"results.json").read_text(encoding="utf-8"));histories[component]=report["historical"]
        checks+=compare(root/component/"historical_predictions.npz",report["historical"],key)
        with np.load(root/component/"historical_predictions.npz",allow_pickle=False) as data,np.load(bundle/"test.npz",allow_pickle=False) as truth:
            assert np.array_equal(data["sample_ids"],truth["sample_ids"]) and np.array_equal(data["labels"],truth["labels"]);checks+=1
        details.append("historical/"+component)
    splits={name:load_split(bundle/(name+".npz")) for name in ["train","selection","calibration","test"]}
    for i,a in enumerate(splits):
        for b in list(splits)[i+1:]:
            assert not set(splits[a]["source_groups"]) & set(splits[b]["source_groups"]);checks+=1
    router=json.loads((root/"availability_router/results.json").read_text(encoding="utf-8"))
    face_data=load_split(root/"face_features/test.npz")
    ft=json.loads((root/"finetune/results.json").read_text(encoding="utf-8"));name=ft["winner"]
    with np.load(root/"face_search/historical_predictions.npz",allow_pickle=False) as data:primary=data["scores"].copy()
    with np.load(root/"ensemble"/(name+"_historical.npz"),allow_pickle=False) as data:fallback=data["logits"].copy()
    fallback[:,1]+=ft["development"][name]["bias"]
    replay,mask=route_scores(face_data,primary,fallback,router["minimum_face_coverage"])
    with np.load(root/"availability_router/historical_predictions.npz",allow_pickle=False) as data:
        assert np.array_equal(mask,data["used_face"]) and np.max(np.abs(replay-data["scores"]))<1e-12;checks+=1
        p,r,f,n=precision_recall_fscore_support(data["labels"],replay.argmax(1),labels=[0,1,2],zero_division=0)
        final_classes={"precision":p.tolist(),"recall":r.tolist(),"f1":f.tolist(),"support":n.tolist()}
    with np.load(previous/"concat/test_predictions.npz",allow_pickle=False) as old:
        baseline_f1=f1_score(old["labels"],old["logits"].argmax(1),average="macro")
        baseline_accuracy=accuracy_score(old["labels"],old["logits"].argmax(1))
        old_classes=precision_recall_fscore_support(old["labels"],old["logits"].argmax(1),labels=[0,1,2],zero_division=0)[2].tolist()
    achieved=router["historical"]["macro_f1"]>=baseline_f1*1.1
    assert achieved;checks+=1
    external=json.loads((root/"router_confirmation/results.json").read_text(encoding="utf-8"))
    with np.load(root/"router_confirmation/predictions.npz",allow_pickle=False) as data:
        assert not data["used_face"].any();checks+=1
        for key,label in [("reference","original_concat_text_only"),("scores","availability_router_text_only")]:
            assert abs(f1_score(data["labels"],data[key].argmax(1),average="macro")-external["metrics"][label]["macro_f1"])<1e-12;checks+=1
    sets={}
    for name,path in [("aux_train",root/"external/train.jsonl"),("aux_dev",root/"external/external_valid.jsonl"),
                      ("external2021",root/"fresh_holdout/holdout.jsonl"),("confirmation2019",root/"router_confirmation/holdout.jsonl")]:
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len({row["id"] for row in rows})==len(rows);checks+=1
        import re
        sets[name]={re.sub(r"[^\w\u4e00-\u9fff]","",row["text"].casefold()) for row in rows}
    for i,a in enumerate(sets):
        for b in list(sets)[i+1:]:assert not sets[a] & sets[b];checks+=1
    simulation=json.loads((root/"simulation/simulation_results.json").read_text(encoding="utf-8"))
    service=json.loads((root/"simulation/agent_service_simulation.json").read_text(encoding="utf-8"))
    assert simulation["protocol"]["human_participants"]==0 and not simulation["support"]["failures"] and service["synthetic"] and service["human_participants"]==0;checks+=1
    result={"status":"passed","checks":checks,"details":details,"target_relative_macro_f1":.1,"target_achieved_on_fixed_history":bool(achieved),
        "baseline":{"macro_f1":baseline_f1,"accuracy":baseline_accuracy,"per_class_f1":old_classes},"final":router["historical"],"final_classes":final_classes,
        "history_replayed_exactly":True,"external_data_exact_overlap":False,"human_effectiveness_verified":False,"verification_code_sha256":sha256_file(Path(__file__))}
    (root/"verification").mkdir(exist_ok=True);save_json(root/"verification/recomputed.json",result)
    print(json.dumps({"status":"passed","checks":checks,"relative_f1":router["relative_f1_improvement"],"classes":final_classes}),flush=True);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["root","bundle","previous"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();verify(args.root,args.bundle,args.previous)
