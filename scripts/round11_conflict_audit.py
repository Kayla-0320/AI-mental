"""官方单模态标注仅用于开发分层审计，不作为融合推理输入。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import ROOT,OUTPUT,PREVIOUS,digest,save


def metrics(y: np.ndarray,prediction: np.ndarray) -> dict[str,Any]:
    """固定三类，即使某分层某类没有样本也不改变平均类数。"""
    matrix=np.bincount(3*y+prediction,minlength=9).reshape(3,3)
    support=matrix.sum(1)
    denominator=support+matrix.sum(0)
    f1=np.divide(2*np.diag(matrix),denominator,out=np.zeros(3),where=denominator>0)
    recall=np.divide(np.diag(matrix),support,out=np.zeros(3),where=support>0)
    return {"n":len(y),"macro_f1":float(f1.mean()),"per_class_f1":f1.tolist(),
        "per_class_recall":recall.tolist(),
        "accuracy":float(np.trace(matrix)/max(1,len(y))),"support":support.tolist(),"confusion_matrix":matrix.tolist()}


def main() -> None:
    """统一depth4无可靠性头的因子对照，成功与回退案例均保留。"""
    folder=OUTPUT/"evidence_deterministic"
    with np.load(folder/"family00/oof.npz",allow_pickle=False) as source:
        ids=source["sample_ids"].astype(str)
        y=source["labels"]
        groups=source["source_groups"]
        old_all=source["probabilities"]
    with np.load(folder/"family10/oof.npz",allow_pickle=False) as source:
        if not np.array_equal(ids,source["sample_ids"]):
            raise ValueError("对照ID未对齐")
        new_all=source["probabilities"]
    selections=[]
    for family in ("family00","family10"):
        selections.append(json.loads((folder/family/"complete.json").read_text("utf-8"))["results"])
    official={row["video_id"]+"_"+row["clip_id"]:row for row in csv.DictReader((PREVIOUS/"official_meta.csv").open(encoding="utf-8"))}
    labels=np.array([[int(np.sign(float(official[key.replace("csbdgs2_","csbgs2_")][field])))+1
                      for field in ("label_T","label_A","label_V")] for key in ids])
    with np.load(OUTPUT/"input_audit/development_masks.npz",allow_pickle=False) as source:
        if not np.array_equal(ids,source["sample_ids"]):
            raise ValueError("质量审计ID未对齐")
        visible=source["modality_mask"]
    raw={}
    for split in ("train","selection"):
        with np.load(ROOT/f"_国赛优化/round5/text_features/{split}.npz",allow_pickle=False) as source:
            raw.update(dict(zip(source["sample_ids"].astype(str),source["raw_text"].astype(str))))
    strata={"all":np.ones(len(y),bool),"M_neutral":y==1,
        "T_A_opposite":(abs(labels[:,0]-labels[:,1])==2)&visible[:,1],
        "T_V_opposite":(abs(labels[:,0]-labels[:,2])==2)&visible[:,2],
        "T_A_or_V_opposite":((abs(labels[:,0]-labels[:,1])==2)&visible[:,1])|((abs(labels[:,0]-labels[:,2])==2)&visible[:,2])}
    best_indices=[max(range(len(rows)),key=lambda i:rows[i]["selection_score"]) for rows in selections]
    reports={}
    probabilities=[]
    for comparison,indices in (("matched_depth4_no_reliability",[0,0]),("same_search_budget_selected",best_indices)):
        probabilities=[]
        for p,rows,i in zip((old_all,new_all),selections,indices):
            logits=np.log(np.maximum(p[i],1e-12))+np.array(rows[i]["bias"])
            values=np.exp(logits-logits.max(1,keepdims=True))
            probabilities.append(values/values.sum(1,keepdims=True))
        predictions=[p.argmax(1) for p in probabilities]
        report=[]
        for name,active in strata.items():
            report.append({"stratum":name,"n":int(active.sum()),"groups":len(np.unique(groups[active])),
                "old":metrics(y[active],predictions[0][active]),"new":metrics(y[active],predictions[1][active]),
                "corrected":int(((predictions[0]!=y)&(predictions[1]==y)&active).sum()),
                "regressed":int(((predictions[0]==y)&(predictions[1]!=y)&active).sum())})
        reports[comparison]={"meta_indices":indices,"strata":report}
    cases={}
    for name,active in (("corrected",(predictions[0]!=y)&(predictions[1]==y)),
                        ("regressed",(predictions[0]==y)&(predictions[1]!=y))):
        chosen=np.flatnonzero(active&strata["T_A_or_V_opposite"])
        chosen=sorted(chosen,key=lambda i:ids[i])[:6]
        cases[name]=[{"id":ids[i],"text":raw[ids[i]],"M3":int(y[i]),"T_A_V":labels[i].tolist(),
            "old_prediction":int(predictions[0][i]),"new_prediction":int(predictions[1][i]),
            "old_probability":probabilities[0][i].tolist(),"new_probability":probabilities[1][i].tolist(),
            "scope":"posthoc qualitative development case;not independent test evidence"} for i in chosen]
    save(OUTPUT/"development_conflict_audit_v2.json",{"comparisons":reports,
        "matched_head":"depth4,no reliability,both use identical12point bias search",
        "old_recipe":"legacy quality,MSP3class","new_recipe":"pts quality,MSP+Sense3class",
        "definition":"official T/A/V signed polarity opposite;annotations used only for auditing",
        "development_only":True,"strata":report,"cases":cases,
        "cases_comparison":"same_search_budget_selected",
        "single_class_stratum_notice":"M_neutral subset uses recall;its fixed3class macroF1 is not overall neutralF1",
        "code_sha256":digest(Path(__file__))})
    print(json.dumps([{k:v for k,v in row.items() if k in ("stratum","n","corrected","regressed")}|
        {"old_f1":row["old"]["macro_f1"],"new_f1":row["new"]["macro_f1"]} for row in report]),flush=True)


if __name__=="__main__":
    main()
