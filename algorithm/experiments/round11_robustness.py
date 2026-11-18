"""来源留出模型的全样本缺失/错配，以及固定媒体子集的真实波形压力。"""
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
from experiments.round11_features import load_data,alignment_feature
from experiments.round11_evidence import expert_evidence
from experiments.round10_fusion import take
from experiments.round7_decision_oof import metric


def modify(data: dict[str,np.ndarray],condition: str) -> dict[str,np.ndarray]:
    """保持原样本和目标，显式缺失不能伪装成中性。"""
    result={k:v.copy() for k,v in data.items()}
    for index,name in enumerate(("T","A","V")):
        if f"missing_{name}" in condition or condition=="text_only" and name in ("A","V"):
            result["modality_mask"][:,index]=False;result["quality"][:,index]=0.
            if name=="V":result["coverage"][:]=0.;result["face_summary"][:]=0.;result["deep_face"][:]=0.;result["visual"][:]=0.
    if condition=="audio_mismatch":
        # 相反/同类均可能；只按来源约束和ID选供体，不读取标注。
        fold_for={}
        for fold in range(4):
            with np.load(OUTPUT/f"evidence_deterministic/family10/fold{fold}.npz") as z:
                fold_for.update({str(k):fold for k in z["valid_ids"]})
        groups=data["source_groups"];order=np.argsort(data["sample_ids"].astype(str));donors=np.empty(len(order),int)
        for i in range(len(order)):
            donor=next(int(j) for j in np.roll(order,-i) if groups[j]!=groups[i]
                and fold_for[str(data["sample_ids"][j])]==fold_for[str(data["sample_ids"][i])]);donors[i]=donor
        for field in ("audio","speech","speech_emotion","speech_available","sensevoice","sensevoice_query","sensevoice_emotion","sensevoice_available","asr_alignment"):
            result[field]=data[field][donors].copy()
        result["modality_mask"][:,1]=data["modality_mask"][donors,1]
        result["quality"][:,1]=data["quality"][donors,1]
    return result


def source_predict(data: dict[str,np.ndarray],family: int,bias: list[float]) -> np.ndarray:
    """每条媒体只用没见过该来源的外层模型；绝不重拟合压力集。"""
    folder=OUTPUT/"evidence_deterministic"/f"family{family:02d}"
    output=np.full((len(data["sample_ids"]),3),np.nan)
    policy=json.loads((OUTPUT/"missing_route/selection_locked.json").read_text("utf-8")) if family==10 else {}
    for fold in range(4):
        with np.load(folder/f"fold{fold}.npz",allow_pickle=False) as z:held=set(z["valid_ids"].astype(str))
        ix=np.array([i for i,key in enumerate(data["sample_ids"]) if str(key) in held])
        if not len(ix):continue
        with (folder/f"fold{fold}_models.pkl").open("rb") as handle:model=pickle.load(handle)
        part=take(data,ix)
        if set(part["source_groups"])&set(model["training_groups"]):raise ValueError("压力验证来源进入专家训练")
        x=expert_evidence(model["experts"],part)
        p=model["meta_models"][0].predict_proba(x)
        probability=softmax(np.log(np.maximum(p,1e-12))+bias,axis=1)
        if policy.get("promoted",False):
            from experiments.round11_missing_route import route
            missing=~(part["modality_mask"][:,0]&(part["quality"][:,0]>0))
            if missing.any():
                # 完全缺失回退的先验只取该外层模型的训练ID，不能读取留出标签来造先验。
                with np.load(folder/"oof.npz",allow_pickle=False) as z:
                    selected=np.isin(z["sample_ids"],model["training_sample_ids"])
                    counts=np.bincount(z["labels"][selected],minlength=3)
                probability=route(probability,x,missing,policy["winner"]["audio_weight"],bias,counts/counts.sum())
        output[ix]=probability
    if not np.isfinite(output).all():raise ValueError("压力样本来源OOF未完整覆盖")
    return output


def main() -> None:
    """同预算旧表征对照与新锁定头；所有诊断仍属开发来源。"""
    folder=OUTPUT/"robustness";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"results.json").exists():return
    conditions=["clean","missing_A","missing_V","missing_T","text_only","audio_mismatch"]
    plan={"conditions":conditions,"models":["family00 depth4","family10 depth4"],
        "selection":"already locked;diagnostics do not pick new recipe","sample_rule":"all2876 for missing/mismatch/waveform,unchanged IDs and gold",
        "code_sha256":digest(Path(__file__))}
    pre=folder/"preregister.json"
    if pre.exists():
        if json.loads(pre.read_text("utf-8"))!=plan:raise ValueError("压力诊断计划发生改变")
    else:save(pre,plan)
    data=load_data("pts");legacy=load_data("legacy")
    oldrows=json.loads((OUTPUT/"evidence_deterministic/family00/complete.json").read_text("utf-8"))["results"]
    oldbias=oldrows[0]["bias"]
    newbias=json.loads((OUTPUT/"blend/selection_locked.json").read_text("utf-8"))["winner"]["bias"]
    rows=[]
    for condition in conditions:
        p=source_predict(modify(data,condition),10,newbias);old=source_predict(modify(legacy,condition),0,oldbias)
        row={"condition":condition,"n":len(p),"coverage":1.,"new":metric(data["labels"],p.argmax(1)),"old_same_budget":metric(data["labels"],old.argmax(1))}
        rows.append(row);np.savez_compressed(folder/f"{condition}.npz",sample_ids=data["sample_ids"],labels=data["labels"],new=p,old=old,source_groups=data["source_groups"])
        print(json.dumps(row),flush=True)
    if (OUTPUT/"wave_stress/complete.json").exists():
        with np.load(OUTPUT/"wave_stress/sense_clean.npz") as z:ids=z["sample_ids"]
        lookup={str(k):i for i,k in enumerate(data["sample_ids"])};ix=np.array([lookup[str(k)] for k in ids]);subset=take(data,ix);oldsubset=take(legacy,ix)
        for snr in ("clean",20,10):
            newdata={k:v.copy() for k,v in subset.items()};olddata={k:v.copy() for k,v in oldsubset.items()}
            for kind in ("sense","msp"):
                with np.load(OUTPUT/f"wave_stress/{kind}_{snr}.npz",allow_pickle=False) as z:
                    if not np.array_equal(z["sample_ids"],ids):raise ValueError("波形压力ID不符")
                    for target in (newdata,olddata):target["audio"]=z["audio"]
                    if kind=="sense":
                        newdata["sensevoice"]=z["features"];newdata["sensevoice_query"]=z["query"];newdata["sensevoice_emotion"]=z["emotion"]
                        newdata["asr_alignment"]=np.stack([alignment_feature(str(a),str(b)) for a,b in zip(newdata["raw_text"],z["transcript"])])
                    else:
                        for target in (newdata,olddata):target["speech"]=z["features"];target["speech_emotion"]=z["emotion"]
            p=source_predict(newdata,10,newbias);old=source_predict(olddata,0,oldbias)
            row={"condition":f"waveform_SNR_{snr}","n":len(p),"coverage_of_registered_subset":1.,
                "new":metric(subset["labels"],p.argmax(1)),"old_same_budget":metric(subset["labels"],old.argmax(1))}
            rows.append(row);np.savez_compressed(folder/f"wave_{snr}.npz",sample_ids=ids,labels=subset["labels"],new=p,old=old)
            print(json.dumps(row),flush=True)
    else:raise ValueError("真实波形压力编码尚未完成，不能写总完成结果")
    save(folder/"results.json",{"rows":rows,"development_diagnostic_only":True,"no_robustness_based_reselection":True,
        "waveform_limits":"all2876 development clips;center10s;Gaussian simulation,not real environment corpus;quality masks held fixed"})


if __name__=="__main__":main()
