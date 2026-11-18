"""来源开发锁定后拟合3579条与五种子模型，历史集只做一次诊断。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
from typing import Any
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
from scipy.special import softmax
from catboost import CatBoostClassifier
from experiments.round11_common import OUTPUT,PREVIOUS,digest,save
from experiments.round11_features import alignment_feature
from experiments import round11_evidence as evidence
from experiments.round10_fusion import folds,take
from experiments.round10_domain import weights_for
from experiments.round10_large_speech import data_for
from experiments.round7_decision_oof import metric,probability_metrics


def load_data(splits: tuple[str,...],mode: str = "7b",quality: str = "pts") -> dict[str,np.ndarray]:
    """新旧缓存逐ID核对，纯输入修正与主任务标签不发生互相依赖。"""
    data=data_for(splits);ids=data["sample_ids"]
    for component in ("msp_complete_features","sensevoice_deterministic_features"):
        parts=[]
        for split in splits:
            with np.load(OUTPUT/component/f"{split}.npz",allow_pickle=False) as z:parts.append({k:z[k] for k in z.files})
        if not np.array_equal(ids,np.concatenate([p["sample_ids"] for p in parts])):raise ValueError("最终输入ID不符")
        if component=="msp_complete_features":
            for dst,src in (("speech","speech"),("speech_emotion","speech_emotion"),("speech_available","available")):
                data[dst]=np.concatenate([p[src] for p in parts])
        else:
            data["sensevoice"]=np.concatenate([p["sensevoice"] for p in parts])
            data["sensevoice_query"]=np.concatenate([p["query"][:,1,:] for p in parts])
            data["sensevoice_emotion"]=np.concatenate([p["emotion_logprobs"][:,1,:] for p in parts])
            data["sensevoice_available"]=np.concatenate([p["available"] for p in parts])
            texts=np.concatenate([p["transcript"] for p in parts])
            data["asr_alignment"]=np.stack([alignment_feature(str(a),str(b)) for a,b in zip(data["raw_text"],texts)])
    if quality=="pts":
        masks=[];qualities=[]
        for split in splits:
            filename="development_masks.npz" if split in ("train","selection") else f"{split}_masks.npz"
            with np.load(OUTPUT/"input_audit"/filename,allow_pickle=False) as z:
                lookup={str(k):i for i,k in enumerate(z["sample_ids"])}
                with np.load(PREVIOUS/"llm_large_features"/f"{split}.npz",allow_pickle=False) as old:order=[lookup[str(k)] for k in old["sample_ids"]]
                masks.append(z["modality_mask"][order]);qualities.append(z["quality"][order])
        data["modality_mask"]=np.concatenate(masks);data["quality"]=np.concatenate(qualities)
    if mode=="3b":
        for field in ("llm_last","llm_mean","llm_scores"):
            parts=[]
            for split in splits:
                with np.load(PREVIOUS/"llm_features"/f"{split}.npz",allow_pickle=False) as z:parts.append(z[field])
            data[field]=np.concatenate(parts)
    return data


def predict(bundle: dict[str,Any],data: dict[str,np.ndarray]) -> np.ndarray:
    """预测无需金标，五种子与集成单列。"""
    x=evidence.expert_evidence(bundle["experts"],data)
    if bundle["meta"]["reliability"]:x=np.column_stack((x,evidence.reliability(data)))
    result=np.stack([softmax(np.log(np.maximum(model.predict_proba(x),1e-12))+bundle["bias"],axis=1) for model in bundle["meta_models"]])
    if bundle.get("missing_route",{}).get("promoted",False):
        from experiments.round11_missing_route import route
        missing=~(data["modality_mask"][:,0]&(data["quality"][:,0]>0))
        result=route(result,x,missing,bundle["missing_route"]["winner"]["audio_weight"],bundle["bias"],bundle["training_prior"])
    return result


def fit(mode: str,destination: Path | None = None) -> None:
    """来源选型冻结，703仅用于最后拟合，821完全不参与。"""
    output=OUTPUT/"final_models"/mode if destination is None else destination
    output.mkdir(parents=True,exist_ok=True)
    if (output/"model_manifest.json").exists():return
    family=OUTPUT/("evidence_deterministic" if mode=="7b" else f"deployment_{mode}/evidence_deterministic")/"family10"
    rows=json.loads((family/"complete.json").read_text("utf-8"))["results"]
    winner=max(rows,key=lambda r:r["selection_score"])
    bias=winner["bias"]
    if mode=="7b":bias=json.loads((OUTPUT/"blend/selection_locked.json").read_text("utf-8"))["winner"]["bias"]
    policy_path=OUTPUT/"missing_route/selection_locked.json"
    policy=json.loads(policy_path.read_text("utf-8"))
    plan={"mode":mode,"family":winner["config"],"meta":winner["meta"],"bias":bias,"seeds":list(range(42,47)),
        "missing_route_sha256":digest(policy_path),
        "inner_folds_final":4,"train_splits":["train","selection","calibration"],"test_used":False,
        "code_sha256":digest(Path(__file__)),"selection_sha256":digest(family/"complete.json")}
    lock=output/"selection_locked.json"
    if lock.exists():
        if json.loads(lock.read_text("utf-8"))!=plan:raise ValueError("最终拟合计划不符")
    else:save(lock,plan)
    evidence.TEXT=("aux_mean","bge_cls") if mode=="light" else ("llm_last","aux_mean","bge_cls")
    data=load_data(("train","selection","calibration"),mode)
    inner_path=output/"inner_oof.npz"
    if inner_path.exists():
        with np.load(inner_path,allow_pickle=False) as z:
            if not np.array_equal(z["sample_ids"],data["sample_ids"]):raise ValueError("最终OOF身份不符")
            inner=z["evidence"]
    else:
        inner=None;fold_ids=np.zeros(len(data["sample_ids"]),int)
        for f,(tr,va) in enumerate(folds(data,4)):
            models=evidence.fit_experts(take(data,tr),winner["config"])
            value=evidence.expert_evidence(models,take(data,va))
            if inner is None:inner=np.full((len(data["sample_ids"]),value.shape[1]),np.nan)
            inner[va]=value;fold_ids[va]=f
            print(json.dumps({"mode":mode,"final_inner_fold":f}),flush=True)
        if not np.isfinite(inner).all():raise ValueError("最终OOF覆盖不足")
        np.savez_compressed(inner_path,evidence=inner,sample_ids=data["sample_ids"],source_groups=data["source_groups"],fold_ids=fold_ids)
    x=np.column_stack((inner,evidence.reliability(data))) if winner["meta"]["reliability"] else inner
    weight=weights_for(data["labels"],data["source_groups"],0.,.5)
    heads=[]
    for seed in plan["seeds"]:
        model=CatBoostClassifier(iterations=300,depth=winner["meta"]["depth"],learning_rate=.03,l2_leaf_reg=3.,
            loss_function="MultiClass",random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
        heads.append(model.fit(x,data["labels"],sample_weight=weight))
    bundle={"experts":evidence.fit_experts(data,winner["config"]),"meta_models":heads,"meta":winner["meta"],"bias":bias,
        "config":winner["config"],"mode":mode,"training_ids":data["sample_ids"],"training_groups":np.unique(data["source_groups"]),
        "missing_route":policy,"training_prior":np.bincount(data["labels"],minlength=3)/len(data["labels"])}
    path=output/"model.pkl"
    with path.open("wb") as handle:pickle.dump(bundle,handle)
    sample=take(data,np.arange(min(16,len(data["sample_ids"]))))
    p=predict(bundle,sample)
    with path.open("rb") as handle:again=predict(pickle.load(handle),sample)
    if not np.array_equal(p,again):raise ValueError("最终模型重放不一致")
    save(output/"model_manifest.json",{"training_n":len(data["sample_ids"]),"n_groups":len(bundle["training_groups"]),
        "sha256":digest(path),"selection_sha256":digest(lock),"seeds":plan["seeds"],"bytes":path.stat().st_size,
        "reload_max_abs_error":float(np.abs(p-again).max()),"test_used":False})
    print(json.dumps({"final_model":mode,"sha256":digest(path)}),flush=True)


def evaluate() -> None:
    """唯一一次821历史诊断；不能因结果不好再用该集反复改型。"""
    output=OUTPUT/"historical_diagnostic"
    if (output/"results.json").exists():return
    files=[OUTPUT/"final_models"/m/"model.pkl" for m in ("7b","3b","light")]
    lock={"all_selection_complete":True,"task":"historical unchanged821 M3;previously exposed",
        "models":[{"path":str(p),"sha256":digest(p)} for p in files],"code_sha256":digest(Path(__file__))}
    lock_path=output/"evaluation_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text("utf-8"))!=lock:raise ValueError("历史诊断锁后配方改变")
    else:save(lock_path,lock)
    original=data_for(("test",));ids=original["sample_ids"];y=original["labels"]
    results={}
    for mode,path in zip(("7b","3b","light"),files):
        data=load_data(("test",),mode)
        if not np.array_equal(ids,data["sample_ids"]) or not np.array_equal(y,data["labels"]):raise ValueError("821样本或金标改变")
        with path.open("rb") as handle:bundle=pickle.load(handle)
        if set(bundle["training_groups"])&set(data["source_groups"]):raise ValueError("最终训练测试来源重合")
        p=predict(bundle,{k:v for k,v in data.items() if k!="labels"})
        rows=[{"seed":s,**metric(y,prob.argmax(1)),**probability_metrics(y,prob)} for s,prob in zip(range(42,47),p)]
        results[mode]={"mean_macro_f1":float(np.mean([r["macro_f1"] for r in rows])),"std_macro_f1":float(np.std([r["macro_f1"] for r in rows])),
            "seeds":rows,"ensemble":metric(y,p.mean(0).argmax(1)),"n":len(y),"coverage":1.}
        np.savez_compressed(output/f"{mode}_predictions.npz",probabilities=p,labels=y,sample_ids=ids,source_groups=data["source_groups"])
    old=json.loads((PREVIOUS/"refined_final/results.json").read_text("utf-8"))
    results["round10_mean_macro_f1"]=old["mean_macro_f1"]
    results["main_relative_gain_vs_round10"]=results["7b"]["mean_macro_f1"]/old["mean_macro_f1"]-1
    results["original_round9_reported_baseline"]=.6388
    results["original_requested_threshold"]=.70268
    results["round11_relative_gain_vs_original_round9"]=results["7b"]["mean_macro_f1"]/.6388-1
    results["round11_meets_original_10percent_threshold"]=results["7b"]["mean_macro_f1"]>=.70268
    results["preserved_round10_meets_original_10percent_threshold"]=old["mean_macro_f1"]>=.70268
    results["historical_preexposed"]=True
    save(output/"results.json",results);print(json.dumps(results),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("fit","evaluate"));parser.add_argument("--mode",choices=("7b","3b","light"),default="7b")
    parser.add_argument("--destination",type=Path)
    args=parser.parse_args();fit(args.mode,args.destination) if args.action=="fit" else evaluate()
