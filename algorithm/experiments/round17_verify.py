"""第十七轮用户聚类统计和两个运行时的逐项核验，不用统计结果再选型。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from experiments.round14_common import ROOT,save
from experiments.round17_forecast import OUT,SOURCES
from experiments.round17_runtime import verify_forecast,predict_tv_quality
from experiments.round16_statistics import holm,macro


def run() -> None:
    """报告MAE和RMSE的配对区间，同时记录新的代价。"""
    runtime=verify_forecast();comparisons=[]
    for name in SOURCES:
        z=np.load(OUT/f"forecast/{name}/predictions.npz",allow_pickle=False);y=z["truth"];g=z["people"];people=np.unique(g)
        count=np.array([(g==p).sum() for p in people]);ix=np.random.default_rng(42).integers(0,len(people),(5000,len(people)))
        for baseline in ("previous_ema","previous_robust","last","mean7"):
            delta=np.array([(np.abs(z["selected"][g==p]-y[g==p])-np.abs(z[baseline][g==p]-y[g==p])).mean() for p in people])
            first=np.array([np.square(z["selected"][g==p]-y[g==p]).sum() for p in people]);second=np.array([np.square(z[baseline][g==p]-y[g==p]).sum() for p in people])
            rmse=100*(np.sqrt(first[ix].sum(1)/count[ix].sum(1))-np.sqrt(second[ix].sum(1)/count[ix].sum(1)))
            signs=np.random.default_rng(42).choice([-1,1],(5000,len(people)));p=float((1+np.sum(np.abs((signs*delta).mean(1))>=abs(delta.mean())))/5001)
            comparisons.append({"dataset":name,"baseline":baseline,"person_mae_delta":float(delta.mean()*100),"person_mae_ci95":(np.quantile(delta[ix].mean(1),[.025,.975])*100).tolist(),
                "rmse_delta_ci95":np.quantile(rmse,[.025,.975]).tolist(),"person_mae_cluster_signflip_p":p})
    for row,p in zip(comparisons,holm([r["person_mae_cluster_signflip_p"] for r in comparisons]),strict=True):row["holm_p_8_mae_comparisons"]=p
    save(OUT/"forecast/inference.json",{"comparisons":comparisons,"status":"exploratory inference, 5000 user-cluster bootstrap, seed42, no test-driven retuning"})
    from experiments.round13_train import ch_data
    from experiments.round10_fusion import folds
    data=ch_data();z=np.load(OUT/"quality/guarded_predictions.npz",allow_pickle=False)
    policies=json.loads((OUT/"quality/guarded_selection_locks.json").read_text(encoding="utf-8"));replays=[]
    from experiments.round17_correction import predict_correction
    import pickle
    expert_cache={c:np.load(ROOT/f"round16/moe/{c}_predictions.npz",allow_pickle=False) for c in ("TV","T_only","V_only")}
    bank=np.stack((expert_cache["T_only"]["uniform"],expert_cache["V_only"]["uniform"],expert_cache["TV"]["ordinary_TV"]),1)
    correction=np.load(OUT/"quality/correction_predictions.npz",allow_pickle=False);correction_replays=[]
    for fold,(_,va) in enumerate(folds(data)):
        p,meta=predict_tv_quality(data["T"][va],data["V"][va],data["mask"][va][:,[0,2]],data["raw_text"][va],z["quality"][va],ROOT/f"round16/moe/export/fold{fold}",policies[fold]["winner"]["config"])
        error=float(np.max(np.abs(p-z["selected"][va])));assert error<1e-12
        assert np.array_equal(meta["mask"],z["used_mask"][va])
        replays.append({"fold":fold,"samples":len(va),"max_error":error,"executed_pairs":meta["executed_sample_head_pairs"],"audio_heads":meta["audio_heads"]})
        with (ROOT/f"round16/moe/export/fold{fold}/experts.pkl").open("rb") as stream:prior=pickle.load(stream)["prior"]
        mask=data["mask"][va].copy();mask[:,1]=False
        cp=predict_correction(bank[va],mask,z["quality"][va],OUT/f"quality/correction_models/fold{fold}",prior)
        ce=float(np.max(np.abs(cp-correction["selected"][va])));assert ce<1e-12
        correction_replays.append({"fold":fold,"samples":len(va),"max_error":ce})
    groups=z["groups"];y=z["truth"];unique=np.unique(groups);ix=np.random.default_rng(42).integers(0,len(unique),(3000,len(unique)))
    cms={}
    for method in ("baseline","selected"):
        pred=z[method].argmax(1);cms[method]=np.array([np.bincount(3*y[groups==p]+pred[groups==p],minlength=9).reshape(3,3) for p in unique])
    diff=macro(cms["selected"][ix].sum(1))-macro(cms["baseline"][ix].sum(1))
    probes=json.loads((OUT/"quality/detector_probes_v2.json").read_text(encoding="utf-8"));probe_summary={}
    for condition in sorted({r["condition"] for r in probes}):
        rows=[r for r in probes if r["condition"]==condition]
        probe_summary[condition]={"n":len(rows),"old_flag_rate_at_0.51":float(np.mean([r["old_score"]<=.51 for r in rows])),"new_flag_rate_at_0.51":float(np.mean([r["new_score"]<=.51 for r in rows]))}
    save(OUT/"quality/verification.json",{"replays":replays,"macro_f1_delta_ci95":np.quantile(diff,[.025,.975]).tolist(),"detector_probes":probe_summary,
        "probe_classifier_reencoded":False,"clean_probe_flag_rate_is_not_human_false_positive_rate":True})
    corrected=correction["selected"].argmax(1);base=correction["baseline"].argmax(1)
    cc=np.array([np.bincount(3*y[groups==p]+corrected[groups==p],minlength=9).reshape(3,3) for p in unique]);bc=cms["baseline"]
    correction_f1=macro(cc[ix].sum(1))-macro(bc[ix].sum(1))
    a,b=cc[ix].sum(1),bc[ix].sum(1)
    accuracy=np.trace(a,axis1=1,axis2=2)/a.sum((1,2))-np.trace(b,axis1=1,axis2=2)/b.sum((1,2))
    old_gate=expert_cache["TV"]["learned"].argmax(1)
    save(OUT/"quality/correction_verification.json",{"replays":correction_replays,"macro_f1_delta_ci95":np.quantile(correction_f1,[.025,.975]).tolist(),
        "accuracy_delta_ci95":np.quantile(accuracy,[.025,.975]).tolist(),"old_learned_gate_vs_same_uniform":{"rescued":int(np.sum((base!=y)&(old_gate==y))),"introduced_errors":int(np.sum((base==y)&(old_gate!=y)))},"status":"exposed-source exploratory, no default replacement"})
    save(OUT/"runtime_verification.json",{"forecast":runtime,"quality":replays,"correction":correction_replays})
    print(json.dumps({"forecast_ci":comparisons,"quality":replays,"quality_f1_ci":np.quantile(diff,[.025,.975]).tolist(),"probes":probe_summary},ensure_ascii=False),flush=True)


if __name__=="__main__":run()
