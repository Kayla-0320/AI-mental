"""逐条回放、来源聚类区间与交付前核验，不据此再次选型。"""
from __future__ import annotations
import json
import pickle
from typing import Any
import numpy as np
from experiments.round14_common import ROOT,REPO,save,sha256
from experiments.round18_runtime import OUT,predict_neutral,predict_day,predict_quality_features
from experiments.round18_images import CONDITIONS
from experiments.round18_graph import validate,retrieve
from experiments.round10_fusion import folds
from experiments.round16_statistics import macro


def grouped_comparison(truth: np.ndarray,groups: np.ndarray,first: np.ndarray,second: np.ndarray) -> dict[str,Any]:
    """来源聚类配对自助法，旧来源反复曝光，区间仅探索性描述。"""
    ids=np.unique(groups);ix=np.random.default_rng(42).integers(0,len(ids),(3000,len(ids)))
    def cm(p: np.ndarray) -> np.ndarray:
        return np.array([np.bincount(3*truth[groups==g]+p.argmax(1)[groups==g],minlength=9).reshape(3,3) for g in ids])
    a,b=cm(first)[ix].sum(1),cm(second)[ix].sum(1)
    def values(c: np.ndarray) -> dict[str,np.ndarray]:
        denom=c.sum(-1)+c.sum(-2);n=2*c[:,1,1]/np.maximum(denom[:,1],1)
        return {"macro_f1":macro(c),"neutral_f1":n,"neutral_recall":c[:,1,1]/np.maximum(c[:,1].sum(1),1),"accuracy":np.trace(c,axis1=1,axis2=2)/c.sum((1,2))}
    av,bv=values(a),values(b)
    return {"groups":len(ids),"delta_ci95":{k:np.quantile(av[k]-bv[k],[.025,.975]).tolist() for k in av},"status":"exposed-source exploratory, no significance or independent-user claim"}


def run() -> None:
    """核对冻结模型接口、真实重编码头、跨日点预测和未审核图谱默认拒答。"""
    from experiments.round13_train import ch_data
    data=ch_data();data["mask"][:,1]=False
    z=np.load(OUT/"neutral/predictions.npz",allow_pickle=False)
    saved={c:np.load(ROOT/f"round16/moe/{c}_predictions.npz",allow_pickle=False) for c in ("TV","T_only","V_only")}
    bank=np.stack((saved["T_only"]["uniform"],saved["V_only"]["uniform"],saved["TV"]["ordinary_TV"]),1)
    rec=json.loads((ROOT/"round17/quality/raw_quality_v2.json").read_text(encoding="utf-8"));q=np.array([rec[str(i)]["score"] if rec[str(i)]["measured"] else np.nan for i in data["sample_ids"]])
    np.savez_compressed(OUT/"neutral/runtime_inputs.npz",bank=bank,mask=data["mask"],quality=q,labels=data["labels"],source_groups=data["source_groups"])
    replays=[]
    for fold,(_,va) in enumerate(folds(data)):
        p=predict_neutral(bank[va],data["mask"][va],q[va],OUT/f"neutral/fold{fold}");error=float(np.max(np.abs(p-z["selected"][va])));assert error<1e-12
        replays.append({"fold":fold,"samples":len(va),"max_error":error})
    save(OUT/"neutral/inference.json",{k:grouped_comparison(z["truth"],z["groups"],z["selected"],z[k]) for k in ("baseline","previous")})
    x=np.load(OUT/"images/reencoded.npz",allow_pickle=False);image_replays=[]
    predictions={c:np.zeros((len(x["labels"]),3,3)) for c in CONDITIONS}
    for fold,(_,va) in enumerate(folds({"labels":x["labels"],"source_groups":x["groups"]})):
        with (OUT/f"images/fold{fold}.pkl").open("rb") as stream:bundle=pickle.load(stream)
        for c in CONDITIONS:
            t=bundle["projectors"]["T"].transform(bundle["scalers"]["T"].transform(x["T"][va]))/np.sqrt(x["T"].shape[1])
            v=bundle["projectors"]["V"].transform(bundle["scalers"]["V"].transform(x[c][va]))/np.sqrt(x[c].shape[1])
            predictions[c][va]=np.stack([h.predict_proba(m) for h,m in zip(bundle["heads"],(t,v,np.column_stack((t,v))),strict=True)],1)
    for c in CONDITIONS:
        prior=np.load(OUT/f"images/{c}_predictions.npz",allow_pickle=False);guard=np.load(OUT/f"images/{c}_guard.npz",allow_pickle=False);b=predictions[c]
        quality=b.mean(1);drop=x[c+"_quality"]<=.51;quality[drop]=b[drop,0]
        conservative=b.mean(1);conservative[guard["drop"]]=b[guard["drop"],0]
        error=max(float(np.max(np.abs(actual-expected))) for actual,expected in ((b[:,2],prior["ordinary_TV"]),(b.mean(1),prior["uniform"]),(quality,prior["quality"]),(conservative,guard["selected"])))
        assert error<1e-12;image_replays.append({"condition":c,"samples":len(b),"max_error":error})
    windows=json.loads((OUT/"forecast/windows.json").read_text(encoding="utf-8"));actual=[]
    for row in windows:
        payload={"source_schema":"openesm:beck:sad:1_5","target_day":row["target_day"],"observations":[{"person":row["person"],"day":int(d),"value":1+4*v} for d,v in zip(row["history_days"],row["history"],strict=True)]}
        actual.append(predict_day(payload)["normalized_prediction"])
    forecast=np.load(OUT/"forecast/predictions.npz",allow_pickle=False);error=float(np.max(np.abs(actual-forecast["prediction"])));assert error<1e-12
    graph=json.loads((OUT/"knowledge_graph/graph.json").read_text(encoding="utf-8"));assert not validate(graph)
    assert not retrieve(graph,"锚定当下","act")["facts"]
    old=json.loads((ROOT/"round14/scope_and_promotion.json").read_text(encoding="utf-8"))["protected_current_sha256"]
    hashes={p:sha256(REPO/p) for p in old}
    quality_features=np.load(OUT/"quality_model/features.npz",allow_pickle=False);quality_saved=np.load(OUT/"quality_model/test_predictions.npz",allow_pickle=False)
    qp=predict_quality_features(quality_features["X"][960:],OUT/"quality_model");qerror=float(np.max(np.abs(qp-quality_saved["selected"])));assert qerror<1e-12
    calibration=json.loads((OUT/"forecast/calibration_lock.json").read_text(encoding="utf-8"));lo=np.maximum(0,forecast["prediction"]-calibration["constant_radius"]);hi=np.minimum(1,forecast["prediction"]+calibration["constant_radius"])
    covered=(forecast["truth"]>=lo)&(forecast["truth"]<=hi);people=forecast["people"];unique=np.unique(people);counts=np.array([(people==p).sum() for p in unique]);hits=np.array([covered[people==p].sum() for p in unique]);ix=np.random.default_rng(42).integers(0,len(unique),(3000,len(unique)))
    save(OUT/"forecast/coverage_inference.json",{"window_coverage_cluster_ci95":np.quantile(hits[ix].sum(1)/counts[ix].sum(1),[.025,.975]).tolist(),"person_equal_mean_coverage":float(np.mean(hits/counts)),"users_with_5plus_windows":int(np.sum(counts>=5)),"median_person_coverage_5plus":float(np.median((hits/counts)[counts>=5])),"scope":"exploratory user cluster interval, no individual coverage guarantee"})
    save(OUT/"verification.json",{"neutral":replays,"images":image_replays,"quality_model":{"test_images":len(qp),"max_error":qerror},"forecast":{"windows":len(windows),"max_error":error},"graph_structure_valid":True,"unreviewed_default_abstains":True,
        "protected_current_sha256":hashes,"protected_matches_round14":{p:hashes[p]==d for p,d in old.items()},"scope":"round18 edits only offline experiments and tests; existing workspace differences retained; no production promotion","human_ratings_generated":0})
    print(json.dumps({"neutral":replays,"images":image_replays,"forecast_error":error,"protected_matches_round14":{p:hashes[p]==d for p,d in old.items()}}),flush=True)


if __name__=="__main__":run()
