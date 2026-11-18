"""依据可观察质量和替代专家可信度切换；来源内选型，保留误伤统计。"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round13_train import ch_data
from experiments.round10_fusion import folds
from experiments.round17_quality import route,score,text_usable

OUT=ROOT/"round17/quality"


def guarded_mask(mask: np.ndarray,texts: np.ndarray,quality: np.ndarray,bank: np.ndarray,config: dict[str,float]) -> np.ndarray:
    """硬缺失直接降级，软质量问题只在文字替代专家足够明确时切换。"""
    result=mask.copy();result[:,1]=False
    result[:,0]&=np.array([text_usable(t) for t in texts])
    hard=np.isfinite(quality)&(quality<=0)
    soft=np.isfinite(quality)&(quality>0)&(quality<=config["threshold"])&result[:,0]&(bank[:,0].max(1)>=config["text_confidence"])
    result[:,2]&=~(hard|soft)
    return result


def run() -> None:
    """与旧均值和旧质量路由逐项比较，质量判断不消费图文情绪一致性。"""
    options=[{"threshold":0.,"text_confidence":0.}]+[{"threshold":t,"text_confidence":c} for t in (.25,.51) for c in (0.,.6,.8)]
    freeze(OUT/"switch_protocol.json",{"options":options,"source_sha256":sha256(Path(__file__)),"quality_source":"raw_quality_v2.json; labels never used to measure pixels",
        "fit":"source-inner OOF, held source excluded, F1 minus switch/harm penalty; clean accuracy guard .005 where feasible",
        "status":"previously exposed CH development diagnostic; not independent external validation"})
    data=ch_data();data["mask"][:,1]=False
    records=json.loads((OUT/"raw_quality_v2.json").read_text(encoding="utf-8"));q=np.array([records[str(i)]["score"] if records[str(i)]["measured"] else np.nan for i in data["sample_ids"]])
    old={c:np.load(ROOT/f"round16/moe/{c}_predictions.npz",allow_pickle=False) for c in ("TV","T_only","V_only")}
    bank=np.stack((old["T_only"]["uniform"],old["V_only"]["uniform"],old["TV"]["ordinary_TV"]),1)
    output=np.zeros((len(q),3));locks=[];all_masks=data["mask"].copy()
    for fold,(tr,va) in enumerate(folds(data)):
        cache=np.load(ROOT/f"round13/neural/ch/fold{fold}/inner_oof.npz",allow_pickle=False)["missing_A"][:,[0,3,4],:]
        with (ROOT/f"round16/moe/export/fold{fold}/experts.pkl").open("rb") as stream:prior=pickle.load(stream)["prior"]
        original=route(cache,data["mask"][tr],prior);base=score(data["labels"][tr],original);candidates=[]
        for config in options:
            m=guarded_mask(data["mask"][tr],data["raw_text"][tr],q[tr],cache,config);p=route(cache,m,prior)
            metrics=score(data["labels"][tr],p);switch=float(np.mean(np.any(m!=data["mask"][tr],1)))
            harm=float(np.mean((original.argmax(1)==data["labels"][tr])&(p.argmax(1)!=data["labels"][tr])))
            candidates.append({"config":config,**metrics,"switch_rate":switch,"harm_rate":harm,"utility":metrics["macro_f1"]-.005*harm-.001*switch})
        eligible=[r for r in candidates if r["accuracy"]>=base["accuracy"]-.005]
        winner=max(eligible or candidates[:1],key=lambda r:(r["utility"],-r["switch_rate"]))
        m=guarded_mask(data["mask"][va],data["raw_text"][va],q[va],bank[va],winner["config"])
        output[va]=route(bank[va],m,prior);all_masks[va]=m
        locks.append({"fold":fold,"winner":winner,"candidates":candidates,"accuracy_guard_feasible":bool(eligible),
            "train_groups":np.unique(data["source_groups"][tr]).tolist(),"held_groups":np.unique(data["source_groups"][va]).tolist()})
    baseline=old["TV"]["uniform"];y=data["labels"]
    result={"baseline":score(y,baseline),"guarded":score(y,output),"switch_count":int(np.sum(np.any(all_masks!=data["mask"],1))),
        "rescued":int(np.sum((baseline.argmax(1)!=y)&(output.argmax(1)==y))),"introduced_errors":int(np.sum((baseline.argmax(1)==y)&(output.argmax(1)!=y))),"samples":len(y)}
    save(OUT/"guarded_selection_locks.json",locks);save(OUT/"guarded_results.json",result)
    np.savez_compressed(OUT/"guarded_predictions.npz",truth=y,groups=data["source_groups"],sample_ids=data["sample_ids"],selected=output,baseline=baseline,quality=q,used_mask=all_masks,original_mask=data["mask"])
    print(json.dumps(result),flush=True)


if __name__=="__main__":run()
