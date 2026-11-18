"""优先保护中性F1与召回的元分类和专家切换，外层来源只用于一次诊断。"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score,recall_score,accuracy_score
from scipy.special import softmax
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round13_train import ch_data
from experiments.round10_fusion import folds,take
from experiments.round17_correction import features
from experiments.round17_quality import route

OUT=ROOT/"round18"


def metrics(y: np.ndarray,p: np.ndarray) -> dict[str,Any]:
    """固定三类评价，单列中性召回及F1，避免准确率掩盖中性损失。"""
    pred=p.argmax(1);fs=f1_score(y,pred,labels=[0,1,2],average=None,zero_division=0)
    return {"macro_f1":float(fs.mean()),"neutral_f1":float(fs[1]),"neutral_recall":float(recall_score(y,pred,labels=[1],average="macro",zero_division=0)),
        "accuracy":float(accuracy_score(y,pred)),"class_f1":fs.tolist()}


def apply_policy(base: np.ndarray,meta: np.ndarray,config: dict[str,float]) -> np.ndarray:
    """中性离开门槛仅针对分类决策，不是心理安全或危机阈值。"""
    p=(1-config["weight"])*base+config["weight"]*meta
    logits=np.log(np.maximum(p,1e-9));logits[:,1]+=config["bias"];p=softmax(logits,axis=1)
    old,new=base.argmax(1),p.argmax(1)
    blocked=(old==1)&(new!=1)&((p[np.arange(len(p)),new]-p[:,1])<config["leave_margin"])
    p[blocked]=base[blocked];return p


def fit(x: np.ndarray,y: np.ndarray,neutral_weight: float) -> Any:
    """使用固定容量分类器，权重只在外层训练来源中拟合。"""
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=3.,learning_rate=.05,random_state=42)
    return model.fit(x,y,sample_weight=np.where(y==1,neutral_weight,1.))


def run() -> None:
    """中性F1/召回守门条件下选型，所有被曝光数据均明确作为开发诊断。"""
    folder=OUT/"neutral";folder.mkdir(parents=True,exist_ok=True)
    freeze(folder/"protocol.json",{"seed":42,"neutral_weights":[1.,1.5,2.],"blend_weights":[.25,.5,.75,1.],"neutral_bias":[-.2,0.,.2],"leave_margin":[0.,.1,.2],
        "selection":".7macroF1+.3neutralF1; neutralF1>=base, neutral recall>=base-.01, accuracy>=base-.002, else exact base",
        "source_sha256":sha256(Path(__file__)),"status":"exposed CH source-development diagnostic; no new independent effectiveness claim"})
    data=ch_data();data["mask"][:,1]=False;qrec=json.loads((ROOT/"round17/quality/raw_quality_v2.json").read_text(encoding="utf-8"))
    q=np.array([qrec[str(i)]["score"] if qrec[str(i)]["measured"] else np.nan for i in data["sample_ids"]])
    saved={c:np.load(ROOT/f"round16/moe/{c}_predictions.npz",allow_pickle=False) for c in ("TV","T_only","V_only")}
    outerbank=np.stack((saved["T_only"]["uniform"],saved["V_only"]["uniform"],saved["TV"]["ordinary_TV"]),1)
    result=np.zeros((len(q),3));locks=[]
    for fold,(tr,va) in enumerate(folds(data)):
        train=take(data,tr);inner=np.load(ROOT/f"round13/neural/ch/fold{fold}/inner_oof.npz",allow_pickle=False)["missing_A"][:,[0,3,4],:]
        with (ROOT/f"round16/moe/export/fold{fold}/experts.pkl").open("rb") as stream:prior=pickle.load(stream)["prior"]
        base=route(inner,train["mask"],prior);bm=metrics(train["labels"],base);x=features(inner,train["mask"],q[tr]);options=[]
        for nw in (1.,1.5,2.):
            oof=np.zeros_like(base)
            for a,b in folds(train,3):oof[b]=fit(x[a],train["labels"][a],nw).predict_proba(x[b])
            for weight in (.25,.5,.75,1.):
                for bias in (-.2,0.,.2):
                    for margin in (0.,.1,.2):
                        c={"neutral_weight":nw,"weight":weight,"bias":bias,"leave_margin":margin};m=metrics(train["labels"],apply_policy(base,oof,c))
                        if m["neutral_f1"]>=bm["neutral_f1"] and m["neutral_recall"]>=bm["neutral_recall"]-.01 and m["accuracy"]>=bm["accuracy"]-.002:
                            options.append({"config":c,"metrics":m,"utility":.7*m["macro_f1"]+.3*m["neutral_f1"]})
        fallback={"config":{"neutral_weight":1.,"weight":0.,"bias":0.,"leave_margin":0.},"metrics":bm,"utility":.7*bm["macro_f1"]+.3*bm["neutral_f1"]}
        winner=max([fallback]+options,key=lambda r:r["utility"])
        model=fit(x,train["labels"],winner["config"]["neutral_weight"])
        base_outer=saved["TV"]["uniform"][va];meta=model.predict_proba(features(outerbank[va],data["mask"][va],q[va]))
        result[va]=apply_policy(base_outer,meta,winner["config"])
        path=folder/f"fold{fold}";path.mkdir(exist_ok=True)
        with (path/"model.pkl").open("wb") as stream:pickle.dump(model,stream)
        save(path/"policy.json",{"config":winner["config"],"sha256":sha256(path/"model.pkl")})
        locks.append({"fold":fold,"winner":winner,"eligible_candidates":len(options),"base_inner":bm,"train_groups":np.unique(train["source_groups"]).tolist(),"held_groups":np.unique(data["source_groups"][va]).tolist()})
        print(json.dumps({"neutral_fold":fold,"config":winner["config"]}),flush=True)
    y=data["labels"];base=saved["TV"]["uniform"];old=np.load(ROOT/"round17/quality/correction_predictions.npz",allow_pickle=False)["selected"]
    save(folder/"selection_locks.json",locks)
    np.savez_compressed(folder/"predictions.npz",truth=y,groups=data["source_groups"],sample_ids=data["sample_ids"],baseline=base,previous=old,selected=result)
    report={"baseline":metrics(y,base),"previous":metrics(y,old),"selected":metrics(y,result),"rescued":int(np.sum((base.argmax(1)!=y)&(result.argmax(1)==y))),
        "introduced_errors":int(np.sum((base.argmax(1)==y)&(result.argmax(1)!=y))),"neutral_exits":int(np.sum((base.argmax(1)==1)&(result.argmax(1)!=1)))}
    save(folder/"results.json",report);print(json.dumps(report),flush=True)


if __name__=="__main__":run()
