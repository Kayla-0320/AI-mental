"""外部用户聚类检验、多重比较校正和MoE得失诊断，均不反向用于选型。"""
from __future__ import annotations
import json
import numpy as np
from experiments.round16_external import OUT,SOURCES
from experiments.round14_common import save


def holm(values: list[float]) -> list[float]:
    """Holm逐步校正，返回与输入同序的校正p值。"""
    order=np.argsort(values);output=np.ones(len(values));running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(values)-rank)*values[index]));output[index]=running
    return output.tolist()


def macro(cm: np.ndarray) -> np.ndarray:
    """批量混淆矩阵的三类宏F1，固定类别全集。"""
    diagonal=np.diagonal(cm,axis1=-2,axis2=-1)
    denominator=cm.sum(-1)+cm.sum(-2)
    return np.mean(np.divide(2*diagonal,denominator,out=np.zeros_like(diagonal,dtype=float),where=denominator>0),axis=-1)


def run() -> None:
    """不把外部统计作为再次挑选预测器或路由的依据。"""
    comparisons=[];rmse_ci={}
    for name in SOURCES:
        z=np.load(OUT/f"forecast/{name}/predictions.npz",allow_pickle=False);g=z["people"];y=z["truth"];people=np.unique(g)
        ix=np.random.default_rng(42).integers(0,len(people),(5000,len(people)));counts=np.array([(g==p).sum() for p in people])
        for candidate in ("source_selected","robust"):
            for baseline in ("last","mean7","previous_ema"):
                d=np.array([(np.abs(z[candidate][g==p]-y[g==p])-np.abs(z[baseline][g==p]-y[g==p])).mean() for p in people])
                signs=np.random.default_rng(42).choice([-1,1],size=(5000,len(people)))
                pvalue=float((1+np.sum(np.abs((signs*d).mean(1))>=abs(d.mean())))/5001)
                comparisons.append({"dataset":name,"candidate":candidate,"baseline":baseline,"person_mae_delta":float(100*d.mean()),"two_sided_cluster_signflip_p":pvalue})
                first=np.array([np.square(z[candidate][g==p]-y[g==p]).sum() for p in people]);second=np.array([np.square(z[baseline][g==p]-y[g==p]).sum() for p in people])
                delta=100*(np.sqrt(first[ix].sum(1)/counts[ix].sum(1))-np.sqrt(second[ix].sum(1)/counts[ix].sum(1)))
                rmse_ci[f"{name}/{candidate}/{baseline}"]=np.quantile(delta,[.025,.975]).tolist()
    corrected=holm([r["two_sided_cluster_signflip_p"] for r in comparisons])
    for row,pvalue in zip(comparisons,corrected,strict=True):row["holm_p_12_comparisons"]=pvalue
    save(OUT/"forecast/inference.json",{"comparison_family":12,"person_mae_comparisons":comparisons,"rmse_exploratory_ci95":rmse_ci,
        "interpretation":"prelocked predictors on first-use sources; post-evaluation statistical family declaration; sign-flip exchangeability assumption; no confirmatory preregistered clinical claim"})
    moe={}
    for condition in ("TV","T_only","V_only"):
        z=np.load(OUT/f"moe/{condition}_predictions.npz",allow_pickle=False);y=z["truth"];groups=z["groups"];people=np.unique(groups)
        ix=np.random.default_rng(42).integers(0,len(people),(3000,len(people)));old=z["ordinary_TV"].argmax(1)
        def matrices(prediction: np.ndarray) -> np.ndarray:
            """按来源记录混淆矩阵，保持同来源配对。"""
            return np.array([np.bincount(3*y[groups==p]+prediction[groups==p],minlength=9).reshape(3,3) for p in people])
        old_cm=matrices(old);moe[condition]={}
        for name in ("uniform","learned","anchored"):
            new=z[name].argmax(1);cm=matrices(new)
            delta=macro(cm[ix].sum(1))-macro(old_cm[ix].sum(1))
            moe[condition][name]={"rescued_errors":int(np.sum((old!=y)&(new==y))),"newly_introduced_errors":int(np.sum((old==y)&(new!=y))),
                "macro_f1_delta_ci95":np.quantile(delta,[.025,.975]).tolist(),"source_groups":len(people)}
    save(OUT/"moe/complementarity.json",{"conditions":moe,"status":"descriptive exposed-source analysis, not new independent validation","no_oracle_used_at_inference":True})
    print(json.dumps({"external_comparisons":comparisons,"moe":moe},ensure_ascii=False),flush=True)


if __name__=="__main__":run()
