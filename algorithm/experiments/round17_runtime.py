"""第十七轮冻结预测与质量切换的可调用离线运行时，原始量表和资格严格校验。"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np
from experiments.round14_common import ROOT,save,sha256
from experiments.round17_forecast import OUT,SOURCES,balanced_forecast
from experiments.round17_switch import guarded_mask
from experiments.round13_models import matrix,probabilities


def predict_day(payload: dict[str,Any]) -> dict[str,Any]:
    """只允许两个指定来源量表，检查未来泄漏、日序和长缺测后的冷启动。"""
    name=payload.get("dataset")
    if name not in SOURCES:raise ValueError("未验证来源")
    spec=SOURCES[name];schema=f'openesm:{name}:{spec["target"]}:{spec["low"]}_{spec["high"]}'
    if payload.get("source_schema")!=schema:raise ValueError("量表不匹配，不接受项目心情或压力自动迁移")
    target=payload.get("target_day");rows=payload.get("observations",[])
    if type(target) is not int or not rows:raise ValueError("缺少整数目标日或过去观测")
    if len({str(r.get("person","")) for r in rows})!=1 or not str(rows[0].get("person","")):raise ValueError("历史须属于同一用户")
    days=[];values=[]
    for row in rows:
        d,v=row.get("day"),row.get("value")
        if type(d) is not int or d>=target:raise ValueError("日序无效或混入未来/目标日")
        if type(v) not in {float,int} or not np.isfinite(v) or not spec["low"]<=v<=spec["high"]:raise ValueError("原量表值无效")
        days.append(d);values.append((v-spec["low"])/(spec["high"]-spec["low"]))
    if days!=sorted(set(days)):raise ValueError("日序重复或未递增")
    if days[-1]!=target-1:return {"status":"abstain","reason":"latest_day_missing"}
    indices=[i for i,d in enumerate(days) if days[-1]-14<d];days=[days[i] for i in indices];values=[values[i] for i in indices]
    gaps=[i for i in range(1,len(days)) if days[i]-days[i-1]>3]
    if gaps:values=values[gaps[-1]:]
    if len(values)<3:return {"status":"abstain","reason":"insufficient_history_after_gap_reset"}
    policy=json.loads((OUT/"forecast/selection_lock.json").read_text(encoding="utf-8"));p=balanced_forecast(values,policy["winner"])
    return {"status":"research_prediction","normalized_prediction":p,"prediction":spec["low"]+p*(spec["high"]-spec["low"]),
        "method":policy["winner"],"clinical_alert":False,"automatic_intervention":False}


def predict_tv_quality(text: np.ndarray,visual: np.ndarray,mask: np.ndarray,texts: np.ndarray,quality: np.ndarray,folder: Path,config: dict[str,float]) -> tuple[np.ndarray,dict[str,Any]]:
    """依据原始质量测量和文字专家信心选择合格专家，只执行有资格的分类头。"""
    n=len(text)
    if mask.shape!=(n,2) or mask.dtype!=bool or len(visual)!=n or len(texts)!=n or quality.shape!=(n,):raise ValueError("图文输入形状或可用性无效")
    if np.any(np.isinf(quality)) or np.any((quality[np.isfinite(quality)]<0)|(quality[np.isfinite(quality)]>1)):raise ValueError("质量必须是0–1可观察分数，未知为NaN")
    for values,active in ((text,mask[:,0]),(visual,mask[:,1])):
        if values.ndim!=2 or not np.isfinite(values[active]).all():raise ValueError("可见特征不合法")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    for name,digest in manifest["files"].items():
        if sha256(folder/name)!=digest:raise ValueError("专家摘要不一致")
    with (folder/"projection.pkl").open("rb") as stream:projection=pickle.load(stream)
    with (folder/"experts.pkl").open("rb") as stream:bundle=pickle.load(stream)
    if projection.members["A"] is not None or set(bundle["models"])!={1,4,5}:raise ValueError("不得含音频专家")
    original=np.column_stack((mask[:,0],np.zeros(n,bool),mask[:,1]));packed={"T":text,"V":visual,"A":np.zeros((n,1)),"mask":original}
    initial=projection.transform(packed);bank=np.tile(bundle["prior"],(n,3,1));active=original[:,0]
    if active.any():bank[active,0]=probabilities(bundle["models"][1],matrix({k:v[active] for k,v in initial.items()},1,None))
    selected=guarded_mask(original,texts,quality,bank,config);packed["mask"]=selected;projected=projection.transform(packed)
    code=selected.astype(int)@np.array([1,2,4]);eligible=np.column_stack([(code&p)==p for p in (1,4,5)])
    calls=int(active.sum())
    for j,p in ((1,4),(2,5)):
        ix=eligible[:,j]
        if ix.any():bank[ix,j]=probabilities(bundle["models"][p],matrix({k:v[ix] for k,v in projected.items()},p,None));calls+=int(ix.sum())
    weights=eligible/np.maximum(eligible.sum(1,keepdims=True),1);result=(weights[:,:,None]*bank).sum(1);result[~eligible.any(1)]=bundle["prior"]
    return result,{"mask":selected,"executed_sample_head_pairs":calls,"audio_heads":0,"clinical_alert":False}


def verify_forecast() -> dict[str,Any]:
    """回放两个新来源的所有预测，使用同一原量表输入验证接口。"""
    result={}
    for name,spec in SOURCES.items():
        rows=json.loads((OUT/f"forecast/{name}/windows.json").read_text(encoding="utf-8"));stored=np.load(OUT/f"forecast/{name}/predictions.npz",allow_pickle=False)["selected"]
        actual=[]
        for row in rows:
            payload={"dataset":name,"source_schema":f'openesm:{name}:{spec["target"]}:{spec["low"]}_{spec["high"]}',"target_day":row["target_day"],
                "observations":[{"person":row["person"],"day":d,"value":spec["low"]+v*(spec["high"]-spec["low"])} for d,v in zip(row["history_days"],row["history"],strict=True)]}
            actual.append(predict_day(payload)["normalized_prediction"])
        error=float(np.max(np.abs(np.array(actual)-stored)));assert error<1e-12
        result[name]={"windows":len(rows),"max_error":error}
    save(OUT/"forecast/runtime_verification.json",result);return result


if __name__=="__main__":print(json.dumps(verify_forecast()))
