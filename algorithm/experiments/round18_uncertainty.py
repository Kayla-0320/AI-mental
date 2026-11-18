"""尾部误差与自适应误差带；开发选型和按用户隔离校准后验证新长跨度来源。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests
from experiments.round14_common import ROOT,save,freeze,sha256
from experiments.round14_longitudinal import daily_samples
from experiments.round15_forecast import forecast
from experiments.round16_external import stats

OUT=ROOT/"round18/forecast"
DATA=Path(r"D:\Develop\AIC\data\public\round18_sources")


def point(history: list[float],mode: str) -> float:
    """只调整预测平滑强度，不重写本人自评；只看过去变化。"""
    h=np.asarray(history,float);alpha=.4
    if mode=="noise_guard" and len(h)>=5 and np.abs(np.diff(h[-7:])).mean()>.12:alpha=.2
    elif mode=="trend_guard" and len(h)>=4:
        delta=np.diff(h[-4:]);alpha=.6 if np.all(delta>0) or np.all(delta<0) else .25
    elif mode not in {"ema4","noise_guard","trend_guard"}:raise ValueError("未知预测器")
    return forecast(history,{"kind":"ema","alpha":alpha})


def scale(history: list[float]) -> float:
    """过去相邻变化量作为不确定性尺度，设下限防止恒定历史产生零宽误差带。"""
    h=np.asarray(history,float)
    if len(h)<3 or not np.isfinite(h).all():raise ValueError("过去历史不足")
    return max(.05,float(np.abs(np.diff(h[-7:])).mean()))


def weighted_quantile(values: np.ndarray,weights: np.ndarray,level: float) -> float:
    """按来源及用户等权计算经验分位数，不声称时序数据的分布无关保证。"""
    if not 0<level<1 or len(values)==0 or not np.isfinite(values).all() or np.any(weights<=0):raise ValueError("分位数输入无效")
    order=np.argsort(values);cum=np.cumsum(weights[order]);return float(values[order][min(np.searchsorted(cum,level*cum[-1]),len(values)-1)])


def lock() -> dict[str,Any]:
    """五个旧来源按用户70/30隔离开发与校准，外部Beck未参与任何选择。"""
    path=OUT/"calibration_lock.json"
    if path.exists():return json.loads(path.read_text(encoding="utf-8"))
    paths={"studentlife":ROOT/"round14/longitudinal/studentlife_calendar_days/daily_observations.csv",**{n:ROOT/f"round16/forecast/{n}/daily.csv" for n in ("sicorello","dejonckheere")},**{n:ROOT/f"round17/forecast/{n}/daily.csv" for n in ("westhoff","wang")}}
    datasets={n:daily_samples(pd.read_csv(p,dtype={"person":str},float_precision="round_trip")) for n,p in paths.items()};selection={};cal=[];partitions={}
    for name,rows in datasets.items():
        ids=sorted({r["person"] for r in rows});order=np.random.default_rng(42).permutation(ids);cut=int(.7*len(ids));a,b=set(order[:cut]),set(order[cut:])
        selection[name]=[r for r in rows if r["person"] in a];cal.extend(dict(r,dataset=name) for r in rows if r["person"] in b)
        partitions[name]={"selection_people":sorted(a),"calibration_people":sorted(b)}
    scores={}
    for mode in ("ema4","noise_guard","trend_guard"):
        scores[mode]={name:stats(rows,np.array([point(r["history"],mode) for r in rows])) for name,rows in selection.items()}
    base=scores["ema4"];objectives={}
    for mode in scores:
        ratios=np.array([[scores[mode][n][m]/base[n][m] for m in ("person_mae_0_100","rmse_0_100","worst_quarter_person_mae")] for n in selection])
        objectives[mode]=float(ratios.mean()) if np.all(ratios[:,:2].mean(0)<=1.005) and ratios[:,:2].max()<=1.02 else 100.
    chosen=min(objectives,key=objectives.get)
    errors=np.array([abs(point(r["history"],chosen)-r["y"]) for r in cal]);scales=np.array([scale(r["history"]) for r in cal])
    weights=np.array([1/sum(r2["person"]==r["person"] and r2["dataset"]==r["dataset"] for r2 in cal)/len(partitions[r["dataset"]]["calibration_people"]) for r in cal])
    result={"point_method":chosen,"scores":scores,"objectives":objectives,"partitions":partitions,"calibration_rows":len(cal),"constant_radius":weighted_quantile(errors,weights,.9),
        "adaptive_multiplier":weighted_quantile(errors/scales,weights,.9),"nominal_coverage":.9,"source_sha256":sha256(Path(__file__)),
        "scope":"empirical calibration with disjoint people, NOT guaranteed individual/conformal coverage under time/source shift","external_target":"Beck repeated sad item1–5, day timestamps; not clinical state"}
    freeze(path,result);return result


def evaluate_band(truth: np.ndarray,prediction: np.ndarray,radius: np.ndarray) -> dict[str,float]:
    """同时报告覆盖和宽度、区间分数，不以扩大到全量表伪造可靠性。"""
    lo=np.maximum(0,prediction-radius);hi=np.minimum(1,prediction+radius)
    interval=(hi-lo)+20*np.maximum(lo-truth,0)+20*np.maximum(truth-hi,0)
    return {"coverage":float(np.mean((truth>=lo)&(truth<=hi))),"mean_width_0_100":float(np.mean(hi-lo)*100),"mean_interval_score_0_100":float(interval.mean()*100)}


def run() -> None:
    """固定模型和两种区间后首次读取新来源结局，保留失败覆盖和分组误差。"""
    OUT.mkdir(parents=True,exist_ok=True);calibration=lock()
    freeze(OUT/"protocol.json",{"seed":42,"target":"sad","theoretical_scale":[1,5],"beep_policy":"beep absent; key=id/date/hour/minute; conflicting keys quarantined",
        "minimum_daily_responses":2,"new_source_training":False,"source_sha256":sha256(Path(__file__)),"calibration_sha256":sha256(OUT/"calibration_lock.json")})
    DATA.mkdir(parents=True,exist_ok=True);metadata=OUT/"beck_metadata.json"
    if metadata.exists():meta=json.loads(metadata.read_text(encoding="utf-8"))
    else:
        response=requests.get('https://zenodo.org/api/records/22764751',timeout=30);response.raise_for_status();meta=response.json();save(metadata,meta)
    item=next(f for f in meta["files"] if f["key"]=='0060_beck_ts.tsv');path=DATA/item["key"]
    if not path.exists():
        response=requests.get(item["links"]["self"],timeout=(20,150));response.raise_for_status();path.write_bytes(response.content)
    assert hashlib.md5(path.read_bytes()).hexdigest()==item["checksum"].split(':')[-1]
    f=pd.read_csv(path,sep='\t',na_values=['NA'],dtype={'id':str});initial=len(f)
    if not f.sad.dropna().between(1,5).all():raise ValueError("量表边界未匹配")
    f=f[['id','date','hour','minute','sad']].dropna();key=['id','date','hour','minute'];conflict=f.groupby(key).sad.transform('nunique')>1
    quarantined=int(conflict.sum());f=f.loc[~conflict].drop_duplicates(key).copy()
    dates=pd.to_datetime(f.date,errors='raise');f['day']=(dates-dates.groupby(f.id).transform('min')).dt.days;f['value']=(f.sad-1)/4
    daily=f.groupby(['id','day']).value.agg(['mean','count']).reset_index();daily=daily[daily['count']>=2].rename(columns={'id':'person','mean':'value'})[['person','day','value']]
    rows=daily_samples(daily);daily.to_csv(OUT/'daily.csv',index=False);save(OUT/'windows.json',rows)
    y=np.array([r['y'] for r in rows]);p=np.array([point(r['history'],calibration['point_method']) for r in rows]);s=np.array([scale(r['history']) for r in rows])
    baseline=np.array([forecast(r['history'],{'kind':'ema','alpha':.4}) for r in rows])
    result={'record_id':meta['id'],'source_sha256':sha256(path),'raw_rows':initial,'conflicting_rows':quarantined,'daily_rows':len(daily),'point_method':calibration['point_method'],
        'point':stats(rows,p),'previous_point':stats(rows,baseline),'constant_band':evaluate_band(y,p,np.full(len(y),calibration['constant_radius'])),
        'adaptive_band':evaluate_band(y,p,calibration['adaptive_multiplier']*s),'future_outcome_used_for_scale':False,'clinical_guarantee':False}
    np.savez_compressed(OUT/'predictions.npz',truth=y,people=np.array([r['person'] for r in rows]),prediction=p,baseline=baseline,scale=s)
    save(OUT/'results.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
