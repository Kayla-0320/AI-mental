"""跨来源同时约束MAE/RMSE的个人预测；旧外部数据转开发，新来源只验证。"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round14_longitudinal import daily_samples
from experiments.round15_forecast import forecast
from experiments.round16_external import robust_forecast,stats,prepare,intervals

OUT=ROOT/"round17"
DATA=Path(r"D:\Develop\AIC\data\public\round17_sources")
SOURCES={"westhoff":{"concept":17347510,"filename":"0008_westhoff_ts.tsv","target":"affect_valence","low":0,"high":100,"day":"day","min_responses":2,"page":"https://openesmdata.org/datasets/0008_westhoff/"},
    "wang":{"concept":17348326,"filename":"0055_wang_ts.tsv","target":"sad","low":1,"high":100,"day":"day","min_responses":2,"page":"https://openesmdata.org/datasets/0055_wang/"}}


@lru_cache(maxsize=12000)
def old_predictions(history: tuple[float,...]) -> tuple[float,float]:
    """缓存两个旧候选，仅为计算复用，不混入用户身份或未来标签。"""
    return forecast(list(history),{"kind":"ema","alpha":.3}),robust_forecast(list(history),{"kind":"robust","eta":10.,"mix":1.})


def balanced_forecast(history: list[float],config: dict[str,Any]) -> float:
    """估计向个人中心回归的强度，并向旧EMA保守收缩，不使用群体结局先验。"""
    h=np.asarray(history,float);base,old=old_predictions(tuple(history));kind=config["kind"]
    if kind=="baseline":return base
    if kind=="old_robust":return old
    if kind=="ema":return forecast(history,config)
    if kind=="blend":return float((1-config["weight"])*base+config["weight"]*old)
    if kind!="ar":raise ValueError("未知预测方法")
    if len(h)<5:return base
    # 所有中心、协方差和先验强度仅来自当前可见的历史。
    center=float(h[-7:].mean());a=h[:-1]-center;b=h[1:]-center
    variance=max(float(np.mean(a*a)),1e-4)
    strength=config["strength"]*variance
    phi=float(np.clip((a@b+strength*config["prior_phi"])/(a@a+strength),0.,.8))
    personal=center+phi*(h[-1]-center)
    return float(np.clip((1-config["weight"])*base+config["weight"]*personal,0,1))


def candidates() -> list[dict[str,Any]]:
    """固定20个小候选，禁止根据新来源结果扩展网格。"""
    values=[{"id":"baseline","kind":"baseline"}]
    values += [{"id":f"blend_{w}","kind":"blend","weight":w} for w in (.25,.5,.75)]
    values += [{"id":f"ema_{a}","kind":"ema","alpha":a} for a in (.15,.2,.4,.5)]
    values += [{"id":f"ar_{p}_{s}_{w}","kind":"ar","prior_phi":p,"strength":s,"weight":w} for p in (0.,.3) for s in (3.,10.) for w in (.25,.5,1.)]
    return values


def lock() -> dict[str,Any]:
    """仅在已曝光的三个开发来源锁型，同时约束误差和最差来源退步幅度。"""
    path=OUT/"forecast/selection_lock.json"
    if path.exists():return json.loads(path.read_text(encoding="utf-8"))
    frames={"studentlife":ROOT/"round14/longitudinal/studentlife_calendar_days/daily_observations.csv",
        "sicorello":ROOT/"round16/forecast/sicorello/daily.csv","dejonckheere":ROOT/"round16/forecast/dejonckheere/daily.csv"}
    datasets={k:daily_samples(pd.read_csv(p,dtype={"person":str},float_precision="round_trip")) for k,p in frames.items()}
    bank=candidates();scores={}
    for config in bank:
        scores[config["id"]]={name:stats(rows,np.array([balanced_forecast(r["history"],config) for r in rows])) for name,rows in datasets.items()}
    base=scores["baseline"];eligible=[];objectives={}
    for config in bank:
        ratios=np.array([[scores[config["id"]][name][key]/base[name][key] for key in ("person_mae_0_100","rmse_0_100")] for name in datasets])
        objectives[config["id"]]=float(ratios.mean())
        if np.all(ratios.mean(0)<=1+1e-12) and np.max(ratios)<=1.02+1e-12:eligible.append(config)
    winner=min(eligible,key=lambda c:objectives[c["id"]])
    value={"winner":winner,"candidates":bank,"scores":scores,"objectives":objectives,"eligible":[c["id"] for c in eligible],
        "rule":"both source-mean relative person-MAE and relative RMSE <=baseline; no source/metric >2% degradation; minimize mean relative loss",
        "development_status":"StudentLife and both round16 sources previously exposed, now development; not independent tests",
        "development_sha256":{k:sha256(p) for k,p in frames.items()},"new_sources":SOURCES,"source_sha256":sha256(Path(__file__))}
    freeze(path,value);print(json.dumps({"winner":winner,"eligible":value["eligible"]}),flush=True);return value


def acquire(name: str,spec: dict[str,Any]) -> tuple[Path,dict]:
    """锁型后下载固定版本文件，保存发布MD5与来源页，拒绝静默更新。"""
    DATA.mkdir(parents=True,exist_ok=True)
    metadata_path=OUT/"data"/(name+"_metadata.json")
    if metadata_path.exists():meta=json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        response=requests.get(f'https://zenodo.org/api/records/{spec["concept"]}',timeout=40);response.raise_for_status();meta=response.json();save(metadata_path,meta)
    item=next(f for f in meta["files"] if f["key"]==spec["filename"])
    path=DATA/spec["filename"]
    if not path.exists():
        response=requests.get(item["links"]["self"],timeout=(20,150));response.raise_for_status();path.write_bytes(response.content)
    if hashlib.md5(path.read_bytes()).hexdigest()!=item["checksum"].split(":")[-1]:raise ValueError("新来源发布摘要不一致")
    page=OUT/"data"/(name+"_source_page.html")
    if not page.exists():
        response=requests.get(spec["page"],timeout=30);response.raise_for_status();page.write_bytes(response.content)
    return path,meta


def run() -> None:
    """两个新来源一并报告，预测器在读取新目标前锁定，不作按测试来源择优。"""
    OUT.mkdir(exist_ok=True);policy=lock()
    freeze(OUT/"forecast/protocol.json",{"sources":SOURCES,"selected":policy["winner"],"selection_lock_sha256":sha256(OUT/"forecast/selection_lock.json"),
        "methods":["selected","previous_ema","previous_robust","last","mean7"],"new_source_training":False,"seed":42})
    for name,spec in SOURCES.items():
        folder=OUT/"forecast"/name
        if (folder/"results.json").exists():continue
        path,meta=acquire(name,spec);frame,audit=prepare(path,spec);rows=daily_samples(frame)
        if not rows:raise ValueError("新来源没有合格窗口，不能换目标刷分")
        freeze(folder/"audit.json",audit);frame.to_csv(folder/"daily.csv",index=False)
        configs={"selected":policy["winner"],"previous_ema":{"kind":"baseline"},"previous_robust":{"kind":"old_robust"}}
        p={k:np.array([balanced_forecast(r["history"],c) for r in rows]) for k,c in configs.items()}
        p["last"]=np.array([r["history"][-1] for r in rows]);p["mean7"]=np.array([np.mean(r["history"][-7:]) for r in rows])
        np.savez_compressed(folder/"predictions.npz",truth=np.array([r["y"] for r in rows]),people=np.array([r["person"] for r in rows]),**p)
        save(folder/"windows.json",rows)
        result={"audit":audit,"record_id":meta["id"],"license":meta["metadata"].get("license"),"metrics":{k:stats(rows,v) for k,v in p.items()},
            "person_mae_ci":{k:intervals(rows,p["selected"],p[k]) for k in p if k!="selected"},"new_source_used_for_selection":False}
        save(folder/"results.json",result);print(json.dumps({"dataset":name,**result},ensure_ascii=False),flush=True)


if __name__=="__main__":run()
