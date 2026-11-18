"""旧来源锁型后对两个首次使用的公开ESM来源一次评测，禁止外部结果驱动重选。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round14_longitudinal import daily_samples,select_partitions
from experiments.round15_forecast import forecast,configs,predict

OUT=ROOT/"round16"
DATA=Path(r"D:\Develop\AIC\data\public\round16_sources")
SOURCES={
    "sicorello":{"record":22809936,"filename":"0073_sicorello_ts.tsv","md5":"a4b03b3c47b824dcc46c009cd7ffeca6","target":"overwhelming_emotions","low":1,"high":5,"day":"date","min_responses":2,
        "page":"https://openesmdata.org/datasets/0073_sicorello/","meaning":"feeling overwhelmed by emotions, not PAM valence or clinical outcome","license":"CC BY-NC4.0"},
    "dejonckheere":{"record":17347570,"filename":"0012_dejonckheere_ts.tsv","md5":"34e72297d13b90ae9b2f3c4339cdd1b5","target":"sad","low":0,"high":100,"day":"day","min_responses":2,
        "page":"https://openesmdata.org/datasets/0012_dejonckheere/","meaning":"sadness, next study day; calendar timestamps not released","license":"CC BY4.0"}}


def robust_forecast(history: list[float],config: dict[str,Any]) -> float:
    """只用过去的预序列误差给个人专家加权，限制少量观测下的过度适应。"""
    if config["kind"]!="robust":return forecast(history,config)
    h=np.asarray(history,float)
    forecast(history,{"kind":"last"})
    bank=[{"kind":"last"},{"kind":"mean","window":3},{"kind":"median","window":3},{"kind":"ema","alpha":.3}]
    if len(h)<6:return forecast(history,{"kind":"ema","alpha":.3})
    losses=np.array([np.mean([abs(forecast(h[:i].tolist(),c)-h[i]) for i in range(max(3,len(h)-7),len(h))]) for c in bank])
    weights=np.exp(-config["eta"]*(losses-losses.min()));weights/=weights.sum()
    combined=float(weights@np.array([forecast(history,c) for c in bank]))
    # 与稳健个人中心保守混合，不使用群体目标均值或未来波动。
    return float(config["mix"]*combined+(1-config["mix"])*np.median(h[-7:]))


def stats(rows: list[dict],prediction: np.ndarray) -> dict[str,Any]:
    """同时报告全覆盖总体误差、用户等权误差和高误差尾部。"""
    y=np.array([r["y"] for r in rows]);g=np.array([r["person"] for r in rows]);e=np.abs(prediction-y)
    pe=np.array([e[g==p].mean() for p in np.unique(g)])
    tail=np.sort(pe)[-max(1,int(np.ceil(len(pe)*.25))):]
    return {"windows":len(rows),"people":len(pe),"mae_0_100":float(e.mean()*100),"person_mae_0_100":float(pe.mean()*100),
        "rmse_0_100":float(np.sqrt(np.mean((prediction-y)**2))*100),"q90_abs_error":float(np.quantile(e,.9)*100),
        "worst_quarter_person_mae":float(tail.mean()*100),"coverage":1.0}


def lock_policy() -> dict[str,Any]:
    """先在旧StudentLife验证期选型并锁定，外部数据的目标不进入此函数。"""
    path=OUT/"forecast/source_selection_lock.json"
    if path.exists():return json.loads(path.read_text(encoding="utf-8"))
    frame=pd.read_csv(ROOT/"round14/longitudinal/studentlife_calendar_days/daily_observations.csv",dtype={"person":str},float_precision="round_trip")
    samples=daily_samples(frame);parts,_=select_partitions(samples,"same_person_future");valid=parts["validation"]
    bank=configs()+[{"id":f"robust_{eta}_{mix}","kind":"robust","eta":eta,"mix":mix} for eta in (1.,4.,10.) for mix in (.5,1.)]
    scores={c["id"]:stats(valid,np.array([robust_forecast(r["history"],c) for r in valid])) for c in bank}
    # 源验证固定加权目标，显式同时关心普通用户与高误差用户。
    winner=min(bank,key=lambda c:.7*scores[c["id"]]["person_mae_0_100"]+.3*scores[c["id"]]["worst_quarter_person_mae"])
    robust=min([c for c in bank if c["kind"]=="robust"],key=lambda c:.7*scores[c["id"]]["person_mae_0_100"]+.3*scores[c["id"]]["worst_quarter_person_mae"])
    result={"winner":winner,"robust_candidate":robust,"scores":scores,"selection_data":"old StudentLife validation only, previously exposed",
        "target_usage":"no external labels used to select either config","external_source_specs":SOURCES,"code_sha256":sha256(Path(__file__))}
    freeze(path,result);return result


def download(name: str,spec: dict[str,Any]) -> Path:
    """按发布MD5核验固定版本，同时保存发布元数据、许可和来源页。"""
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/spec["filename"]
    if not path.exists():
        url=f'https://zenodo.org/api/records/{spec["record"]}/files/{spec["filename"]}/content'
        response=requests.get(url,timeout=(20,120));response.raise_for_status();path.write_bytes(response.content)
    if hashlib.md5(path.read_bytes()).hexdigest()!=spec["md5"]:raise ValueError("外部来源发布MD5不一致")
    metadata=requests.get(f'https://zenodo.org/api/records/{spec["record"]}',timeout=30);metadata.raise_for_status()
    save(OUT/"data"/(name+"_metadata.json"),metadata.json())
    response=requests.get(spec["page"],timeout=30);response.raise_for_status()
    (OUT/"data"/(name+"_source_page.html")).write_bytes(response.content)
    return path


def prepare(path: Path,spec: dict[str,Any]) -> tuple[pd.DataFrame,dict]:
    """按已核实的量表界限归一化，去除冲突时点，缺失不补零。"""
    f=pd.read_csv(path,sep="\t",na_values=["NA"],dtype={"id":str});initial=len(f)
    target=pd.to_numeric(f[spec["target"]],errors="raise")
    if not target.dropna().between(spec["low"],spec["high"]).all():raise ValueError("外部量表范围变化")
    f=f[["id",spec["day"],"beep",spec["target"]]].dropna().copy()
    if spec["day"]=="date":
        dates=pd.to_datetime(f["date"],errors="raise").dt.normalize()
        f["day_index"]=(dates-dates.groupby(f["id"]).transform("min")).dt.days
    else:
        d=pd.to_numeric(f["day"],errors="raise")
        if not np.all(d==np.floor(d)):raise ValueError("研究日必须是整数")
        f["day_index"]=d.astype(int)
    key=["id","day_index","beep"]
    conflict=f.groupby(key,observed=True)[spec["target"]].transform("nunique")>1
    quarantined=int(conflict.sum());f=f.loc[~conflict].drop_duplicates(key)
    f["value"]=(f[spec["target"]]-spec["low"])/(spec["high"]-spec["low"])
    daily=f.groupby(["id","day_index"]).value.agg(["mean","count"]).reset_index()
    excluded=int((daily["count"]<spec["min_responses"]).sum());daily=daily[daily["count"]>=spec["min_responses"]]
    daily=daily.rename(columns={"id":"person","day_index":"day","mean":"value"})[["person","day","value"]]
    return daily,{"raw_rows":initial,"conflicting_rows_quarantined":quarantined,"low_response_days_excluded":excluded,"daily_rows":len(daily),"people":int(daily.person.nunique()),
        "source_sha256":sha256(path),"target":spec["target"],"day_semantics":"actual date" if spec["day"]=="date" else "study-day index; no calendar date inference"}


def intervals(rows: list[dict],first: np.ndarray,second: np.ndarray) -> dict[str,Any]:
    """用户聚类误差差区间，不把日记录当独立人数。"""
    g=np.array([r["person"] for r in rows]);y=np.array([r["y"] for r in rows]);people=np.unique(g)
    delta=np.array([(np.abs(first[g==p]-y[g==p])-np.abs(second[g==p]-y[g==p])).mean() for p in people])
    ix=np.random.default_rng(42).integers(0,len(people),(5000,len(people)))
    return {"person_mae_delta_0_100":float(delta.mean()*100),"ci95":(np.quantile(delta[ix].mean(1),[.025,.975])*100).tolist()}


def run() -> None:
    """整个新来源只作外部评测；不根据它们的结果重选参数或筛掉较差来源。"""
    OUT.mkdir(exist_ok=True);lock=lock_policy()
    freeze(OUT/"forecast/external_protocol.json",{"sources":SOURCES,"methods":{"source_selected":lock["winner"],"robust":lock["robust_candidate"],"previous_ema":{"kind":"ema","alpha":.3},"last":{"kind":"last"},"mean7":{"kind":"mean","window":7}},
        "all_users_external":True,"minimum_history":3,"lookback_days":14,"missing_gap_reset":3,"minimum_daily_responses":2,"fresh_source_training":False,"selection_lock_sha256":sha256(OUT/"forecast/source_selection_lock.json")})
    methods=json.loads((OUT/"forecast/external_protocol.json").read_text(encoding="utf-8"))["methods"]
    for name,spec in SOURCES.items():
        folder=OUT/"forecast"/name
        if (folder/"results.json").exists():continue
        path=download(name,spec);daily,audit=prepare(path,spec);rows=daily_samples(daily)
        if not rows:raise ValueError("外部数据无合格跨日窗口")
        freeze(folder/"field_audit.json",audit);daily.to_csv(folder/"daily.csv",index=False)
        predictions={k:np.array([robust_forecast(r["history"],c) for r in rows]) for k,c in methods.items()}
        np.savez_compressed(folder/"predictions.npz",truth=np.array([r["y"] for r in rows]),people=np.array([r["person"] for r in rows]),**predictions)
        save(folder/"windows.json",rows)
        result={"source":spec,"audit":audit,"metrics":{k:stats(rows,p) for k,p in predictions.items()},
            "paired_person_ci":{k:{base:intervals(rows,predictions[k],predictions[base]) for base in ("last","mean7","previous_ema")} for k in ("source_selected","robust")},
            "scope":"first project use external-source validation of normalized forecasting recipe, not clinical/PAM/Chinese-user transfer"}
        save(folder/"results.json",result);print(json.dumps({"dataset":name,"metrics":result["metrics"],"ci":result["paired_person_ci"]},ensure_ascii=False),flush=True)


if __name__=="__main__":run()
