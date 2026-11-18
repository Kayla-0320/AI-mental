"""外部验证配套的按日离线推理与回放，拒绝未来信息和未经验证的量表。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
from experiments.round16_external import OUT,SOURCES,robust_forecast
from experiments.round14_common import save


def predict_observations(payload: dict[str,Any]) -> dict[str,Any]:
    """检查日索引、量表、单人身份和缺测后，调用同一冻结推理函数。"""
    name=payload.get("dataset")
    if name not in SOURCES:raise ValueError("未验证的数据来源")
    spec=SOURCES[name]
    expected=f'openesm:{name}:{spec["target"]}:{spec["low"]}_{spec["high"]}'
    if payload.get("source_schema")!=expected:raise ValueError("量表声明不匹配；不能自动套用项目心情或压力")
    target=payload.get("target_day");rows=payload.get("observations",[])
    if type(target) is not int or not rows:raise ValueError("缺少有效目标日和过去观测")
    if len({str(r.get("person","")) for r in rows})!=1 or not str(rows[0].get("person","")):raise ValueError("必须是同一匿名用户")
    days=[];values=[]
    for row in rows:
        day,value=row.get("day"),row.get("value")
        if type(day) is not int or day>=target:raise ValueError("日索引无效或包含未来/目标日")
        if type(value) not in {float,int} or not np.isfinite(value) or not spec["low"]<=value<=spec["high"]:raise ValueError("数值超出来源量表")
        days.append(day);values.append((value-spec["low"])/(spec["high"]-spec["low"]))
    if days!=sorted(set(days)):raise ValueError("日索引必须递增且不重复")
    if days[-1]!=target-1:return {"status":"abstain","reason":"latest_day_missing"}
    indices=[i for i,d in enumerate(days) if days[-1]-14<d]
    days=[days[i] for i in indices];values=[values[i] for i in indices]
    gaps=[i for i in range(1,len(days)) if days[i]-days[i-1]>3]
    if gaps:days,values=days[gaps[-1]:],values[gaps[-1]:]
    if len(values)<3:return {"status":"abstain","reason":"insufficient_history_after_gap_reset"}
    methods=json.loads((OUT/"forecast/external_protocol.json").read_text(encoding="utf-8"))["methods"]
    method=payload.get("method","source_selected")
    if method not in {"source_selected","robust"}:raise ValueError("只提供预先锁定的研究候选")
    value=robust_forecast(values,methods[method])
    return {"status":"research_prediction","method":method,"normalized_prediction":value,"original_scale_prediction":spec["low"]+value*(spec["high"]-spec["low"]),
        "origin_day":days[-1],"target_day":target,"history_days":len(values),"clinical_alert":False,"automatic_intervention":False,
        "warning":"validated only as normalized daily forecasting recipe on named public sources; no clinical or project-user transfer"}


def verify() -> dict[str,Any]:
    """对所有外部窗口和两个候选逐项回放，不改选型或覆盖原始实验结果。"""
    result={}
    for name,spec in SOURCES.items():
        rows=json.loads((OUT/f"forecast/{name}/windows.json").read_text(encoding="utf-8"));stored=np.load(OUT/f"forecast/{name}/predictions.npz",allow_pickle=False)
        for method in ("source_selected","robust"):
            actual=[]
            for row in rows:
                payload={"dataset":name,"source_schema":f'openesm:{name}:{spec["target"]}:{spec["low"]}_{spec["high"]}',"method":method,"target_day":row["target_day"],
                    "observations":[{"person":row["person"],"day":day,"value":spec["low"]+v*(spec["high"]-spec["low"])} for day,v in zip(row["history_days"],row["history"],strict=True)]}
                actual.append(predict_observations(payload)["normalized_prediction"])
            delta=float(np.max(np.abs(np.array(actual)-stored[method])))
            if delta>1e-12:raise ValueError("运行时与外部评测预测不一致")
            result[name+":"+method]={"windows":len(rows),"maximum_error":delta}
    save(OUT/"runtime_verification.json",result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--history",type=Path);args=parser.parse_args()
    print(json.dumps(predict_observations(json.loads(args.history.read_text(encoding="utf-8"))) if args.history else verify(),ensure_ascii=False))
