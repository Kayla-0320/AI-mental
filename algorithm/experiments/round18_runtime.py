"""第十八轮离线研究接口，不接入平台或改动生产安全规则。"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np
from experiments.round14_common import ROOT,sha256
from experiments.round17_correction import features
from experiments.round18_neutral import apply_policy
from experiments.round18_uncertainty import point,scale

OUT=ROOT/"round18"


def predict_quality_features(values: np.ndarray,folder: Path) -> np.ndarray:
    """用指定冻结CLIP与十项像素统计预测MOS，不将其当语义可信度或人工评分。"""
    if values.ndim!=2 or values.shape[1]!=522 or not np.isfinite(values).all():raise ValueError("画质特征须为有限522维，不能用情绪概率代替")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if sha256(folder/"model.pkl")!=manifest["model_sha256"]:raise ValueError("画质模型摘要不符")
    with (folder/"model.pkl").open("rb") as stream:model=pickle.load(stream)
    return np.clip(model.predict(values),1,5)


def predict_neutral(bank: np.ndarray,mask: np.ndarray,quality: np.ndarray,folder: Path) -> np.ndarray:
    """对合格图文概率执行冻结策略；缺模态回退由上游均值专家处理。"""
    n=len(bank)
    if bank.shape!=(n,3,3) or mask.shape!=(n,3) or mask.dtype!=bool or quality.shape!=(n,):raise ValueError("图文概率形状无效")
    if mask[:,1].any() or not mask[:,[0,2]].all():raise ValueError("此元分类器仅验证完整图文，音频不得启用")
    if not np.isfinite(bank).all() or np.any(bank<0) or not np.allclose(bank.sum(2),1):raise ValueError("专家概率无效")
    if np.isinf(quality).any() or np.any((quality[np.isfinite(quality)]<0)|(quality[np.isfinite(quality)]>1)):raise ValueError("质量输入无效")
    policy=json.loads((folder/"policy.json").read_text(encoding="utf-8"))
    if sha256(folder/"model.pkl")!=policy["sha256"]:raise ValueError("模型摘要不匹配")
    with (folder/"model.pkl").open("rb") as stream:model=pickle.load(stream)
    return apply_policy(bank.mean(1),model.predict_proba(features(bank,mask,quality)),policy["config"])


def predict_day(payload: dict[str,Any]) -> dict[str,Any]:
    """在Beck重复sad项目原量表上输出点预测和经验误差带，拒绝未来数据。"""
    if payload.get("source_schema")!="openesm:beck:sad:1_5":raise ValueError("未验证的来源或量表")
    rows=payload.get("observations",[]);target=payload.get("target_day")
    if type(target) is not int or not rows:raise ValueError("目标日或历史缺失")
    if len({str(r.get("person","")) for r in rows})!=1 or not str(rows[0].get("person","")):raise ValueError("必须同一用户")
    days=[];values=[]
    for r in rows:
        d,v=r.get("day"),r.get("value")
        if type(d) is not int or d>=target:raise ValueError("混入未来或非法日序")
        if type(v) not in (float,int) or not np.isfinite(v) or not 1<=v<=5:raise ValueError("原量表值无效")
        days.append(d);values.append((v-1)/4)
    if days!=sorted(set(days)):raise ValueError("日序必须唯一递增")
    if days[-1]!=target-1:return {"status":"abstain","reason":"latest_day_missing"}
    indices=[i for i,d in enumerate(days) if days[-1]-14<d];days=[days[i] for i in indices];values=[values[i] for i in indices]
    gaps=[i for i in range(1,len(days)) if days[i]-days[i-1]>3]
    if gaps:values=values[gaps[-1]:]
    if len(values)<3:return {"status":"abstain","reason":"insufficient_history"}
    lock=json.loads((OUT/"forecast/calibration_lock.json").read_text(encoding="utf-8"));p=point(values,lock["point_method"]);radius=lock["constant_radius"]
    return {"status":"research_prediction","prediction":1+4*p,"normalized_prediction":p,"interval":[1+4*max(0,p-radius),1+4*min(1,p+radius)],"past_change_scale":scale(values),"interval_kind":"empirical_constant_reference","nominal_coverage":.9,"individual_coverage_guaranteed":False,"clinical_alert":False,"automatic_intervention":False}
