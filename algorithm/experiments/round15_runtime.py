"""新用户离线预测的证据量回退；仅改变研究预测器，不改变任何风险阈值。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from experiments.round15_forecast import forecast,choose,configs,OUT
from experiments.round14_common import ROOT,save,freeze,sha256
from experiments.round14_longitudinal import daily_samples,select_partitions
import pandas as pd


def reliable_choice(validation: list[dict]) -> tuple[dict,dict]:
    """小验证集不足以支持个性化超参数时回退保持法，阈值为研究启发式。"""
    count=len({r["person"] for r in validation})
    if count<10 or len(validation)<100:
        return {"id":"fallback_last","kind":"last"},{"fallback":True,"validation_people":count,"validation_windows":len(validation),"reason":"insufficient_selection_evidence"}
    selected,scores=choose(validation,configs())
    return selected,{"fallback":False,"validation_people":count,"validation_windows":len(validation),"scores":scores}


def predict_new_user(history: list[float], artifact: dict[str,Any], source_schema: str) -> dict[str,Any]:
    """要求调用者先按日和缺测契约整理历史，量表不一致时拒绝迁移。"""
    if source_schema!="studentlife_pam_normalized":raise ValueError("仅验证StudentLife归一化PAM，不能自动应用项目心情或压力分")
    return {"prediction_normalized":forecast(history,artifact["config"]),"method":artifact["config"],"selection":artifact["selection"],
        "clinical_alert":False,"research_only":True,"history_contract":"already validated daily history, max14calendar days, no gap>3days, at least3 observed days"}


def run() -> None:
    """补充保守回退版本；先前候选负结果和原始协议不覆盖。"""
    from experiments.round15_forecast import evaluate
    frame=pd.read_csv(ROOT/"round14/longitudinal/studentlife_calendar_days/daily_observations.csv",dtype={"person":str},float_precision="round_trip")
    parts,settings=select_partitions(daily_samples(frame),"unseen_person_future")
    freeze(OUT/"forecast/reliability_protocol.json",{"min_validation_people":10,"min_validation_windows":100,"status":"exploratory heuristic added after initial candidate screen; not preregistered confirmation",
        "code_sha256":sha256(Path(__file__)),"clinical_rules_changed":False})
    config,selection=reliable_choice(parts["validation"])
    save(OUT/"forecast/runtime_artifact.json",{"schema":"pam_new_user_research.v1","config":config,"selection":selection,"source_schema":"studentlife_pam_normalized","temporal_split":settings})
    # 单独保存，不覆盖首轮strict负结果。
    from experiments.round15_forecast import predict
    import numpy as np
    original=np.load(OUT/"forecast/strict_predictions.npz",allow_pickle=False)
    truth=original["truth"];pred=predict(parts["test"],config)
    from experiments.round14_longitudinal import metrics,cluster_ci
    save(OUT/"forecast/reliability_results.json",{"selected":metrics(parts["test"],pred,3.,1/3),"config":config,"selection":selection,
        "ci95_vs_old_ema":cluster_ci(parts["test"],pred,original["old_ema_0.1"],3.),"equals_last":bool(np.array_equal(pred,original["last"])),"last_baseline_beaten":False})
    np.savez_compressed(OUT/"forecast/reliability_predictions.npz",truth=truth,selected=pred)


if __name__=="__main__":run()
