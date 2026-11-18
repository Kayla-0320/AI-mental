"""最终分支真实执行、目标无关性、新测试重放和来源折从头重训。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import time
from typing import Any
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np
from experiments.round10_fusion import folds,take
from experiments.round11_features import load_data
from experiments.round11_common import OUTPUT as ROUND11,digest
from experiments.round13_data import OUTPUT,save
from experiments.round13_models import Projection,SentimentLexicon,fit_heads
from experiments.round13_train import ch_data,read
from experiments.round13_evaluate import predict
from experiments.round13_neural import train_head
from experiments.round13_runtime import CHPredictor


def raw_condition(data: dict[str,np.ndarray],condition: str) -> dict[str,np.ndarray]:
    """把所有视觉代理同步移除，避免仅改全局mask后人脸仍泄露。"""
    result = {k:v.copy() for k,v in data.items() if k!="labels"}
    absent = {"missing_A":(1,),"missing_T":(0,),"missing_V":(2,),"text_only":(1,2),"all_missing":(0,1,2)}.get(condition,())
    for j in absent:
        result["modality_mask"][:,j] = False
        result["quality"][:,j] = 0.
        if j==2:
            for name in ("face_summary","deep_face","visual","coverage"):
                result[name][:] = 0.
    return result


def holm(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """对完整报告家族统一校正，保留其探索性而不冒称预注册主要终点。"""
    result = []
    running = 0.
    for i,row in enumerate(sorted(rows,key=lambda r:r["p"])):
        running = max(running,(len(rows)-i)*row["p"])
        result.append({**row,"holm_adjusted_p":min(1.,running)})
    return result


def main() -> None:
    """全部验证通过后写入一次性总验证记录。"""
    if (OUTPUT/"verification.json").exists():
        return
    raw = load_data("pts")
    _,ix = np.unique(raw["source_groups"],return_index=True)
    sample = take(raw,np.sort(ix[:16]))
    predictor = CHPredictor()
    restored = CHPredictor()
    scenarios = []
    for condition in ("clean","missing_A","missing_T","missing_V","text_only","all_missing"):
        inputs = raw_condition(sample,condition)
        p,details = predictor.predict(inputs)
        q,_ = restored.predict(inputs)
        error = float(np.max(np.abs(p-q)))
        # 人工改变隐藏的原始音频缓存，检查缺音频路径确实不消费它们。
        poison_error = None
        if condition in ("missing_A","text_only"):
            poisoned = {k:v.copy() for k,v in inputs.items()}
            for key in ("audio","speech","speech_emotion","sensevoice","sensevoice_query","sensevoice_emotion"):
                poisoned[key][:] = 1e6
            changed,_ = predictor.predict(poisoned)
            poison_error = float(np.max(np.abs(p-changed)))
        if error>1e-6 or (poison_error is not None and poison_error>1e-6):
            raise ValueError("退化分支重放或隐藏音频隔离不符")
        scenarios.append(dict(condition=condition,reload_error=error,hidden_audio_poison_error=poison_error,
            fallback_count=sum(details["fallback_used"]),n=len(p)))
    data = ch_data()
    tr,va = folds(data)[0]
    td,vd = take(data,tr),take(data,va)
    decision = json.loads((OUTPUT/"final_ch/decision_locked.json").read_text("utf-8"))
    choice = decision["choice"]
    config,family = choice["winner"]["config"],choice["winner"]["family"]
    projection = Projection().fit(td)
    knowledge = SentimentLexicon().encode(td["raw_text"])
    config_for_experts = choice["teacher"] if family=="neural" else config
    heads = fit_heads(projection.transform(td),td["labels"],config_for_experts,knowledge)
    output = OUTPUT/"independent_retrain/fold0"
    if family=="neural":
        oof = read(OUTPUT/"neural/ch/fold0/inner_oof.npz")
        repeated = train_head(config,td,vd,projection,heads,oof,output)
    else:
        from experiments.round13_train import evaluate
        repeated = evaluate(heads,projection,vd,SentimentLexicon().encode(vd["raw_text"]))
    previous = read(OUTPUT/("neural" if family=="neural" else "training")/"ch"/config["id"]/"fold0/predictions.npz")
    retrain_error = max(float(np.max(np.abs(repeated[k]-previous[k]))) for k in repeated)
    if retrain_error>1e-6:
        raise ValueError("新中文来源折从头重训不符")
    replays = []
    secondary = []
    for dataset in ("mosi","meld"):
        lock = json.loads((OUTPUT/"new_test"/dataset/"evaluation_lock.json").read_text("utf-8"))
        if any(digest(Path(row["path"]))!=row["sha256"] for row in lock["models"]):
            raise ValueError("新语料模型在评测锁后变化")
        inputs = read(OUTPUT/"datasets"/dataset/"test_inputs.npz")
        actual,_ = predict(dataset,lock["choice"],inputs)
        prior = read(OUTPUT/"new_test"/dataset/"predictions_no_labels.npz")
        error = max(float(np.max(np.abs(actual[k]-prior[k]))) for k in actual)
        if error>1e-6:
            raise ValueError("新语料锁定预测重放不符")
        replays.append(dict(dataset=dataset,n=len(inputs["sample_ids"]),max_abs_error=error,labels_opened=False))
        scored = json.loads((OUTPUT/"new_test"/dataset/"results.json").read_text("utf-8"))
        for row in scored["rows"]:
            secondary.append(dict(dataset=dataset,endpoint=row["condition"],delta=row["delta"],p=row["permutation_p_improvement"]))
        neutral = scored["clean_neutral_comparison"]
        secondary.append(dict(dataset=dataset,endpoint="clean_neutral_F1",delta=neutral["delta"],p=neutral["permutation_p_improvement"]))
    save(OUTPUT/"secondary_holm.json",dict(family="all7conditions + clean neutral F1, both new corpora;16 comparisons including duplicate/no-op conditions",
        designation="exploratory secondary analysis; not a retrospectively relabeled primary endpoint",rows=holm(secondary)))
    # 核对旧交付清单所记录的主模型，当前轮从未覆盖这些路径。
    manifest = json.loads((ROUND11/"artifact_manifest_final.json").read_text("utf-8"))
    artifacts = manifest["artifacts"]
    checks = []
    for record in artifacts:
        if isinstance(record,dict) and str(record.get("path","")).endswith("model.pkl") and "final_models" in record.get("path",""):
            path = Path(record["path"])
            if not path.is_absolute():
                path = ROUND11/path
            if path.exists():
                matches = digest(path)==record["sha256"]
                checks.append(dict(path=str(path),matches_original_manifest=matches))
                if not matches:
                    raise ValueError("旧主模型变化")
    save(OUTPUT/"verification.json",dict(runtime_scenarios=scenarios,independent_fold0_retrain_max_abs_error=retrain_error,
        independent_retrain_scope="new scaler/PCA + fusion experts + router; same frozen inner-OOF teacher evidence, no foundation pretraining",
        new_test_replays=replays,previous_main_model_checks=checks,historical821_or_external1929_rescored=False,
        new_targets_reopened_for_replay=False))


if __name__ == "__main__":
    main()
