"""相同来源与预算下审查中文语音收益，不读取旧历史集或外部test成绩。"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
from sklearn.linear_model import LogisticRegression
from experiments.round10_fusion import folds, take
from experiments.round10_evidence import supervision
from experiments.round10_domain import bias_select, weights_for
from experiments.round7_decision_oof import metric
from experiments.round11_common import OUTPUT, digest, save
from experiments.round11_features import AUDIO_PACKS, load_data, observed, transform


def main() -> None:
    """极性A和整体M的音频预测是两项明确分开的开发任务。"""
    output = OUTPUT / "audio_screen_deterministic"
    if (output/"complete.json").exists():
        raise ValueError("拒绝覆盖开发音频对照")
    configurations = [{"quality": q, "encoder": e, "C": c, "target": t}
        for q,e,c,t in itertools.product(("legacy","pts"), AUDIO_PACKS, (.03,.1,1.), ("A","M"))]
    save(output/"preregister.json", {"configs": configurations,
        "selection_data": "train+selection2876;4 whole-source folds;all transformations fold-fitted",
        "coverage": "all2876 predictions retained even if quality rejects audio; absent expert returns uniform",
        "class_power": .5, "bias_grid": "same fixed12-point source-balanced grid for all candidates",
        "old_history_and_external_test": "not evaluated", "sensevoice_frontend_dither": 0.,
        "code_sha256": digest(Path(__file__)), "features_code_sha256": digest(Path(__file__).with_name("round11_features.py"))})
    predictions, results, cache = [], [], {}
    data_by_quality = {q:load_data(q) for q in ("legacy","pts")}
    labels, _ = supervision(data_by_quality["legacy"])
    started = time.monotonic()
    for i, config in enumerate(configurations):
        data = data_by_quality[config["quality"]]
        y = labels[:,2] if config["target"] == "A" else labels[:,0]
        probability = np.full((len(y),3), np.nan)
        for f,(tr,va) in enumerate(folds(data)):
            key = config["quality"],config["encoder"],f
            train, valid = take(data,tr),take(data,va)
            if key not in cache:
                x, fitted = transform(train,AUDIO_PACKS[config["encoder"]])
                v,_ = transform(valid,AUDIO_PACKS[config["encoder"]],fitted)
                cache[key] = x,v
            x,v = cache[key]
            active = observed(train, "audio")
            weight = weights_for(y[tr][active],train["source_groups"][active],0.,.5)
            model = LogisticRegression(C=config["C"]*2000/active.sum(),max_iter=700,random_state=42)
            model.fit(x[active],y[tr][active],sample_weight=weight)
            p = model.predict_proba(v)
            p[~observed(valid,"audio")] = 1/3
            probability[va] = p
        if not np.isfinite(probability).all():
            raise ValueError("开发音频OOF覆盖不完整")
        target_data = {**data,"labels":y}
        row = {"config":config,"argmax":metric(y,probability.argmax(1)),
               **bias_select(target_data,np.log(np.maximum(probability,1e-12)))}
        results.append(row)
        predictions.append(probability)
        print(json.dumps({"candidate":i+1,"total":len(configurations),"config":config,
            "oof_macro_f1":row["metrics"]["macro_f1"],"seconds":round(time.monotonic()-started,1)}),flush=True)
        save(output/"progress.json",{"results":results},immutable=False)
    data = data_by_quality["legacy"]
    np.savez_compressed(output/"oof.npz",probabilities=np.stack(predictions),sample_ids=data["sample_ids"],
        source_groups=data["source_groups"],A_labels=labels[:,2],M_labels=labels[:,0])
    save(output/"complete.json",{"results":results,"development_only":True,
        "same_input_control":True,"original_calibration703_used":False})


if __name__ == "__main__":
    main()
