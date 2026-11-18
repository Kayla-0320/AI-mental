"""中文主任务的准确率约束退化决策、最终拟合、可调用重放与条件区间。"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np
from scipy.special import softmax
from experiments.round10_fusion import folds,take
from experiments.round11_common import OUTPUT as ROUND11,digest
from experiments.round13_data import OUTPUT,save
from experiments.round13_models import CONDITIONS,Projection,SentimentLexicon,fit_heads,modify,donor_indices
from experiments.round13_train import read,ch_data
from experiments.round13_evaluate import choose,interval
from experiments.round13_neural import crossfit,train_head
from experiments.round13_runtime import CHPredictor,pack_raw,route
from experiments.round7_decision_oof import metric


def main() -> None:
    """仅用既有开发来源选择；新测试结果不用于本中文决策网格。"""
    folder = OUTPUT/"final_ch"
    if (folder/"verification.json").exists():
        return
    choice = choose("ch")
    config,family = choice["winner"]["config"],choice["winner"]["family"]
    bias_grid = [[a,b,0.] for a,b in itertools.product((-.25,0.,.25),(-1.,-.75,-.5,-.25,0.,.25))]
    plan = dict(selection=choice,bias_grid=bias_grid,branch_conditions={"5":"missing_A","3":"missing_V","6":"missing_T","1":"text_only"},
        constraints="branch macroF1 >= previous+.01; branch accuracy >= previous-.01; branch neutral F1 >= previous",
        full_observation="retain original7B decisions; original natural missing rows may route",
        evaluation="previously exposed development2876 only;no historical821/calibration703/new corpus test used for this bias search",
        code_sha256=digest(Path(__file__)))
    target = folder/"preregister.json"
    if target.exists():
        if json.loads(target.read_text("utf-8"))!=plan:
            raise ValueError("中文退化决策协议变化")
    else:
        save(target,plan)
    source = OUTPUT/("neural" if family=="neural" else "training")/"ch"/config["id"]/"development_predictions.npz"
    current = read(source)
    previous = read(OUTPUT.parent/"round12/previous_baselines/7b.npz")
    if not np.array_equal(current["sample_ids"],previous["sample_ids"]):
        raise ValueError("旧模型与新开发样本ID不一致")
    data = ch_data()
    priors = np.bincount(data["labels"],minlength=3)/len(data["labels"])
    biases,selection = {},[]
    for pattern,condition in ((5,"missing_A"),(3,"missing_V"),(6,"missing_T"),(1,"text_only")):
        baseline = metric(data["labels"],previous[condition].argmax(1))
        logits = np.log(np.maximum(current["specific_"+condition],1e-12))
        candidates = []
        for bias in bias_grid:
            p = softmax(logits+bias,axis=1)
            scores = metric(data["labels"],p.argmax(1))
            if (scores["macro_f1"]>=baseline["macro_f1"]+.01 and
                scores["accuracy"]>=baseline["accuracy"]-.01 and
                scores["per_class_f1"][1]>=baseline["per_class_f1"][1]):
                candidates.append(dict(bias=bias,metrics=scores))
        if candidates:
            best = max(candidates,key=lambda r:(r["metrics"]["macro_f1"],-sum(abs(v) for v in r["bias"])))
            biases[str(pattern)] = best["bias"]
            selection.append(dict(pattern=pattern,condition=condition,old=baseline,winner=best))
    save(folder/"decision_locked.json",dict(choice=choice,enabled_patterns=[int(k) for k in biases],
        bias_by_pattern=biases,branch_search=selection,independent_confirmation=False))
    hybrid = {c:np.full_like(previous[c],np.nan) for c in CONDITIONS}
    for f,(_,va) in enumerate(folds(data)):
        part = take(data,va)
        donors = donor_indices(part["source_groups"])
        audit = json.loads((OUTPUT.parent/"round12/audit"/f"fold{f}.json").read_text("utf-8"))
        if not np.array_equal(donors,np.array(audit["valid_donors"])):
            raise ValueError("压力供体与旧对照不一致")
        for condition in CONDITIONS:
            changed = modify(part,condition,donors)
            fallback = current["specific_"+condition][va].copy()
            codes = changed["mask"].astype(int) @ np.array([1,2,4])
            for code,bias in biases.items():
                active = codes==int(code)
                fallback[active] = softmax(np.log(np.maximum(fallback[active],1e-12))+bias,axis=1)
            hybrid[condition][va] = route(previous[condition][va],fallback,changed["mask"],[int(k) for k in biases])
    rows = []
    for condition in CONDITIONS:
        rows.append(dict(condition=condition,old=metric(data["labels"],previous[condition].argmax(1)),
            current=metric(data["labels"],hybrid[condition].argmax(1)),
            **interval(data["labels"],previous[condition],hybrid[condition],data["source_groups"])))
    np.savez_compressed(folder/"development_hybrid.npz",sample_ids=data["sample_ids"],labels=data["labels"],
        source_groups=data["source_groups"],**hybrid)
    save(folder/"development_results.json",dict(rows=rows,choice=choice,selection=selection,
        limitation="same previously exposed development sources; decision grid selected on these predictions; conditional intervals are optimistic after selection",
        not_independent_test=True))
    projection = Projection().fit(data)
    knowledge = SentimentLexicon().encode(data["raw_text"])
    if family == "neural":
        teacher = choice["teacher"]
        heads = fit_heads(projection.transform(data),data["labels"],teacher,knowledge)
        (folder/"final_teacher").mkdir(parents=True,exist_ok=True)
        oof = crossfit(data,teacher,folder/"final_teacher")
        _,ix = np.unique(data["source_groups"],return_index=True)
        train_head(config,data,take(data,np.sort(ix[:16])),projection,heads,oof,folder/"neural")
        import shutil
        shutil.copy2(folder/"neural/head.pt",folder/"head.pt")
    else:
        heads = fit_heads(projection.transform(data),data["labels"],config,knowledge)
    with (folder/"model.pkl").open("wb") as handle:
        pickle.dump(dict(projection=projection,heads=heads,family=family,config=config,training_ids=data["sample_ids"]),handle)
    baseline_model = ROUND11/"final_models/7b/model.pkl"
    save(folder/"model_manifest.json",dict(config=config,family=family,training_n=len(data["labels"]),training_splits=["train","selection"],
        enabled_patterns=[int(k) for k in biases],bias_by_pattern=biases,model_sha256=digest(folder/"model.pkl"),
        original_model_sha256=digest(baseline_model),original_model=str(baseline_model),
        raw_input_contract="same round11 7B language+audio+visual frozen features; no label argument required",
        statistical_scope="source-OOF development estimates; final fitted weights not scored on old tests",
        seed=42,original_seed_ensemble="original full model retains five heads; added fallback is one seed42 model"))
    from experiments.round11_features import load_data
    raw = load_data("pts")
    # 多个来源的前16个输入用于序列化与目标无关验证，不能当训练外成绩。
    _,ix = np.unique(raw["source_groups"],return_index=True)
    sample = take(raw,np.sort(ix[:16]))
    predictor = CHPredictor()
    p,details = predictor.predict(sample)
    no_labels = {k:v for k,v in sample.items() if k!="labels"}
    q,_ = CHPredictor().predict(no_labels)
    sample["labels"] = (sample["labels"]+1)%3
    again,_ = predictor.predict(sample)
    error = max(float(np.max(np.abs(p-q))),float(np.max(np.abs(p-again))))
    if error>1e-6 or not np.allclose(p.sum(1),1,atol=1e-6):
        raise ValueError("中文最终模型重放或目标独立性未通过")
    durations = []
    for _ in range(20):
        started = time.perf_counter()
        predictor.predict(no_labels)
        durations.append(time.perf_counter()-started)
    save(folder/"verification.json",dict(reload_and_gold_change_max_abs_error=error,sample_n=len(p),
        cached_feature_batch16_latency_p50=float(np.quantile(durations,.5)),cached_feature_batch16_latency_p95=float(np.quantile(durations,.95)),
        timing_scope="cached frozen features to prediction; excludes raw media/foundation encoders; not end-to-end latency",details=details))


if __name__ == "__main__":
    main()
