"""先锁模型和无标签逐条预测，再一次性评分新语料；禁止测试驱动重选。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
from typing import Any
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np
import torch
from experiments.round11_common import digest
from experiments.round13_data import OUTPUT,save
from experiments.round13_models import CONDITIONS,SentimentLexicon
from experiments.round13_neural import FrozenBankRouter,dispatch
from experiments.round13_train import read,evaluate
from experiments.round7_decision_oof import metric


def choose(dataset: str) -> dict[str,Any]:
    """按预先固定的开发综合规则选型，测试标签不进入函数。"""
    core = json.loads((OUTPUT/"training"/dataset/"selection_locked.json").read_text("utf-8"))
    neural = json.loads((OUTPUT/"neural"/dataset/"selection_locked.json").read_text("utf-8"))
    ordinary = next(r for r in core["rows"] if r["config"]["id"]==core["ordinary"]["id"])
    base_clean = ordinary["plain"]["metrics"]["clean"]["macro_f1"]
    candidates = [dict(family="specific",**r) for r in core["rows"] if not r["config"]["shuffled"]]
    candidates += [dict(family="neural",**r) for r in neural["rows"]]
    candidates = [r for r in candidates if r["specific"]["metrics"]["clean"]["macro_f1"]>=base_clean-.01]
    best = max(candidates,key=lambda r:r["specific"]["selection_score"])
    return dict(dataset=dataset,winner=best,ordinary=ordinary,teacher=neural["teacher"],
        rule=".4clean+.4mean5stress+.2neutralF1; clean >= best ordinary-.01; all from development only")


def load(dataset: str,choice: dict[str,Any]) -> tuple[Any,Any,Any,Any]:
    """恢复完整预处理与模型，不拟合当前输入。"""
    folder = OUTPUT/"training"/dataset
    with (folder/"fold0/projection.pkl").open("rb") as handle:
        projection = pickle.load(handle)
    with (folder/choice["ordinary"]["config"]["id"]/"fold0/model.pkl").open("rb") as handle:
        baseline = pickle.load(handle)
    winner = choice["winner"]
    model = None
    if winner["family"]=="specific":
        with (folder/winner["config"]["id"]/"fold0/model.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
    else:
        with (folder/choice["teacher"]["id"]/"fold0/model.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        path = OUTPUT/"neural"/dataset/winner["config"]["id"]/"fold0/head.pt"
        weights = torch.load(path,map_location="cpu",weights_only=False)
        model = FrozenBankRouter(weights["dimensions"],weights["config"],weights["full"],weights["prior"])
        model.load_state_dict(weights["state_dict"],strict=True)
        model.eval()
    return projection,baseline,bundle,model


def predict(dataset: str,choice: dict[str,Any],inputs: dict[str,np.ndarray]) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """标签无关的所有条件推理，返回固定来源的全覆盖概率。"""
    projection,baseline,bundle,model = load(dataset,choice)
    knowledge = SentimentLexicon().encode(inputs["raw_text"])
    old = evaluate(baseline,projection,inputs,knowledge)
    if model is None:
        new = evaluate(bundle,projection,inputs,knowledge)
    else:
        from experiments.round13_models import donor_indices,modify
        donors = donor_indices(inputs["source_groups"])
        new = {}
        for condition in CONDITIONS:
            new["specific_"+condition],_ = dispatch(model,bundle,projection.transform(modify(inputs,condition,donors)))
    values = {"baseline_"+c:old["plain_"+c] for c in CONDITIONS}
    values.update({"candidate_"+c:new["specific_"+c] for c in CONDITIONS})
    return values,dict(candidate=choice["winner"]["config"]["id"],family=choice["winner"]["family"])


def difference(labels: np.ndarray,old: np.ndarray,new: np.ndarray,neutral: bool = False) -> float:
    """固定M3或中性F1差值。"""
    a,b = metric(labels,old),metric(labels,new)
    return float(b["per_class_f1"][1]-a["per_class_f1"][1] if neutral else b["macro_f1"]-a["macro_f1"])


def interval(labels: np.ndarray,old: np.ndarray,new: np.ndarray,groups: np.ndarray,
             neutral: bool = False,replications: int = 3000) -> dict[str,Any]:
    """来源配对bootstrap区间与来源成组交换置换检验，固定种子42。"""
    old,new = old.argmax(1),new.argmax(1)
    rng = np.random.default_rng(42)
    blocks = [np.flatnonzero(groups==g) for g in np.unique(groups)]
    deltas = []
    for _ in range(replications):
        ix = np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])
        deltas.append(difference(labels[ix],old[ix],new[ix],neutral))
    actual = difference(labels,old,new,neutral)
    count = 0
    for _ in range(2000):
        swap = np.zeros(len(labels),bool)
        for j in np.flatnonzero(rng.integers(0,2,len(blocks))):
            swap[blocks[j]] = True
        a,b = np.where(swap,new,old),np.where(swap,old,new)
        count += difference(labels,a,b,neutral)>=actual-1e-12
    return dict(delta=actual,paired_group_ci95=np.quantile(deltas,[.025,.975]).tolist(),
        permutation_p_improvement=(count+1)/2001,group_count=len(blocks),bootstrap_replications=replications,
        assumption="conditions on published sources; group prediction exchangeability under null; not population-wide guarantee")


def score(dataset: str) -> None:
    """唯一正式新测试评分入口；结果存在后只能重放，不能追加改权重。"""
    folder = OUTPUT/"new_test"/dataset
    if (folder/"results.json").exists():
        return
    choice = choose(dataset)
    weight_files = [OUTPUT/"training"/dataset/"fold0/projection.pkl",
                    OUTPUT/"training"/dataset/choice["ordinary"]["config"]["id"]/"fold0/model.pkl"]
    if choice["winner"]["family"]=="specific":
        weight_files += [OUTPUT/"training"/dataset/choice["winner"]["config"]["id"]/"fold0/model.pkl"]
    else:
        weight_files += [OUTPUT/"training"/dataset/choice["teacher"]["id"]/"fold0/model.pkl",
                        OUTPUT/"neural"/dataset/choice["winner"]["config"]["id"]/"fold0/head.pt"]
    lock = dict(choice=choice,models=[dict(path=str(p),sha256=digest(p)) for p in weight_files],
        evaluation_source_sha256=digest(Path(__file__)),test_inputs_sha256=digest(OUTPUT/"datasets"/dataset/"test_inputs.npz"),
        sequence="selection + model lock -> inference + no-label prediction lock -> target scoring")
    target = folder/"evaluation_lock.json"
    if target.exists():
        if json.loads(target.read_text("utf-8"))!=lock:
            raise ValueError("评测锁后模型/决策发生变化")
    else:
        save(target,lock)
    inputs = read(OUTPUT/"datasets"/dataset/"test_inputs.npz")
    if "labels" in inputs:
        raise ValueError("测试输入含标签")
    prediction_path = folder/"predictions_no_labels.npz"
    if prediction_path.exists():
        values = read(prediction_path)
        ids = values.pop("sample_ids")
        if not np.array_equal(ids,inputs["sample_ids"]):
            raise ValueError("逐条预测ID不符")
    else:
        values,_ = predict(dataset,choice,inputs)
        if not all(np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6) for p in values.values()):
            raise ValueError("测试概率覆盖或归一化异常")
        np.savez_compressed(prediction_path,sample_ids=inputs["sample_ids"],**values)
        save(folder/"predictions_locked.json",dict(sha256=digest(prediction_path),labels_used=False,
            model_lock_sha256=digest(target),n=len(inputs["sample_ids"])))
    # 此点之前，当前进程不打开封存测试标签工件。
    gold = read(OUTPUT/"datasets"/dataset/"sealed_test_labels.npz")
    if not np.array_equal(gold["sample_ids"],inputs["sample_ids"]):
        raise ValueError("封存标签ID未对齐")
    labels = gold["labels"]
    rows = []
    for condition in CONDITIONS:
        old,new = values["baseline_"+condition],values["candidate_"+condition]
        statistics = interval(labels,old,new,inputs["source_groups"])
        rows.append(dict(condition=condition,n=len(labels),coverage=1.,baseline=metric(labels,old.argmax(1)),
                         candidate=metric(labels,new.argmax(1)),**statistics))
        print(json.dumps(dict(dataset=dataset,condition=condition,delta=statistics["delta"])),flush=True)
    primary = []
    seen = set()
    for c in ("clean","missing_A","text_only"):
        signature = (values["baseline_"+c].argmax(1).tobytes(),values["candidate_"+c].argmax(1).tobytes())
        if signature in seen:
            continue
        seen.add(signature)
        row = next(r for r in rows if r["condition"]==c)
        primary.append(dict(endpoint=c,delta=row["delta"],p=row["permutation_p_improvement"]))
    neutral = interval(labels,values["baseline_clean"],values["candidate_clean"],inputs["source_groups"],True)
    primary.append(dict(endpoint="clean_neutral_F1",delta=neutral["delta"],p=neutral["permutation_p_improvement"]))
    ordered = sorted(primary,key=lambda r:r["p"])
    running = 0.
    for i,row in enumerate(ordered):
        running = max(running,(len(ordered)-i)*row["p"])
        row["holm_adjusted_p"] = min(1.,running)
    replay,_ = predict(dataset,choice,inputs)
    err = max(float(np.max(np.abs(replay[k]-v))) for k,v in values.items())
    if err>1e-6:
        raise ValueError("新测试模型重放误差过大")
    save(folder/"results.json",dict(dataset=dataset,choice=choice,rows=rows,primary_endpoints=primary,
        clean_neutral_comparison=neutral,replay_max_abs_error=err,selection_after_test=False,
        training_condition="extra official corpus training supervision; not frozen CH zero-shot transfer",
        limits="English new corpora; not evidence of independent Chinese target-population generalization; no test-based reselection"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",choices=("mosi","meld"))
    args = parser.parse_args()
    score(args.dataset)
