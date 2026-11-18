"""固定开发搜索与一次新语料测试：联合表示、可用性专头及知识对照。"""
from __future__ import annotations

import argparse
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
from experiments.round11_common import digest
from experiments.round13_data import OUTPUT,save
from experiments.round13_models import (NAMES,CONDITIONS,SentimentLexicon,Projection,configurations,
    fit_heads,predict_heads,modify,donor_indices)
from experiments.round7_decision_oof import metric


def read(path: Path) -> dict[str,np.ndarray]:
    """只读取调用者显式指定的一个划分。"""
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def ch_data() -> dict[str,np.ndarray]:
    """原CH开发来源，仅保留原始目标与完整联合语义；不打开703/821。"""
    from experiments.round11_features import load_data,observed
    data = load_data("pts")
    groups = {"T":("llm_last","aux_mean","bge_cls"),
              "A":("audio","speech","speech_emotion","sensevoice","sensevoice_query","sensevoice_emotion"),
              "V":("face_summary","deep_face","visual")}
    result = {k:data[k] for k in ("sample_ids","source_groups","raw_text","labels")}
    for name,fields in groups.items():
        result[name] = np.column_stack([data[f]/np.sqrt(data[f].shape[1]) for f in fields]).astype(np.float32)
    result["mask"] = np.column_stack((data["modality_mask"][:,0] & (data["quality"][:,0]>0),
        observed(data,"audio"),observed(data,"face_summary")|observed(data,"visual")))
    return result


def evaluate(bundle: dict[str,Any], projection: Projection, part: dict[str,np.ndarray],
             knowledge: np.ndarray) -> dict[str,np.ndarray]:
    """同一组固定跨来源供体供普通融合与专用分支共享。"""
    result = {}
    donors = donor_indices(part["source_groups"])
    for condition in CONDITIONS:
        changed = modify(part,condition,donors)
        projected = projection.transform(changed)
        for name,specialized in (("plain",False),("specific",True)):
            result[name+"_"+condition] = predict_heads(bundle,projected,knowledge,specialized)
    return result


def metrics(labels: np.ndarray, p: dict[str,np.ndarray], prefix: str) -> dict[str,Any]:
    """综合指标固定为干净、五压力和中性F1；全缺失不参与选型。"""
    values = {c:metric(labels,p[prefix+"_"+c].argmax(1)) for c in CONDITIONS}
    robust = float(np.mean([values[c]["macro_f1"] for c in CONDITIONS[1:6]]))
    score = .4*values["clean"]["macro_f1"] + .4*robust + .2*values["clean"]["per_class_f1"][1]
    return dict(metrics=values,robustness_mean=robust,selection_score=score)


def train(dataset: str) -> None:
    """CH四来源折；新语料官方train拟合、valid选型，测试工件不在训练入口出现。"""
    folder = OUTPUT / "training" / dataset
    if (folder/"selection_locked.json").exists():
        return
    grid = configurations()
    plan = dict(dataset=dataset,configs=grid,seed=42,conditions=list(CONDITIONS),
        projection="training-only scaler/PCA96 per modality; joint raw representations retained",
        heads="independent M3 heads per observed subset; human overall target unchanged",
        selection=".4clean+.4mean5stress+.2neutralF1; candidates must retain clean within .01 of best ordinary fusion; shuffled controls excluded",
        source_audit="ch:4whole-source development folds;new corpora:official train/valid;no sealed test access",
        source_sha256={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name("round13_models.py"))})
    path = folder/"preregister.json"
    if path.exists():
        if json.loads(path.read_text("utf-8")) != plan:
            raise ValueError("第十三轮开发协议发生变化")
    else:
        save(path,plan)
    if dataset == "ch":
        data = ch_data()
        splits = [(take(data,tr),take(data,va),va) for tr,va in folds(data)]
        y = data["labels"]
        ids = data["sample_ids"]
        groups = data["source_groups"]
    else:
        td,vd = (read(OUTPUT/"datasets"/dataset/f"{s}.npz") for s in ("train","valid"))
        splits = [(td,vd,np.arange(len(vd["labels"])))]
        y,ids,groups = vd["labels"],vd["sample_ids"],vd["source_groups"]
    lexicon = SentimentLexicon()
    predictions = {c["id"]:{kind+"_"+s:np.full((len(y),3),np.nan) for kind in ("plain","specific") for s in CONDITIONS} for c in grid}
    started = time.monotonic()
    for f,(td,vd,ix) in enumerate(splits):
        cache = folder/f"fold{f}/projection.pkl"
        cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.exists():
            with cache.open("rb") as handle:
                projection = pickle.load(handle)
        else:
            projection = Projection().fit(td)
            with cache.open("wb") as handle:
                pickle.dump(projection,handle)
        tx = projection.transform(td)
        tk,vk = lexicon.encode(td["raw_text"]),lexicon.encode(vd["raw_text"])
        if not (folder/f"fold{f}/source_audit.json").exists():
            save(folder/f"fold{f}/source_audit.json",dict(train_ids=td["sample_ids"].tolist(),valid_ids=vd["sample_ids"].tolist(),
                train_groups=np.unique(td["source_groups"]).tolist(),valid_groups=np.unique(vd["source_groups"]).tolist(),
                validation_donors=donor_indices(vd["source_groups"]).tolist(),knowledge_train_hits=int((tk[:,9]>0).sum()),
                knowledge_valid_hits=int((vk[:,9]>0).sum()),knowledge_terms=len(lexicon.words)))
        for config in grid:
            path = folder/config["id"]/f"fold{f}"
            path.mkdir(parents=True,exist_ok=True)
            output = path/"predictions.npz"
            if output.exists():
                p = read(output)
            else:
                bundle = fit_heads(tx,td["labels"],config,tk)
                with (path/"model.pkl").open("wb") as handle:
                    pickle.dump(bundle,handle)
                p = evaluate(bundle,projection,vd,vk)
                np.savez_compressed(output,**p)
            for key in predictions[config["id"]]:
                predictions[config["id"]][key][ix] = p[key]
            print(json.dumps(dict(dataset=dataset,fold=f,config=config["id"],seconds=round(time.monotonic()-started,1))),flush=True)
    rows = []
    for config in grid:
        p = predictions[config["id"]]
        if not all(np.isfinite(v).all() and np.allclose(v.sum(1),1.,atol=1e-6) for v in p.values()):
            raise ValueError("开发预测覆盖不完整或概率不归一")
        np.savez_compressed(folder/config["id"]/"development_predictions.npz",sample_ids=ids,labels=y,source_groups=groups,**p)
        rows.append(dict(config=config,plain=metrics(y,p,"plain"),specific=metrics(y,p,"specific")))
    ordinary = max([r for r in rows if not r["config"]["knowledge"]],key=lambda r:r["plain"]["metrics"]["clean"]["macro_f1"])
    base_clean = ordinary["plain"]["metrics"]["clean"]["macro_f1"]
    eligible = [r for r in rows if not r["config"]["shuffled"] and r["specific"]["metrics"]["clean"]["macro_f1"] >= base_clean-.01]
    winner = max(eligible,key=lambda r:r["specific"]["selection_score"])
    save(folder/"selection_locked.json",dict(rows=rows,winner=winner["config"],ordinary=ordinary["config"],
        selection_score=winner["specific"]["selection_score"],only_development=True,seconds=time.monotonic()-started))
    print(json.dumps(dict(dataset=dataset,winner=winner["config"]["id"],clean=winner["specific"]["metrics"]["clean"]["macro_f1"],
        robust=winner["specific"]["robustness_mean"])),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",choices=("ch","mosi","meld"))
    args = parser.parse_args()
    train(args.dataset)
