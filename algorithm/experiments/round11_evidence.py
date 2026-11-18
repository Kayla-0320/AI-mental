"""新旧语音的严格嵌套来源对照，区分质量修正、表征与可靠性证据收益。"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from experiments.round10_fusion import folds, take
from experiments.round10_evidence import supervision
from experiments.round10_domain import bias_select, weights_for
from experiments.round11_common import OUTPUT, digest, save
from experiments.round11_features import AUDIO_PACKS, TEXT, VISUAL, load_data, observed, transform

VIEWS = ("M","T","A","V")


def view_fields(view: str, encoder: str) -> tuple[str, ...]:
    """M/T/A/V四个监督任务各只使用其约定的观测。"""
    return {"M":TEXT+("face_summary","visual")+AUDIO_PACKS[encoder],
            "T":TEXT,"A":AUDIO_PACKS[encoder],"V":VISUAL}[view]


def visibility(data: dict[str, np.ndarray], view: str) -> np.ndarray:
    """真实缺失为缺证据，不把它映射成中性预测。"""
    if view in ("M","T"):
        return data["modality_mask"][:,0] & (data["quality"][:,0] > 0)
    if view == "A":
        return observed(data,"audio")
    return observed(data,"face_summary") | observed(data,"visual")


def fit_experts(data: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    """监督与回归都只见当前训练来源；变换使用FP64。"""
    classes, values = supervision(data)
    count = config["classes"]
    if count == 5:
        classes = np.digitize(values,[-.5,-.1,.1,.5])
    result = {}
    for j,view in enumerate(VIEWS):
        fields = view_fields(view,config["encoder"])
        x,scalers = transform(data,fields)
        active = visibility(data,view)
        weight = weights_for(classes[active,j],data["source_groups"][active],0.,.5)
        classifier = LogisticRegression(C=(.03 if view == "M" else .1)*2000/active.sum(),
                                        max_iter=700,random_state=42)
        classifier.fit(x[active],classes[active,j],sample_weight=weight)
        regression = Ridge(alpha=10.).fit(x[active],values[active,j],sample_weight=weight)
        result[view] = {"fields":fields,"scalers":scalers,"classifier":classifier,
                        "regression":regression,"classes":count}
    return result


def expert_evidence(experts: dict[str, Any], data: dict[str, np.ndarray]) -> np.ndarray:
    """四路概率/强度/冲突为模型证据，推理不调用监督标注。"""
    probabilities, regressions, masks = [],[],[]
    for view in VIEWS:
        member = experts[view]
        x,_ = transform(data,member["fields"],member["scalers"])
        active = visibility(data,view)
        p = np.zeros((len(x),member["classes"]),np.float64)
        p[:,member["classifier"].classes_] = member["classifier"].predict_proba(x)
        p[~active] = 0.
        regression = member["regression"].predict(x)
        regression[~active] = 0.
        probabilities.append(p)
        regressions.append(regression)
        masks.append(active)
    p = np.stack(probabilities,1)
    present = np.stack(masks,1)
    entropy = -(p*np.log(np.maximum(p,1e-12))).sum(2)
    expected = p@np.linspace(-1.,1.,p.shape[2])
    relations = []
    for a,b in ((1,2),(1,3),(2,3),(0,1),(0,2),(0,3)):
        left,right = p[:,a],p[:,b]
        mean = .5*(left+right)
        js = .5*((left*np.log(np.maximum(left,1e-12)/np.maximum(mean,1e-12))).sum(1)
                 +(right*np.log(np.maximum(right,1e-12)/np.maximum(mean,1e-12))).sum(1))
        active = present[:,a]&present[:,b]
        relations.extend((js*active,abs(expected[:,a]-expected[:,b])*active))
    return np.column_stack((p.reshape(len(p),-1),entropy,expected,np.stack(regressions,1),present,*relations))


def reliability(data: dict[str, np.ndarray]) -> np.ndarray:
    """由实际观测构造质量代理，不含来源ID、转录目标或人类情绪标签。"""
    a = observed(data,"sensevoice")
    native = data["sensevoice_emotion"].astype(np.float64)
    native = native-native.max(1,keepdims=True)
    p = np.exp(native)/np.exp(native).sum(1,keepdims=True)
    agreement = np.where(a[:,None],data["asr_alignment"],0.)
    return np.column_stack((data["quality"],data["modality_mask"],data["coverage"],
                            agreement,np.where(a[:,None],p,0.),a))


def crossfit(data: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    """内层每条证据来自没有见过该来源的四路专家。"""
    output = None
    for tr,va in folds(data,3):
        models = fit_experts(take(data,tr),config)
        value = expert_evidence(models,take(data,va))
        if output is None:
            output = np.full((len(data["sample_ids"]),value.shape[1]),np.nan)
        output[va] = value
    if output is None or not np.isfinite(output).all():
        raise ValueError("内层来源OOF覆盖不完整")
    return output


def main() -> None:
    """12个事先固定的配方共享相同外层来源；允许逐家族继续。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=int, nargs="+")
    args = parser.parse_args()
    output = OUTPUT/"evidence_deterministic"
    families = [{"quality":q,"encoder":e,"classes":n}
        for q,e,n in itertools.product(("legacy","pts"),AUDIO_PACKS,(3,5))]
    meta_configs = [{"depth":d,"reliability":r} for d,r in itertools.product((4,5),(False,True))]
    plan = {"families":families,"meta_configs":meta_configs,"iterations":300,"learning_rate":.03,
        "l2_leaf_reg":3.,"class_power":.5,"seed":42,
        "protocol":"4 whole-source outer folds;3 source inner folds;no outer early stopping",
        "bias_grid":"12 fixed points,.5 pooled macro-F1 + .5 source-balanced macro-F1",
        "quality_and_representation":"full2x3x2 factorial;old encoder missing cache completed before comparison",
        "preprocessing":"FP64 StandardScaler and training-only PCA32 for deep_face in every control",
        "calibration703_and_history821_used":False,"sensevoice_frontend_dither":0.,
        "code_sha256":digest(Path(__file__)),"features_code_sha256":digest(Path(__file__).with_name("round11_features.py"))}
    lock = output/"preregister.json"
    if lock.exists():
        if json.loads(lock.read_text("utf-8")) != plan:
            raise ValueError("预注册代码或方案不一致")
    else:
        save(lock,plan)
    requested = range(len(families)) if args.family is None else args.family
    data_cache = {}
    started = time.monotonic()
    for index in requested:
        config = families[index]
        folder = output/f"family{index:02d}"
        if (folder/"complete.json").exists():
            continue
        if config["quality"] not in data_cache:
            data_cache[config["quality"]] = load_data(config["quality"])
        data = data_cache[config["quality"]]
        predictions = np.full((len(meta_configs),len(data["labels"]),3),np.nan)
        folder.mkdir(parents=True,exist_ok=True)
        for f,(tr,va) in enumerate(folds(data)):
            path = folder/f"fold{f}.npz"
            if path.exists():
                with np.load(path,allow_pickle=False) as prior:
                    if not np.array_equal(prior["valid_ids"],data["sample_ids"][va]):
                        raise ValueError("断点来源顺序不一致")
                    predictions[:,va] = prior["probabilities"]
                continue
            training,validation = take(data,tr),take(data,va)
            inner = crossfit(training,config)
            experts = fit_experts(training,config)
            outer = expert_evidence(experts,validation)
            weight = weights_for(training["labels"],training["source_groups"],0.,.5)
            models,pp = [],[]
            for specification in meta_configs:
                x,v = inner,outer
                if specification["reliability"]:
                    x,v = np.column_stack((x,reliability(training))),np.column_stack((v,reliability(validation)))
                model = CatBoostClassifier(iterations=300,depth=specification["depth"],learning_rate=.03,
                    l2_leaf_reg=3.,loss_function="MultiClass",random_seed=42,thread_count=2,
                    verbose=False,allow_writing_files=False).fit(x,training["labels"],sample_weight=weight)
                pp.append(model.predict_proba(v))
                models.append(model)
            pp = np.stack(pp)
            predictions[:,va] = pp
            with (folder/f"fold{f}_models.pkl").open("wb") as handle:
                pickle.dump({"experts":experts,"meta_models":models,"meta_configs":meta_configs,
                    "config":config,"training_sample_ids":training["sample_ids"],
                    "training_groups":np.unique(training["source_groups"])},handle)
            np.savez_compressed(path,probabilities=pp,valid_ids=validation["sample_ids"],
                train_ids=training["sample_ids"],train_groups=np.unique(training["source_groups"]),
                valid_groups=np.unique(validation["source_groups"]),inner_evidence=inner,outer_evidence=outer)
            print(json.dumps({"family":index,"config":config,"outer_fold":f,
                "seconds":round(time.monotonic()-started,1)}),flush=True)
        rows = [{"config":config,"meta":specification,
                 **bias_select(data,np.log(np.maximum(p,1e-12)))} for specification,p in zip(meta_configs,predictions)]
        np.savez_compressed(folder/"oof.npz",probabilities=predictions,sample_ids=data["sample_ids"],
            labels=data["labels"],source_groups=data["source_groups"])
        save(folder/"complete.json",{"family":index,"results":rows,"development_only":True})
        winner = max(rows,key=lambda row:row["selection_score"])
        print(json.dumps({"family_complete":index,"winner":winner}),flush=True)


if __name__ == "__main__":
    main()
