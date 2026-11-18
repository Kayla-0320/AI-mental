"""真实识别历史的未来七类情绪预测；选择阶段不读取测试目标。"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import pickle
from typing import Any

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import StandardScaler
from experiments.round11_common import OUTPUT,digest,save
from experiments.round11_sequence_inputs import index_windows,materialize


def load() -> tuple[list[dict[str,Any]],dict[str,np.ndarray],dict[str,int]]:
    """只收集train/valid目标；测试输入无标签。"""
    timing={r["id"]:r for r in map(json.loads,(OUTPUT/"timestamps/records.jsonl").read_text("utf-8").splitlines())}
    records,observations,labels=[],{},{}
    for split in ("train","valid","test_inputs"):
        with np.load(OUTPUT/"observed_perception"/f"{split}_observations.npz",allow_pickle=False) as source:
            observations.update(zip(source["sample_ids"].astype(str),source["probabilities"]))
        with np.load(OUTPUT/"emotiontalk_features/sensevoice_deterministic_features"/f"{split}.npz",allow_pickle=False) as source:
            durations=dict(zip(source["sample_ids"].astype(str),source["original_duration"]))
        for row in map(json.loads,(OUTPUT/"corpus"/f"{split}.jsonl").read_text("utf-8").splitlines()):
            tm=timing[row["id"]]
            records.append({key:row[key] for key in ("id","group","conversation","speaker")}|
                {"split":split,"start":tm["start"],"end":max(tm["end"],tm["start"]+float(durations[row["id"]]))})
            if split!="test_inputs":labels[row["id"]]=int(row["M_emotion7"])
    return records,observations,labels


def summary(windows: list[dict[str,Any]],observations: dict[str,np.ndarray],drift: bool) -> np.ndarray:
    """个人基线、EMA与逐类别CUSUM只使用预测截止前的识别概率。"""
    result=[]
    for window in windows:
        p=np.stack([observations[key] for key in window["history_ids"]])
        prefix=np.stack([observations[key] for key in window["prefix_ids"]])
        features=[p[-1],p.mean(0),p.std(0)]
        if drift:
            mean=np.zeros(7);m2=np.zeros(7);ema=prefix[0].copy()
            base=prefix[:5].mean(0);scale=np.maximum(prefix[:5].std(0),.05)
            plus=np.zeros(7);minus=np.zeros(7)
            for i,value in enumerate(prefix):
                delta=value-mean;mean+=delta/(i+1);m2+=delta*(value-mean)
                ema=.2*value+.8*ema
                if i>=5:
                    z=np.clip((value-base)/scale,-5,5)
                    plus=np.minimum(50,np.maximum(0,plus+z-.25))
                    minus=np.minimum(50,np.maximum(0,minus-z-.25))
            features.extend((mean,np.sqrt(m2/max(1,len(prefix))),ema,plus,minus))
        features.append(np.array([np.log1p(len(prefix))]))
        result.append(np.concatenate(features))
    return np.stack(result)


def metrics(y: np.ndarray,p: np.ndarray) -> dict[str,Any]:
    """固定七类，零支持类别也不偷偷改变平均口径。"""
    return {"n":len(y),"macro_f1":float(f1_score(y,p.argmax(1),labels=np.arange(7),average="macro",zero_division=0)),
            "accuracy":float(accuracy_score(y,p.argmax(1)))}


def linear_fit(x: np.ndarray,y: np.ndarray,c: float) -> dict[str,Any]:
    """标准化只拟合训练窗口，类别权重固定平方根频率。"""
    scaler=StandardScaler().fit(x)
    counts=np.bincount(y,minlength=7)
    weights=(len(y)/(7*counts[y]))**.5
    model=LogisticRegression(C=c,max_iter=600,random_state=42).fit(scaler.transform(x),y,sample_weight=weights)
    return {"scaler":scaler,"classifier":model}


def linear_predict(model: dict[str,Any],x: np.ndarray) -> np.ndarray:
    """推理不需要任何标签或未来时间。"""
    p=np.zeros((len(x),7))
    p[:,model["classifier"].classes_]=model["classifier"].predict_proba(model["scaler"].transform(x))
    return p


def select(horizon: int,lead: float) -> None:
    """先锁定观察保持/均值/EMA/线性时序读出，GRU另行比较后才能总评。"""
    output=OUTPUT/"forecast_linear"/f"h{horizon}_lead{lead}"
    if (output/"complete.json").exists():return
    plan={"horizon":horizon,"minimum_lead_seconds":lead,"contexts":[5,10,20],"C":[.01,.1,1.],
        "feature_sets":["history","individual_drift"],"baselines":["observed_last","observed_mean","observed_EMA"],
        "task":"future same-speaker categorical7;history is source-OOF real transcript/audio recognition",
        "selection":"official valid only;models fit official train only;test targets sealed",
        "timing":"max actual decoded audio and annotation end over all past utterances;lead includes.25s alignment tolerance",
        "no_future_input":["target text","target recognition","target duration","target emotion"],
        "code_sha256":digest(Path(__file__)),"window_code_sha256":digest(Path(__file__).with_name("round11_sequence_inputs.py")),
        "perception_model_sha256":digest(OUTPUT/"observed_perception/model.pkl")}
    path=output/"preregister.json"
    if path.exists():
        if json.loads(path.read_text("utf-8"))!=plan:raise ValueError("预测预注册不符")
    else:save(path,plan)
    records,observations,labels=load()
    candidates=[]
    for context in (5,10,20):
        windows,audit=index_windows(records,context,horizon,lead)
        data=materialize(windows,observations,context)
        tr=data["splits"]=="train";va=data["splits"]=="valid"
        if set(data["groups"][tr])&set(data["groups"][va]):raise ValueError("预测来源重合")
        y=np.array([labels[key] for key in data["target_ids"][tr]])
        vy=np.array([labels[key] for key in data["target_ids"][va]])
        for baseline in ("observed_last","observed_mean","observed_EMA"):
            if baseline=="observed_last":p=data["last_observation"]
            elif baseline=="observed_mean":
                p=np.stack([np.stack([observations[key] for key in w["history_ids"]]).mean(0) for w in windows])
            else:
                values=[]
                for w in windows:
                    value=observations[w["prefix_ids"][0]].copy()
                    for key in w["prefix_ids"][1:]:value=.2*observations[key]+.8*value
                    values.append(value)
                p=np.stack(values)
            config={"kind":baseline,"context":context}
            name=f"{baseline}_ctx{context}"
            target=output/"candidates"/name
            target.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target/"predictions.npz",sample_ids=data["target_ids"],probabilities=p)
            candidates.append({"name":name,"config":config,"validation":metrics(vy,p[va])})
        for drift,c in itertools.product((False,True),(.01,.1,1.)):
            config={"kind":"linear","context":context,"drift":drift,"C":c}
            name=f"linear_ctx{context}_drift{drift}_C{c}"
            target=output/"candidates"/name
            target.mkdir(parents=True,exist_ok=True)
            done=target/"complete.json"
            if done.exists():
                row=json.loads(done.read_text("utf-8"))
                if digest(target/"model.pkl")!=row["model_sha256"]:raise ValueError("预测断点模型不符")
            else:
                x=summary(windows,observations,drift)
                model=linear_fit(x[tr],y,c)
                p=linear_predict(model,x)
                with (target/"model.pkl").open("wb") as handle:
                    pickle.dump({"fit":model,"config":config,"training_ids":data["target_ids"][tr],
                        "training_groups":np.unique(data["groups"][tr]),"perception_sha256":plan["perception_model_sha256"]},handle)
                np.savez_compressed(target/"predictions.npz",sample_ids=data["target_ids"],probabilities=p)
                row={"name":name,"config":config,"validation":metrics(vy,p[va]),"model_sha256":digest(target/"model.pkl")}
                save(done,row)
            candidates.append(row)
        print(json.dumps({"forecast_horizon":horizon,"lead":lead,"context":context,
            "best_validation":max(r["validation"]["macro_f1"] for r in candidates)}),flush=True)
    winner=max(candidates,key=lambda row:(row["validation"]["macro_f1"],row["validation"]["accuracy"]))
    save(output/"selection_locked.json",{"winner":winner,"candidates":candidates,"test_targets_used":False,
        "not_global_forecast_lock":"GRU and probability-vote controls still required before test scoring"})
    np.savez_compressed(output/"window_metadata.npz",target_ids=data["target_ids"],splits=data["splits"],
        conversations=data["conversations"],groups=data["groups"],cutoff=data["cutoff"],target_start=data["target_start"],lead_seconds=data["lead_seconds"])
    save(output/"complete.json",{"winner":winner,"training_windows":int(tr.sum()),"validation_windows":int(va.sum()),
        "test_input_windows":int((data["splits"]=="test_inputs").sum()),"test_targets_read":False})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--horizons",type=int,nargs="+",default=[1,3,5])
    parser.add_argument("--leads",type=float,nargs="+",default=[.25,1.25])
    args=parser.parse_args()
    for h in args.horizons:
        for lead in args.leads:select(h,lead)
