"""真实转录/语音观测的七类感知器：训练来源OOF历史，验证测试只用训练来源拟合。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
from typing import Any

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from experiments.round11_common import OUTPUT,digest,save

PACKS={"text":("aux_mean","bge_cls"),"audio":("sensevoice","sensevoice_query","sensevoice_emotion"),
       "text_audio":("aux_mean","bge_cls","sensevoice","sensevoice_query","sensevoice_emotion")}


def load(split: str,targets: bool) -> dict[str,np.ndarray]:
    """输入不含人工历史状态，七类标签只为训练和开发评测读取。"""
    rows=[json.loads(line) for line in (OUTPUT/"corpus"/f"{split}.jsonl").read_text("utf-8").splitlines()]
    data={"sample_ids":np.array([row["id"] for row in rows]),"groups":np.array([row["group"] for row in rows])}
    for component,fields in (("bge_features",("bge_cls",)),("auxiliary_features",("aux_mean",)),
                             ("sensevoice_deterministic_features",("sensevoice","query","emotion_logprobs","available"))):
        with np.load(OUTPUT/"emotiontalk_features"/component/f"{split}.npz",allow_pickle=False) as source:
            if not np.array_equal(source["sample_ids"],data["sample_ids"]):
                raise ValueError("观测感知器输入ID未对齐")
            for field in fields:
                if field=="query":data["sensevoice_query"]=source[field][:,1,:]
                elif field=="emotion_logprobs":data["sensevoice_emotion"]=source[field][:,1,:]
                elif field=="available":data["audio_available"]=source[field]
                else:data[field]=source[field]
    if targets:
        if split=="test_inputs":
            raise ValueError("感知器选择阶段禁止读取测试标签")
        data["labels"]=np.array([row["M_emotion7"] for row in rows])
    return data


def take(data: dict[str,np.ndarray],indices: np.ndarray) -> dict[str,np.ndarray]:
    """所有输入与标注保持同一来源切片。"""
    return {key:values[indices] for key,values in data.items()}


def transform(data: dict[str,np.ndarray],pack: str,scalers: dict[str,Any] | None = None
              ) -> tuple[np.ndarray,dict[str,Any]]:
    """折内拟合有效观测的FP64统计，输出紧凑float32分类输入。"""
    fitting=scalers is None
    scalers={} if fitting else scalers
    blocks=[]
    for name in PACKS[pack]:
        present=data["audio_available"] if name.startswith("sensevoice") else np.ones(len(data["sample_ids"]),bool)
        values=data[name]
        if fitting:scalers[name]=StandardScaler().fit(values[present].astype(np.float64))
        scale=scalers[name]
        values=scale.transform(np.where(present[:,None],values,scale.mean_))/np.sqrt(values.shape[1])
        blocks.extend((np.where(present[:,None],values,0.).astype(np.float32),present[:,None].astype(np.float32)*.1))
    return np.column_stack(blocks),scalers


def fit(data: dict[str,np.ndarray],config: dict[str,Any]) -> dict[str,Any]:
    """感知器只拟合当前训练来源，分类任务固定七类。"""
    x,scalers=transform(data,config["pack"])
    y=data["labels"]
    counts=np.bincount(y,minlength=7)
    weights=(len(y)/(7*counts[y]))**.5
    model=LogisticRegression(C=config["C"]*2000/len(y),max_iter=600,random_state=42)
    model.fit(x,y,sample_weight=weights)
    return {"model":model,"scalers":scalers,"config":config,
            "training_ids":data["sample_ids"],"training_groups":np.unique(data["groups"])}


def predict(model: dict[str,Any],data: dict[str,np.ndarray]) -> np.ndarray:
    """预测只消费观测与保存的变换，不查询任何人类目标。"""
    x,_=transform(data,model["config"]["pack"],model["scalers"])
    result=np.zeros((len(x),7))
    result[:,model["model"].classes_]=model["model"].predict_proba(x)
    return result


def metric(y: np.ndarray,p: np.ndarray) -> dict[str,Any]:
    """七类指标单列，不能与主任务M3的macro-F1混算。"""
    predicted=p.argmax(1)
    return {"n":len(y),"macro_f1":float(f1_score(y,predicted,labels=np.arange(7),average="macro",zero_division=0)),
            "accuracy":float(accuracy_score(y,predicted)),"confusion_matrix":confusion_matrix(y,predicted,labels=np.arange(7)).tolist()}


def main() -> None:
    """官方训练/验证选择完才生成训练组OOF与无标签测试感知概率。"""
    output=OUTPUT/"observed_perception"
    if (output/"complete.json").exists():
        raise ValueError("拒绝覆盖真实观测感知模型")
    train,valid=load("train",True),load("valid",True)
    options=[{"pack":pack,"C":c} for pack in PACKS for c in (.03,.1,1.)]
    plan={"task":"categorical7 multimodal emotion from observed transcript/audio;separate from polarity3",
        "configs":options,"selection":"official validG01/G12 only;train15413 only fitting;valid not merged",
        "training_observations":"5 whole-recording-group crossfit with selected classifier recipe",
        "test":"input-only predictions,no target access;final forecast/evaluation locks required before scoring",
        "input_condition":"official transcription plus raw-audio SenseVoice;no human emotion as input",
        "code_sha256":digest(Path(__file__))}
    path=output/"preregister.json"
    if path.exists():
        if json.loads(path.read_text("utf-8"))!=plan:raise ValueError("观测感知器预注册不一致")
    else:save(path,plan)
    trials=[]
    for i,config in enumerate(options):
        trial=output/"validation_search"/f"candidate{i}.json"
        weights=trial.with_suffix(".pkl")
        if trial.exists():
            result=json.loads(trial.read_text("utf-8"))
            if result["config"]!=config or digest(weights)!=result["model_sha256"]:
                raise ValueError("开发感知器断点身份不符")
        else:
            model=fit(train,config)
            p=predict(model,valid)
            weights.parent.mkdir(parents=True,exist_ok=True)
            with weights.open("wb") as handle:pickle.dump(model,handle)
            result={"config":config,"validation":metric(valid["labels"],p),
                    "model_sha256":digest(weights),"candidate_index":i}
            save(trial,result)
        trials.append(result)
        print(json.dumps({"candidate":i,"config":config,"validation":result["validation"]["macro_f1"]}),flush=True)
    winner=max(trials,key=lambda row:(row["validation"]["macro_f1"],row["validation"]["accuracy"]))
    selected={"winner":winner,"candidates":trials,"test_targets_used":False}
    selected_path=output/"selection_locked.json"
    if selected_path.exists():
        if json.loads(selected_path.read_text("utf-8"))!=selected:raise ValueError("感知器选型断点不符")
    else:save(selected_path,selected)
    oof=np.full((len(train["sample_ids"]),7),np.nan)
    splitter=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    for fold,(tr,va) in enumerate(splitter.split(train["labels"],train["labels"],train["groups"])):
        directory=output/f"fold{fold}"
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/"observations.npz").exists():
            with np.load(directory/"observations.npz",allow_pickle=False) as prior:
                if not np.array_equal(prior["sample_ids"],train["sample_ids"][va]):raise ValueError("观测OOF断点ID不符")
                oof[va]=prior["probabilities"]
            split=json.loads((directory/"split.json").read_text("utf-8"))
            if digest(directory/"model.pkl")!=split["model_sha256"]:raise ValueError("观测OOF断点模型不符")
            if set(split["training_groups"])&set(split["observation_groups"]):raise ValueError("观测OOF来源重合")
            continue
        model=fit(take(train,tr),winner["config"])
        if set(model["training_groups"])&set(train["groups"][va]):raise ValueError("训练观测来源泄漏")
        p=predict(model,take(train,va))
        oof[va]=p
        with (directory/"model.pkl").open("wb") as handle:pickle.dump(model,handle)
        save(directory/"split.json",{"training_ids":train["sample_ids"][tr].tolist(),
            "observation_ids":train["sample_ids"][va].tolist(),"training_groups":np.unique(train["groups"][tr]).tolist(),
            "observation_groups":np.unique(train["groups"][va]).tolist(),"model_sha256":digest(directory/"model.pkl")})
        np.savez_compressed(directory/"observations.npz",sample_ids=train["sample_ids"][va],probabilities=p)
    if not np.isfinite(oof).all():raise ValueError("训练感知OOF没有全覆盖")
    np.savez_compressed(output/"train_observations.npz",sample_ids=train["sample_ids"],probabilities=oof)
    with (output/"validation_search"/f"candidate{winner['candidate_index']}.pkl").open("rb") as handle:
        model=pickle.load(handle)
    with (output/"model.pkl").open("wb") as handle:pickle.dump(model,handle)
    for split in ("valid","test_inputs"):
        data=load(split,False)
        if set(model["training_groups"])&set(data["groups"]):raise ValueError("官方留出来源重合")
        p=predict(model,data)
        np.savez_compressed(output/f"{split}_observations.npz",sample_ids=data["sample_ids"],probabilities=p)
    save(output/"complete.json",{"winner":winner,"training_n":len(train["sample_ids"]),
        "oof_training_history":True,"raw_observations_not_oracle_labels":True,
        "test_targets_read":False,"model_sha256":digest(output/"model.pkl"),
        "training_group_OOF_metrics":metric(train["labels"],oof)})


if __name__=="__main__":
    main()
