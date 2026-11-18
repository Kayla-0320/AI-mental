"""冻结跨数据集M3迁移与仅官方训练监督的域适应，统一锁定后评测。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
from typing import Any
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
from scipy.special import softmax
from experiments.round11_common import OUTPUT,PREVIOUS,digest,save
from experiments import round11_observed_perception as perception
from experiments.round11_external_data import load_data,targets,read_rows
from experiments.round11_final import predict as new_predict
from experiments.round10_refined import predict as previous_predict
from experiments.round7_decision_oof import metric,probability_metrics
from sklearn.linear_model import LogisticRegression


def intervals(y: np.ndarray,p: np.ndarray,baseline: np.ndarray,conversations: np.ndarray) -> dict[str,Any]:
    """会话分组配对重采样，条件于现有两个录制组。"""
    unique=np.unique(conversations);rng=np.random.default_rng(42);draw=rng.integers(0,len(unique),(5000,len(unique)))
    values=[]
    for probability in (p,baseline):
        pred=probability.argmax(1)
        matrices=np.stack([np.bincount(3*y[conversations==g]+pred[conversations==g],minlength=9).reshape(3,3) for g in unique])
        cm=matrices[draw].sum(1);den=cm.sum(1)+cm.sum(2)
        values.append(np.divide(2*np.diagonal(cm,axis1=1,axis2=2),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(1))
    return {"unit":"conversation;conditional on official two test recording groups","clusters":len(unique),"draws":5000,
        "selected_macro_f1_95ci":np.quantile(values[0],[.025,.975]).tolist(),"delta_macro_f1_95ci":np.quantile(values[0]-values[1],[.025,.975]).tolist()}


def adapt() -> None:
    """额外监督的单独对照，不能冒充冻结模型的零样本迁移。"""
    output=OUTPUT/"external_m3/adaptation";output.mkdir(parents=True,exist_ok=True)
    if (output/"selection_locked.json").exists():return
    plan={"configs":[{"pack":p,"C":c} for p in perception.PACKS for c in (.03,.1,1.)],
        "target":"released human Continuous label_result sign ->3class;no7class relabel",
        "fit":"official train15413 only;valid1908 for selection;valid never merged into fit",
        "bias_grid":[[-.25,0.,.25],[-.5,-.25,0.,.25,.5]],"selection":".5 pooled macroF1+.5 equal-recording-weight macroF1",
        "code_sha256":digest(Path(__file__)),"perception_code_sha256":digest(Path(perception.__file__))}
    pre=output/"preregister.json"
    if pre.exists():
        if json.loads(pre.read_text("utf-8"))!=plan:raise ValueError("M3域适应计划不符")
    else:save(pre,plan)
    training=perception.load("train",False);valid=perception.load("valid",False)
    tr_y=targets(("train",),"M3");va_y=targets(("valid",),"M3")
    training["labels"]=np.array([tr_y[k] for k in training["sample_ids"]]);y=np.array([va_y[k] for k in valid["sample_ids"]])
    _,inverse,count=np.unique(valid["groups"],return_inverse=True,return_counts=True);sample_weight=1/count[inverse]
    rows=[];best=None
    for i,config in enumerate(plan["configs"]):
        path=output/f"candidate{i}.pkl";cached=output/f"candidate{i}.json"
        if cached.exists():
            row=json.loads(cached.read_text("utf-8"));rows.extend(row)
            continue
        x,scalers=perception.transform(training,config["pack"])
        counts=np.bincount(training["labels"],minlength=3)
        weights=np.sqrt(len(x)/(3*counts[training["labels"]]))
        classifier=LogisticRegression(C=config["C"]*2000/len(x),max_iter=600,random_state=42).fit(x,training["labels"],sample_weight=weights)
        model={"model":classifier,"scalers":scalers,"config":config,"training_ids":training["sample_ids"],"training_groups":np.unique(training["groups"])}
        with path.open("wb") as handle:pickle.dump(model,handle)
        # 七类感知器predict固定输出7列，本M3分类头按classes显式对齐3列。
        x,_=perception.transform(valid,config["pack"],model["scalers"])
        p=np.zeros((len(y),3));p[:,model["model"].classes_]=model["model"].predict_proba(x)
        config_rows=[]
        for negative in (-.25,0.,.25):
            for neutral in (-.5,-.25,0.,.25,.5):
                bias=[negative,neutral,0.];pred=(np.log(np.maximum(p,1e-12))+bias).argmax(1)
                m=metric(y,pred)
                cm=np.bincount(3*y+pred,weights=sample_weight,minlength=9).reshape(3,3);den=cm.sum(0)+cm.sum(1)
                balanced=float(np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den>0).mean())
                config_rows.append({"index":i,"config":config,"bias":bias,"validation":m,"source_balanced_f1":balanced,
                    "selection_score":.5*(m["macro_f1"]+balanced),"model_path":str(path),"model_sha256":digest(path)})
        save(cached,config_rows);rows.extend(config_rows)
        print(json.dumps({"adapt_candidate":i,"validation_best":max(r["validation"]["macro_f1"] for r in config_rows)}),flush=True)
    winner=max(rows,key=lambda r:(r["selection_score"],r["validation"]["macro_f1"],-sum(abs(v) for v in r["bias"])))
    save(output/"selection_locked.json",{"winner":winner,"candidates":rows,"fit_train_only":True,"test_labels_read":False})


def evaluate() -> None:
    """所有模型、决策与逐条无标签预测先锁定，M3测试成绩只计算一次。"""
    output=OUTPUT/"external_m3"
    if (output/"results.json").exists():return
    adaptation=json.loads((output/"adaptation/selection_locked.json").read_text("utf-8"))["winner"]
    oldpath=PREVIOUS/"refined_final/model.pkl";newpath=OUTPUT/"final_models/7b/model.pkl";lightpath=OUTPUT/"final_models/light/model.pkl"
    modelpaths=[oldpath,newpath,lightpath,Path(adaptation["model_path"])]
    for p in modelpaths:
        if not p.exists():raise ValueError("外部冻结评测缺少已训练模型")
    parts={};ids=None;groups=None;conversations=None
    # 先生成两个划分的无标签概率，然后统一核验摘要才打开测试目标。
    for split in ("valid","test_inputs"):
        target=output/f"{split}_locked_predictions.npz"
        if target.exists():continue
        olddata=load_data((split,),"legacy","7b");newdata=load_data((split,),"pts","7b")
        with oldpath.open("rb") as handle:old=previous_predict(pickle.load(handle),olddata)[0]
        with newpath.open("rb") as handle:new=new_predict(pickle.load(handle),newdata)
        with lightpath.open("rb") as handle:light=new_predict(pickle.load(handle),newdata)
        adata=perception.load(split,False)
        with Path(adaptation["model_path"]).open("rb") as handle:model=pickle.load(handle)
        x,_=perception.transform(adata,adaptation["config"]["pack"],model["scalers"])
        adapted=np.zeros((len(x),3));adapted[:,model["model"].classes_]=model["model"].predict_proba(x)
        adapted=softmax(np.log(np.maximum(adapted,1e-12))+adaptation["bias"],axis=1)
        if not np.array_equal(adata["sample_ids"],newdata["sample_ids"]):raise ValueError("外部域适应ID未对齐")
        np.savez_compressed(target,sample_ids=newdata["sample_ids"],old=old,new=new,light=light,adapted=adapted,
            groups=newdata["source_groups"],conversations=newdata["conversation_ids"])
    selection=[output/"adaptation/selection_locked.json",OUTPUT/"final_models/7b/selection_locked.json",OUTPUT/"final_models/light/selection_locked.json",
        output/"test_inputs_locked_predictions.npz",output/"valid_locked_predictions.npz"]
    lockbody={"tasks":["M3"],"all_selection_complete":True,
        "model_files":[{"path":str(p),"sha256":digest(p)} for p in modelpaths],
        "selection_files":[{"path":str(p),"sha256":digest(p)} for p in selection],
        "label_semantics":"unaltered released continuous polarity;documented category mismatch retained"}
    lock=output/"evaluation_lock.json"
    if lock.exists():
        if json.loads(lock.read_text("utf-8"))!=lockbody:raise ValueError("外部M3锁后数据或模型改变")
    else:save(lock,lockbody)
    results={}
    for split in ("valid","test_inputs"):
        ydict=targets((split,),"M3",lock if split=="test_inputs" else None)
        with np.load(output/f"{split}_locked_predictions.npz",allow_pickle=False) as z:
            y=np.array([ydict[k] for k in z["sample_ids"]]);rows={}
            for key in ("old","new","light","adapted"):
                raw=z[key];p=raw.mean(0) if raw.ndim==3 else raw
                rows[key]={"ensemble":metric(y,p.argmax(1)),"probability_metrics":probability_metrics(y,p),
                    "coverage":1.,"n":len(y),"confusion_matrix":np.bincount(3*y+p.argmax(1),minlength=9).reshape(3,3).tolist(),
                    "by_recording":{str(g):metric(y[z["groups"]==g],p[z["groups"]==g].argmax(1)) for g in np.unique(z["groups"])}}
                if raw.ndim==3:
                    rows[key]["five_seed_mean_macro_f1"]=float(np.mean([metric(y,q.argmax(1))["macro_f1"] for q in raw]))
                    rows[key]["five_seed_std_macro_f1"]=float(np.std([metric(y,q.argmax(1))["macro_f1"] for q in raw]))
            baseline=z["old"].mean(0)
            rows["new_vs_old_paired"]=intervals(y,z["new"].mean(0),baseline,z["conversations"])
            rows["adapted_vs_old_paired"]=intervals(y,z["adapted"],baseline,z["conversations"])
            np.savez_compressed(output/f"{split}_scored_predictions.npz",**{k:z[k] for k in z.files},labels=y)
        results[split]=rows;print(json.dumps({"external_M3":split,"scores":{k:v["ensemble"]["macro_f1"] for k,v in rows.items() if "ensemble" in v}}),flush=True)
    save(output/"results.json",{"task":"released human polarity3","splits":results,
        "frozen":"CH-only trained Round10/Round11;no external supervised fitting","adapted":"extra official EmotionTalk training supervision;not directly comparable training condition",
        "test_scope":"1929 all utterances;two recording groups;limited speaker diversity",
        "semantic_audit":"corpus/label_semantics_audit.json;do not identify seven-category happy with positive polarity"})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("adapt","evaluate"));args=parser.parse_args()
    adapt() if args.action=="adapt" else evaluate()
