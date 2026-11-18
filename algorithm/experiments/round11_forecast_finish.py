"""补齐真实历史GRU与软投票，锁定后一次读取七类测试目标。"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import pickle
from typing import Any

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
import torch
from torch import nn
from scipy.special import softmax
from experiments.round11_common import OUTPUT,digest,save
from experiments.round11_forecast import load,metrics,summary,linear_predict
from experiments.round11_sequence_inputs import index_windows,materialize
from experiments.round11_external_data import targets


class ForecastGRU(nn.Module):
    """对有效历史做打包，填充不会影响末状态。"""
    def __init__(self) -> None:
        super().__init__()
        self.gru=nn.GRU(10,32,batch_first=True)
        self.head=nn.Linear(32,7)

    def forward(self,x: torch.Tensor,lengths: torch.Tensor) -> torch.Tensor:
        packed=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,hidden=self.gru(packed)
        return self.head(hidden[-1])


def prepare(sequence: np.ndarray,training: np.ndarray,scaling: dict[str,np.ndarray] | None = None) -> tuple[np.ndarray,np.ndarray,dict[str,np.ndarray]]:
    """只标准化过去可知的时长和年龄，统计量仅拟合训练窗口。"""
    valid=sequence[:,:,7]>0
    if scaling is None:
        z=sequence[training][valid[training]][:,8:]
        scaling={"mean":z.mean(0),"std":np.maximum(z.std(0),1e-5)}
    output=np.zeros_like(sequence)
    lengths=valid.sum(1).astype(np.int64)
    for i in range(len(sequence)):
        value=sequence[i,valid[i]].copy()
        value[:,8:]=(value[:,8:]-scaling["mean"])/scaling["std"]
        output[i,:lengths[i]]=value
    return output,lengths,scaling


def infer(model: ForecastGRU,x: np.ndarray,lengths: np.ndarray) -> np.ndarray:
    """CPU分批重放，不依赖目标标签。"""
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(x[i:i+256]),torch.from_numpy(lengths[i:i+256])).softmax(1).numpy()
            for i in range(0,len(x),256)])


def paired(y: np.ndarray,p: np.ndarray,baseline: np.ndarray,groups: np.ndarray) -> dict[str,Any]:
    """按对话重采样；这是既定录制来源条件下的区间。"""
    unique=np.unique(groups)
    matrices=[]
    for pred in (p.argmax(1),baseline.argmax(1)):
        matrices.append(np.stack([np.bincount(7*y[groups==g]+pred[groups==g],minlength=49).reshape(7,7) for g in unique]))
    rng=np.random.default_rng(42)
    ix=rng.integers(0,len(unique),size=(5000,len(unique)))
    vals=[]
    for array in matrices:
        cm=array[ix].sum(1);den=cm.sum(1)+cm.sum(2)
        f=np.divide(2*np.diagonal(cm,axis1=1,axis2=2),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(1)
        vals.append(f)
    return {"unit":"conversation;conditional on released test recordings/speaker IDs","n_clusters":len(unique),
        "draws":5000,"delta_macro_f1_95ci":np.quantile(vals[0]-vals[1],[.025,.975]).tolist()}


def main() -> None:
    """所有设置先完成开发筛选及文件摘要，再统一解封测试目标。"""
    torch.set_num_threads(1)
    output=OUTPUT/"forecast_final"
    if (output/"results.json").exists():
        print("forecast final already complete",flush=True);return
    plan={"contexts":[5,10,20],"epochs":[5,15],"hidden":32,"learning_rate":.002,"batch":128,
        "seed":42,"weight_decay":.001,"class_weights":"sqrt inverse train frequency",
        "vote":"linear-stage winner, best GRU, observed mean;nonnegative quarter simplex;valid macroF1",
        "settings":[[h,l] for h,l in itertools.product((1,3,5),(.25,1.25))],
        "fit":"official train only; upstream train probabilities source-OOF",
        "test_prior_exposure":"project earlier rounds used M7 oracle forecasting;current selection never reads test M7",
        "code_sha256":digest(Path(__file__)),"perception_sha256":digest(OUTPUT/"observed_perception/model.pkl")}
    pre=output/"preregister.json"
    if pre.exists():
        if json.loads(pre.read_text("utf-8"))!=plan:raise ValueError("GRU预注册不符")
    else:save(pre,plan)
    records,observations,labels=load()
    settings=[];locked_files=[OUTPUT/"observed_perception/model.pkl"];selection_files=[pre]
    for h,lead in plan["settings"]:
        name=f"h{h}_lead{lead}";folder=output/name;folder.mkdir(parents=True,exist_ok=True)
        oldfolder=OUTPUT/"forecast_linear"/name
        old=json.loads((oldfolder/"selection_locked.json").read_text("utf-8"))
        candidates=[]
        for context in plan["contexts"]:
            windows,_=index_windows(records,context,h,lead);data=materialize(windows,observations,context)
            tr=data["splits"]=="train";va=data["splits"]=="valid"
            y=np.array([labels[key] for key in data["target_ids"][tr]])
            vy=np.array([labels[key] for key in data["target_ids"][va]])
            x,lengths,scaling=prepare(data["sequence"],tr)
            done=folder/f"ctx{context}_complete.json"
            if done.exists():
                candidates.extend(json.loads(done.read_text("utf-8")));continue
            torch.manual_seed(42);rng=np.random.default_rng(42)
            model=ForecastGRU();optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
            counts=np.bincount(y,minlength=7);weight=torch.tensor(np.sqrt(len(y)/(7*np.maximum(counts,1))),dtype=torch.float32)
            loss_function=nn.CrossEntropyLoss(weight=weight)
            xt=torch.from_numpy(x[tr]);lt=torch.from_numpy(lengths[tr]);yt=torch.from_numpy(y)
            rows=[]
            for epoch in range(1,16):
                model.train()
                for batch in np.array_split(rng.permutation(len(y)),max(1,int(np.ceil(len(y)/128)))):
                    optimizer.zero_grad();loss=loss_function(model(xt[batch],lt[batch]),yt[batch]);loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                if epoch in plan["epochs"]:
                    p=infer(model,x,lengths);key=f"gru_ctx{context}_epoch{epoch}"
                    checkpoint=folder/f"{key}.pt"
                    torch.save({"state":model.state_dict(),"scaling":scaling,"context":context,
                        "training_ids":data["target_ids"][tr],"perception_sha256":plan["perception_sha256"]},checkpoint)
                    np.savez_compressed(folder/f"{key}.npz",sample_ids=data["target_ids"],probabilities=p)
                    reloaded=ForecastGRU();reloaded.load_state_dict(torch.load(checkpoint,weights_only=False)["state"])
                    delta=float(np.abs(infer(reloaded,x,lengths)-p).max())
                    if delta>1e-7:raise ValueError("GRU权重重放不一致")
                    rows.append({"name":key,"context":context,"epoch":epoch,"validation":metrics(vy,p[va]),
                        "path":str(checkpoint),"model_sha256":digest(checkpoint),"reload_max_abs_error":delta})
                    print(json.dumps({"forecast":name,**rows[-1]}),flush=True)
            save(done,rows);candidates.extend(rows)
        best=max(candidates,key=lambda r:(r["validation"]["macro_f1"],r["validation"]["accuracy"]))
        with np.load(folder/f"{best['name']}.npz") as z:gru=z["probabilities"];ids=z["sample_ids"]
        with np.load(oldfolder/"candidates"/old["winner"]["name"]/"predictions.npz") as z:
            if not np.array_equal(ids,z["sample_ids"]):raise ValueError("时序比较ID不符")
            linear=z["probabilities"]
        with np.load(oldfolder/"candidates/observed_mean_ctx20/predictions.npz") as z:mean=z["probabilities"]
        with np.load(oldfolder/"candidates/observed_last_ctx5/predictions.npz") as z:baseline=z["probabilities"]
        va=data["splits"]=="valid";vy=np.array([labels[key] for key in ids[va]])
        vote=[]
        for a in range(5):
            for b in range(5-a):
                weights=np.array([a,b,4-a-b])/4
                p=sum(w*q for w,q in zip(weights,(linear,gru,mean)))
                vote.append({"weights":weights.tolist(),"validation":metrics(vy,p[va])})
        winner=max(vote,key=lambda r:(r["validation"]["macro_f1"],r["validation"]["accuracy"],r["weights"][0]))
        p=sum(w*q for w,q in zip(winner["weights"],(linear,gru,mean)))
        lock={"setting":name,"gru":best,"linear":old["winner"],"vote":winner,"candidates":candidates,"vote_grid":vote,
            "tasks":["M_emotion7"],"all_selection_complete":True}
        selection=folder/"selection_locked.json"
        if selection.exists():
            if json.loads(selection.read_text("utf-8"))!=lock:raise ValueError("预测锁已发生变化")
        else:save(selection,lock)
        selection_files.extend((selection,oldfolder/"selection_locked.json"));locked_files.append(Path(best["path"]))
        oldmodel=oldfolder/"candidates"/old["winner"]["name"]/"model.pkl"
        if oldmodel.exists():locked_files.append(oldmodel)
        np.savez_compressed(folder/"locked_predictions.npz",sample_ids=ids,selected=p,linear=linear,gru=gru,
            observed_mean=mean,baseline=baseline,splits=data["splits"],groups=data["groups"],conversations=data["conversations"],
            cutoff=data["cutoff"],target_start=data["target_start"],lead_seconds=data["lead_seconds"])
        selection_files.append(folder/"locked_predictions.npz")
        settings.append(name)
    global_lock=output/"evaluation_lock.json"
    body={"tasks":["M_emotion7"],"all_selection_complete":True,"model_files":[{"path":str(p),"sha256":digest(p)} for p in locked_files],
        "selection_files":[{"path":str(p),"sha256":digest(p)} for p in selection_files],"settings":settings,
        "limitation":"M7 test was used by earlier project rounds;current selection was validation-only"}
    if global_lock.exists():
        if json.loads(global_lock.read_text("utf-8"))!=body:raise ValueError("全预测锁不符")
    else:save(global_lock,body)
    truth=targets(("test_inputs",),"M_emotion7",global_lock)
    results=[]
    for name in settings:
        with np.load(output/name/"locked_predictions.npz",allow_pickle=False) as z:
            te=z["splits"]=="test_inputs";y=np.array([truth[key] for key in z["sample_ids"][te]])
            row={"setting":name,"eligible_test_windows":int(te.sum()),"all_eligible_predicted":True,
                "minimum_actual_lead":float(z["lead_seconds"][te].min()),"metrics":{k:metrics(y,z[k][te]) for k in ("selected","linear","gru","observed_mean","baseline")},
                "paired":paired(y,z["selected"][te],z["baseline"][te],z["conversations"][te]),
                "by_recording":{str(g):metrics(y[z["groups"][te]==g],z["selected"][te][z["groups"][te]==g]) for g in np.unique(z["groups"][te])}}
            np.savez_compressed(output/name/"test_predictions.npz",sample_ids=z["sample_ids"][te],labels=y,
                probabilities=z["selected"][te],baseline=z["baseline"][te],conversations=z["conversations"][te])
        results.append(row);print(json.dumps(row),flush=True)
    save(output/"results.json",{"task":"future same-speaker categorical7","rows":results,"no_test_selection":True,
        "coverage":"all preregistered temporally eligible windows,not all utterances","history":"real transcript and audio source-OOF recognition;not gold states",
        "latency":".25 alignment tolerance;1.25 additionally reserves1s,does not prove hardware real-time SLA"})


if __name__=="__main__":main()
