"""固定来源OOF专家后学习门控，及从完整融合教师蒸馏原始联合表示。"""
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
import torch
from torch import nn,Tensor
from experiments.round10_fusion import folds,take
from experiments.round11_common import digest
from experiments.round13_data import OUTPUT,save
from experiments.round13_models import (NAMES,CONDITIONS,Projection,SentimentLexicon,fit_heads,
    matrix,probabilities,modify,donor_indices,observed)
from experiments.round13_train import read,ch_data,metrics


def options() -> list[dict[str,Any]]:
    """固定6个等更新预算候选，比较门控与正确融合教师。"""
    return [dict(id=n,mode=m,kd=k) for n,m,k in (
        ("dense_bank","dense",0.),("top2_bank","top2",0.),
        ("raw_mlp","mlp",0.),("raw_mlp_kd25","mlp",.25),
        ("raw_mlp_kd50","mlp",.5),("linear_kd25","linear",.25))]


def bank_predict(bundle: dict[str,Any], projected: dict[str,np.ndarray]) -> np.ndarray:
    """七个专家均监督同一个M3任务；返回缺失时独立的资格mask。"""
    p = np.tile(bundle["prior"],(len(projected["mask"]),7,1))
    for pattern,head in bundle["models"].items():
        p[:,pattern-1] = probabilities(head,matrix(projected,pattern,None))
    return p.astype(np.float32)


def flat(projected: dict[str,np.ndarray]) -> np.ndarray:
    """将真实投影与可用性拼为无标签门控/学生输入。"""
    return np.column_stack([projected[n] for n in NAMES]+[projected["mask"]]).astype(np.float32)


def eligible(mask: Tensor, full: int) -> Tensor:
    """专家所需模态必须全部可见；全缺失临时放开一个位置防止NaN。"""
    pattern = mask.long() @ torch.tensor([1,2,4],device=mask.device)
    bits = torch.arange(1,8,device=mask.device)
    result = (pattern[:,None] & bits[None]) == bits[None]
    result &= (bits & ~full)[None] == 0
    result[pattern==0,0] = True
    return result


class FrozenBankRouter(nn.Module):
    """门控与专家分阶段训练，完整联合专家保留为锚点，推理可先路由后执行。"""

    def __init__(self, dimensions: int, config: dict[str,Any], full: int, prior: np.ndarray) -> None:
        super().__init__()
        self.config,self.full = config,full
        self.register_buffer("prior",torch.tensor(prior,dtype=torch.float32))
        output = 7 if config["mode"] in ("dense","top2") else 3
        self.network = nn.Linear(dimensions,output) if config["mode"]=="linear" else nn.Sequential(
            nn.Linear(dimensions,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,output))
        if output == 7:
            final = self.network[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            with torch.no_grad():
                final.bias[full-1] = 2.

    def weights(self, x: Tensor, mask: Tensor) -> Tensor:
        """Top-2在调用专家头之前选索引，不能用已执行的专家输出冒充稀疏执行。"""
        scores = self.network(x).masked_fill(~eligible(mask,self.full),-torch.inf)
        if self.config["mode"] == "top2":
            selected = torch.zeros_like(scores,dtype=torch.bool).scatter_(1,scores.topk(2,1).indices,True)
            scores = scores.masked_fill(~selected,-torch.inf)
        return scores.softmax(1)

    def forward(self,x: Tensor,mask: Tensor,bank: Tensor) -> tuple[Tensor,Tensor]:
        """固定专家概率不回传梯度，避免门控和专家同时塌缩。"""
        if self.config["mode"] in ("dense","top2"):
            gate = self.weights(x,mask)
            p = (gate[:,:,None]*bank).sum(1)
        else:
            p = self.network(x).softmax(1)
            gate = torch.ones((len(x),1),device=x.device)
        p = torch.where(mask.any(1)[:,None],p,self.prior[None])
        return p.clamp_min(1e-9).log(),gate


def dispatch(model: FrozenBankRouter,bundle: dict[str,Any],projected: dict[str,np.ndarray]
             ) -> tuple[np.ndarray,dict[str,Any]]:
    """先算门控，仅对正门控的样本调用相应M3头，记录实际调用对数。"""
    x = torch.from_numpy(flat(projected))
    mask = torch.from_numpy(projected["mask"])
    model.eval()
    with torch.inference_mode():
        if model.config["mode"] not in ("dense","top2"):
            p,_ = model(x,mask,torch.zeros(len(x),7,3))
            return p.exp().numpy(),dict(executed_pairs=0,kind="student")
        weights = model.weights(x,mask).numpy()
    p = np.zeros((len(x),3),np.float32)
    calls = 0
    for pattern,head in bundle["models"].items():
        rows = np.flatnonzero(weights[:,pattern-1]>0)
        if not len(rows):
            continue
        features = matrix(projected,pattern,None)
        p[rows] += weights[rows,pattern-1:pattern] * probabilities(head,features[rows])
        calls += len(rows)
    p[~projected["mask"].any(1)] = bundle["prior"]
    return p,dict(executed_pairs=calls,mean_heads=calls/len(x),kind="actual expert dispatch")


def crossfit(td: dict[str,np.ndarray],config: dict[str,Any],folder: Path) -> dict[str,np.ndarray]:
    """内层投影、全部专家都不见该来源，提供完整融合教师及按场景专家证据。"""
    path = folder/"inner_oof.npz"
    if path.exists():
        return read(path)
    outputs = {c:np.full((len(td["labels"]),7,3),np.nan,np.float32) for c in CONDITIONS[:5]}
    donor_audit = []
    for f,(tr,va) in enumerate(folds(td,3)):
        a,b = take(td,tr),take(td,va)
        projection = Projection().fit(a)
        bundle = fit_heads(projection.transform(a),a["labels"],config,np.zeros((len(tr),12)))
        for condition in outputs:
            outputs[condition][va] = bank_predict(bundle,projection.transform(modify(b,condition)))
        donor_audit.append(dict(inner_fold=f,train_groups=np.unique(a["source_groups"]).tolist(),
            valid_groups=np.unique(b["source_groups"]).tolist()))
        print(json.dumps(dict(inner_teacher=f,n=len(va))),flush=True)
    if not all(np.isfinite(p).all() for p in outputs.values()):
        raise ValueError("源外OOF教师覆盖不完整")
    np.savez_compressed(path,**outputs)
    save(folder/"teacher_source_audit.json",dict(folds=donor_audit,teacher=config,
        target="complete learned M3 fusion classifier on raw T/A/V, source-OOF; no in-sample meta teacher"))
    return outputs


def train_head(config: dict[str,Any],td: dict[str,np.ndarray],vd: dict[str,np.ndarray],projection: Projection,
               bundle: dict[str,Any],oof: dict[str,np.ndarray],folder: Path) -> dict[str,np.ndarray]:
    """每批均衡采样完整/缺失输入，错配不进入有硬标签训练。"""
    torch.manual_seed(42)
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    conditions = CONDITIONS[:5]
    training = {c:projection.transform(modify(td,c)) for c in conditions}
    dims = flat(training["clean"]).shape[1]
    model = FrozenBankRouter(dims,config,bundle["full"],bundle["prior"])
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02)
    y = torch.tensor(td["labels"],dtype=torch.long)
    counts = torch.bincount(y,minlength=3).float()
    weights = (len(y)/(3*counts)).sqrt()
    xs = {c:torch.from_numpy(flat(v)) for c,v in training.items()}
    ms = {c:torch.from_numpy(v["mask"]) for c,v in training.items()}
    banks = {c:torch.from_numpy(oof[c]) for c in conditions}
    teacher = banks["clean"][:,bundle["full"]-1]
    for _ in range(100):
        model.train()
        order = torch.randperm(len(y))
        for start in range(0,len(y),128):
            ix = order[start:start+128]
            c = str(rng.choice(conditions,p=[.6,.1,.1,.1,.1]))
            lp,gate = model(xs[c][ix],ms[c][ix],banks[c][ix])
            hard = nn.functional.nll_loss(lp,y[ix],weight=weights)
            loss = (1-config["kd"])*hard + config["kd"]*nn.functional.kl_div(lp,teacher[ix],reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(),dimensions=dims,config=config,full=bundle["full"],prior=bundle["prior"]),folder/"head.pt")
    p,statistics = {},{}
    donors = donor_indices(vd["source_groups"])
    for condition in CONDITIONS:
        part = projection.transform(modify(vd,condition,donors))
        p["specific_"+condition],statistics[condition] = dispatch(model,bundle,part)
    np.savez_compressed(folder/"predictions.npz",**p)
    save(folder/"dispatch.json",statistics)
    return p


def main(dataset: str) -> None:
    """正式测试仍封存；先完成固定MoE/蒸馏开发对照。"""
    folder = OUTPUT/"neural"/dataset
    if (folder/"selection_locked.json").exists():
        return
    base = json.loads((OUTPUT/"training"/dataset/"selection_locked.json").read_text("utf-8"))
    teacher = base["ordinary"]
    plan = dict(teacher=teacher,options=options(),epochs=100,seed=42,
        teacher_condition="source inner3 OOF complete fusion; scaler/PCA/heads all fit without held source",
        augmented="clean .6; missing A,V,T and text-only each .1; no synthetic mismatched hard labels",
        source_sha256=digest(Path(__file__)))
    lock = folder/"preregister.json"
    if lock.exists():
        if json.loads(lock.read_text("utf-8")) != plan:
            raise ValueError("MoE蒸馏协议变化")
    else:
        save(lock,plan)
    if dataset == "ch":
        data = ch_data()
        parts = [(take(data,tr),take(data,va),va) for tr,va in folds(data)]
        ids,y,groups = data["sample_ids"],data["labels"],data["source_groups"]
    else:
        td,vd = (read(OUTPUT/"datasets"/dataset/f"{s}.npz") for s in ("train","valid"))
        parts = [(td,vd,np.arange(len(vd["labels"])))]
        ids,y,groups = vd["sample_ids"],vd["labels"],vd["source_groups"]
    allp = {c["id"]:{"specific_"+s:np.full((len(y),3),np.nan) for s in CONDITIONS} for c in options()}
    started = time.monotonic()
    for f,(td,vd,ix) in enumerate(parts):
        ff = folder/f"fold{f}"
        ff.mkdir(parents=True,exist_ok=True)
        with (OUTPUT/"training"/dataset/f"fold{f}/projection.pkl").open("rb") as handle:
            projection = pickle.load(handle)
        with (OUTPUT/"training"/dataset/teacher["id"]/f"fold{f}/model.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        oof = crossfit(td,teacher,ff)
        for config in options():
            path = folder/config["id"]/f"fold{f}"
            if (path/"predictions.npz").exists():
                p = read(path/"predictions.npz")
            else:
                p = train_head(config,td,vd,projection,bundle,oof,path)
            for key in p:
                allp[config["id"]][key][ix] = p[key]
            print(json.dumps(dict(dataset=dataset,fold=f,candidate=config["id"],seconds=time.monotonic()-started)),flush=True)
    rows = []
    for config in options():
        p = allp[config["id"]]
        if not all(np.isfinite(v).all() and np.allclose(v.sum(1),1.,atol=1e-6) for v in p.values()):
            raise ValueError("MoE/学生覆盖或概率异常")
        np.savez_compressed(folder/config["id"]/"development_predictions.npz",sample_ids=ids,labels=y,source_groups=groups,**p)
        rows.append(dict(config=config,specific=metrics(y,p,"specific")))
    save(folder/"selection_locked.json",dict(rows=rows,teacher=teacher,development_only=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",choices=("ch","mosi","meld"))
    args = parser.parse_args()
    main(args.dataset)
