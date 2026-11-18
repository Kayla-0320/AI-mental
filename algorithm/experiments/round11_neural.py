"""强化表征上的模态监督、注意力、同池路由与软教师交换对照。"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import time
from typing import Any

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"

import numpy as np
import torch
from torch import nn,Tensor
from experiments.round10_evidence import supervision
from experiments.round10_fusion import folds,take
from experiments.round10_domain import bias_select
from experiments.round11_features import TEXT,VISUAL,AUDIO_PACKS,load_data,transform,observed
from experiments.round11_common import OUTPUT,digest,save

NAMES=("T","A","V")
FIELDS={"T":TEXT,"A":AUDIO_PACKS["both"],"V":VISUAL}


class RoutedHead(nn.Module):
    """共享三路表征；四个整体M专家与单模态真值辅助头职责分开。"""

    def __init__(self,dims: dict[str,int],mode: str,hidden: int = 64) -> None:
        super().__init__()
        self.mode=mode
        self.projections=nn.ModuleDict({name:nn.Sequential(nn.Linear(dim,hidden),nn.LayerNorm(hidden),
            nn.GELU(),nn.Dropout(.1)) for name,dim in dims.items()})
        self.single=nn.ModuleList([nn.Linear(hidden,3) for _ in NAMES])
        self.auxiliary=nn.ModuleList([nn.Linear(hidden,3) for _ in NAMES])
        self.intensity=nn.ModuleList([nn.Linear(hidden,1) for _ in NAMES])
        self.joint_attention=nn.MultiheadAttention(hidden,4,dropout=.1,batch_first=True)
        self.joint_norm=nn.LayerNorm(hidden)
        self.joint_classifier=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,3))
        self.concat_classifier=nn.Sequential(nn.Linear(3*hidden,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,3))
        self.router=nn.Sequential(nn.Linear(3*hidden+6,64),nn.GELU(),nn.Linear(64,4))
        self.global_weights=nn.Parameter(torch.zeros(4))

    def joint(self,h: Tensor,mask: Tensor) -> Tensor:
        """注意力只消费可见模态；缺失位置不参与聚合。"""
        attended,_=self.joint_attention(h,h,h,key_padding_mask=~mask,need_weights=False)
        values=self.joint_norm(h+attended)
        return (values*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)

    def forward(self,x: dict[str,Tensor],mask: Tensor,quality: Tensor,
                auxiliary: bool = True) -> tuple[Tensor,Tensor,list[Tensor],list[Tensor]]:
        """Top-2仅调用选中样本对应的M专家；门控仍需要全部可见模态的投影。"""
        if not mask.any(1).all():
            raise ValueError("没有任何可用模态")
        encoded=[self.projections[name](torch.where(mask[:,j:j+1],x[name],0.))*mask[:,j:j+1]
                 for j,name in enumerate(NAMES)]
        h=torch.stack(encoded,1)
        aux=[head(encoded[j]) for j,head in enumerate(self.auxiliary)] if auxiliary else []
        regressions=[head(encoded[j]).squeeze(1).tanh() for j,head in enumerate(self.intensity)] if auxiliary else []
        if self.mode in ("concat","attention"):
            logits=self.concat_classifier(h.flatten(1)) if self.mode=="concat" else self.joint_classifier(self.joint(h,mask))
            gate=torch.zeros(len(mask),4,device=h.device)
            gate[:,3]=1.
            return logits.log_softmax(1),gate,aux,regressions
        eligible=torch.column_stack((mask,mask.any(1)))
        if self.mode=="uniform":
            scores=torch.zeros_like(eligible,dtype=h.dtype)
        elif self.mode=="global":
            scores=self.global_weights.expand(len(mask),-1)
        else:
            scores=self.router(torch.cat((h.flatten(1),mask.to(h.dtype),quality),1))
        scores=scores.masked_fill(~eligible,-torch.inf)
        if self.mode=="top2":
            chosen=torch.zeros_like(eligible).scatter_(1,scores.topk(2,1).indices,True)
            scores=scores.masked_fill(~chosen,-torch.inf)
        gate=scores.softmax(1)
        probability=torch.zeros(len(mask),3,device=h.device)
        for j in range(4):
            rows=torch.where(gate[:,j]>0)[0]
            if not len(rows):
                continue
            logits=self.single[j](h[rows,j]) if j<3 else self.joint_classifier(self.joint(h[rows],mask[rows]))
            probability=probability.index_add(0,rows,gate[rows,j:j+1]*logits.softmax(1))
        return probability.clamp_min(1e-9).log(),gate,aux,regressions


def teacher(model: RoutedHead,x: dict[str,Tensor],mask: Tensor,quality: Tensor) -> tuple[Tensor,Tensor]:
    """教师关闭dropout并停止梯度，调用后恢复学生先前的训练状态。"""
    training=model.training
    model.eval()
    try:
        with torch.no_grad():
            logits,gate,_,_=model(x,mask,quality,auxiliary=False)
        return logits.exp().detach(),gate.detach()
    finally:
        model.train(training)


def soft_swap_target(model: RoutedHead,x: dict[str,Tensor],mask: Tensor,quality: Tensor,
                     donors: Tensor,kind: str,rows: Tensor | None = None) -> Tensor:
    """交换没有整体硬金标；门控暴露权重仅构造自蒸馏代理，不声称因果识别。"""
    rows=torch.arange(len(mask)) if rows is None else rows
    if not (mask[rows,1]&mask[donors[rows],1]).all():
        raise ValueError("交换教师只能消费两端都可见的真实音频")
    source,gate=teacher(model,{name:value[rows] for name,value in x.items()},mask[rows],quality[rows])
    audio_mask=torch.zeros_like(mask[rows])
    audio_mask[:,1]=True
    donor,_=teacher(model,{name:value[donors[rows]] for name,value in x.items()},audio_mask,quality[donors[rows]])
    # 两个匹配交换臂都执行相同的两次教师前向；一致性臂只不采用供体分布。
    if kind=="consistency":
        return source
    exposure=(gate[:,1]+gate[:,3]/mask[rows].sum(1).clamp_min(1)).clamp(0.,1.).detach()
    return ((1-exposure[:,None])*source+exposure[:,None]*donor).detach()


def paired_swap(x: dict[str,Tensor],mask: Tensor,quality: Tensor,donors: Tensor) -> tuple[dict[str,Tensor],Tensor,Tensor,Tensor]:
    """交换A时可用性与质量一起跟随供体，只有两端均可见的非自身配对才训练。"""
    changed={name:value for name,value in x.items()}
    changed["A"]=x["A"][donors]
    changed_mask=mask.clone()
    changed_quality=quality.clone()
    changed_mask[:,1]=mask[donors,1]
    changed_quality[:,1]=quality[donors,1]
    valid=mask[:,1]&mask[donors,1]&(donors!=torch.arange(len(mask),device=mask.device))
    return changed,changed_mask,changed_quality,valid


def prepared(data: dict[str,np.ndarray],scalers: dict[str,Any] | None = None
             ) -> tuple[dict[str,Tensor],Tensor,Tensor,dict[str,Any]]:
    """仅训练来源拟合每路标准化/PCA；不输入来源ID或目标标注。"""
    fitting=scalers is None
    scalers={} if fitting else scalers
    x={}
    for name in NAMES:
        values,fitted=transform(data,FIELDS[name],None if fitting else scalers[name])
        scalers[name]=fitted
        x[name]=torch.tensor(values,dtype=torch.float32)
    mask=np.column_stack((data["modality_mask"][:,0],observed(data,"audio"),
        observed(data,"visual")|observed(data,"face_summary")))
    return x,torch.tensor(mask),torch.tensor(data["quality"][:,[0,1,2]],dtype=torch.float32),scalers


def options() -> list[dict[str,Any]]:
    """有限、预注册的注意力/路由/匹配交换对照，共享所有原始监督及优化预算。"""
    configurations=[{"mode":mode,"swap":"none"} for mode in ("concat","attention","uniform","global","dynamic","top2")]
    configurations.extend(({"mode":"dynamic","swap":"consistency"},
                           {"mode":"dynamic","swap":"soft"},{"mode":"top2","swap":"soft"}))
    return [{**value,"id":f"neural_{i}","hidden":64,"lr":.001,"decay":.01,
             "auxiliary_weight":.25,"regression_weight":.1,"swap_weight":.1} for i,value in enumerate(configurations)]


def train(config: dict[str,Any],data: dict[str,np.ndarray],valid: dict[str,np.ndarray],
          epochs: tuple[int,...],directory: Path,fold: int) -> dict[int,np.ndarray]:
    """固定轮次输出全部检查点；外层标签只用于之后的开发选型。"""
    torch.manual_seed(42)
    torch.set_num_threads(2)
    x,mask,q,scalers=prepared(data)
    vx,vm,vq,_=prepared(valid,scalers)
    classes,continuous=supervision(data)
    labels=torch.tensor(classes[:,0],dtype=torch.long)
    uni=torch.tensor(classes[:,1:],dtype=torch.long)
    intensity=torch.tensor(continuous[:,1:],dtype=torch.float32)
    class_weights=[]
    for j in range(4):
        visible=torch.ones(len(labels),dtype=torch.bool) if j==0 else mask[:,j-1]
        y=labels if j==0 else uni[:,j-1]
        count=torch.bincount(y[visible],minlength=3).float().clamp_min(1)
        class_weights.append((visible.sum()/(3*count)).sqrt())
    model=RoutedHead({name:values.shape[1] for name,values in x.items()},config["mode"],config["hidden"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["decay"])
    outputs={}
    directory.mkdir(parents=True,exist_ok=True)
    pair_count=0
    for epoch in range(1,max(epochs)+1):
        model.train()
        order=torch.randperm(len(labels))
        for start in range(0,len(labels),128):
            ix=order[start:start+128]
            batch={name:value[ix] for name,value in x.items()}
            visible=mask[ix].clone()
            visible[:,1:]&=torch.rand(len(ix),2)>=.1
            quality=q[ix].clone()*visible
            logp,gate,aux,regressions=model(batch,visible,quality)
            loss=nn.functional.nll_loss(logp,labels[ix],weight=class_weights[0])
            for j in range(3):
                active=visible[:,j]
                if active.any():
                    loss+=config["auxiliary_weight"]/3*nn.functional.cross_entropy(aux[j][active],uni[ix,j][active],weight=class_weights[j+1])
                    loss+=config["regression_weight"]/3*nn.functional.smooth_l1_loss(regressions[j][active],intensity[ix,j][active])
            if config["mode"] not in ("concat","attention","uniform","global"):
                loss+=.01*((gate.mean(0)-.25).square().sum())
            if config["swap"]!="none":
                donors=torch.randperm(len(ix))
                changed,changed_mask,changed_quality,active=paired_swap(batch,visible,quality,donors)
                if active.any():
                    # 非音频可见的供体不会被用于教师；先缩至有效配对，避免全缺失。
                    rows=torch.where(active)[0]
                    target=soft_swap_target(model,batch,visible,quality,donors,config["swap"],rows)
                    swapped=model({name:value[rows] for name,value in changed.items()},changed_mask[rows],changed_quality[rows],auxiliary=False)[0]
                    loss+=config["swap_weight"]*nn.functional.kl_div(swapped,target,reduction="batchmean")
                    pair_count+=len(rows)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        if epoch in epochs:
            model.eval()
            with torch.inference_mode():
                outputs[epoch]=model(vx,vm,vq,auxiliary=False)[0].exp().numpy()
            torch.save({"state_dict":model.state_dict(),"dims":{name:values.shape[1] for name,values in x.items()},
                "scalers":scalers,"config":config,"epoch":epoch,"fold":fold,
                "train_groups":np.unique(data["source_groups"]),"valid_groups":np.unique(valid["source_groups"]),
                "training_sample_ids":data["sample_ids"],"valid_sample_ids":valid["sample_ids"],
                "teacher_proxy_not_causal_identification":True,"valid_swap_pairs":pair_count},directory/f"epoch{epoch}.pt")
    return outputs


def main() -> None:
    """一次固定搜索，允许按配置继续；测试与原calibration703不参与。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=int,nargs="+")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    data=load_data("pts")
    grid=options()
    epochs=(1,) if args.smoke else (10,20,40)
    output=OUTPUT/("neural_smoke" if args.smoke else "neural")
    plan={"configs":grid,"epochs":list(epochs),"source_folds":4,"seed":42,
        "input":"same frozen strongT/A/V representations;pts quality;source-fitted preprocessing",
        "supervision":"M3 primary;official distinctT/A/V3 plus continuous auxiliary;no synthetic hardM label",
        "fairness":"all methods same original examples,losses,epochs,dropout;consistency and soft-swap share swapped examples",
        "swap_teacher":"eval,no_grad;exposureproxy=gateA+gateJoint/available_count;not causally identified target",
        "top2":"actual M-expert dispatch;all modality projections/foundation extraction still required",
        "calibration703_or_history821_used":False,"code_sha256":digest(Path(__file__))}
    lock=output/"preregister.json"
    if lock.exists():
        if json.loads(lock.read_text("utf-8"))!=plan:
            raise ValueError("神经对照预注册不一致")
    else:
        save(lock,plan)
    indices=args.config if args.config is not None else range(len(grid))
    started=time.monotonic()
    for i in indices:
        config=grid[i]
        folder=output/config["id"]
        if (folder/"complete.json").exists():
            continue
        p={epoch:np.full((len(data["labels"]),3),np.nan) for epoch in epochs}
        for f,(tr,va) in enumerate(folds(data)):
            cache=folder/f"fold{f}/predictions.npz"
            if cache.exists():
                with np.load(cache,allow_pickle=False) as source:
                    for epoch in epochs:p[epoch][va]=source[f"epoch{epoch}"]
                continue
            result=train(config,take(data,tr),take(data,va),epochs,folder/f"fold{f}",f)
            for epoch,value in result.items():p[epoch][va]=value
            np.savez_compressed(cache,valid_sample_ids=data["sample_ids"][va],**{f"epoch{k}":v for k,v in result.items()})
            print(json.dumps({"config":i,"mode":config["mode"],"swap":config["swap"],"fold":f,
                              "seconds":round(time.monotonic()-started,1)}),flush=True)
        rows=[{"config":config,"epoch":epoch,**bias_select(data,np.log(np.maximum(value,1e-12)))} for epoch,value in p.items()]
        save(folder/"complete.json",{"results":rows,"development_only":True})
        np.savez_compressed(folder/"oof.npz",sample_ids=data["sample_ids"],labels=data["labels"],
            source_groups=data["source_groups"],**{f"epoch{k}":v for k,v in p.items()})
        print(json.dumps({"completed":i,"best_oof":max(row["metrics"]["macro_f1"] for row in rows)}),flush=True)


if __name__=="__main__":
    main()
