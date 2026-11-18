"""RoBERTa端到端文字适配、真实音视频特征融合及ECISA辅助训练。"""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from transformers import AutoModel,AutoTokenizer
from experiments.round2_moe_train import load_split,grouped_delta_ci
from experiments.round4_accuracy import merge
from experiments.round5_frozen_search import summary
from perception.knowledge_moe.features import FeatureStandardizer,save_json,sha256_file


def seed_all() -> None:
    """固定全部使用的随机源。"""
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(1)


class Rows(Dataset):
    """预先分词，逐样本携带配对音视频特征和掩码。"""
    def __init__(self,data: dict[str,np.ndarray],tokenizer: Any,scaler: FeatureStandardizer) -> None:
        self.tokens=tokenizer(data["raw_text"].astype(str).tolist(),truncation=True,max_length=128)["input_ids"]
        self.y=torch.tensor(data["labels"],dtype=torch.long)
        mask=data["modality_mask"][:,[2,1]] & (data["quality"][:,[2,1]]>0)
        values=scaler.transform({"visual":data["visual"],"audio":data["audio"]},mask)
        self.visual=torch.tensor(values["visual"]);self.audio=torch.tensor(values["audio"])
        self.quality=torch.tensor(data["quality"][:,[2,1]]*mask,dtype=torch.float32)

    def __len__(self) -> int:return len(self.y)

    def __getitem__(self,index: int) -> tuple:
        return self.tokens[index],self.visual[index],self.audio[index],self.quality[index],self.y[index]


def collate(rows: list[tuple]) -> dict[str,torch.Tensor]:
    """动态PAD；BERT中文词表PAD=0，在构建tokenizer时另行验证。"""
    width=max(len(row[0]) for row in rows)
    ids=torch.zeros((len(rows),width),dtype=torch.long);mask=torch.zeros_like(ids)
    for i,row in enumerate(rows):ids[i,:len(row[0])]=torch.tensor(row[0]);mask[i,:len(row[0])]=1
    return {"input_ids":ids,"attention_mask":mask,"visual":torch.stack([row[1] for row in rows]),
            "audio":torch.stack([row[2] for row in rows]),"quality":torch.stack([row[3] for row in rows]),"labels":torch.stack([row[4] for row in rows])}


class MultimodalBERT(nn.Module):
    """可训练语言表征与有掩码的视觉/音频投影；没有媒体时仍可用文字。"""
    def __init__(self,encoder: Path) -> None:
        super().__init__();self.encoder=AutoModel.from_pretrained(encoder,local_files_only=True)
        hidden=self.encoder.config.hidden_size
        self.text=nn.Sequential(nn.LayerNorm(hidden*2),nn.Linear(hidden*2,128),nn.GELU(),nn.Dropout(.25))
        self.visual=nn.Sequential(nn.Linear(512,64),nn.LayerNorm(64),nn.GELU())
        self.audio=nn.Sequential(nn.Linear(46,32),nn.LayerNorm(32),nn.GELU())
        self.head=nn.Sequential(nn.Linear(226,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,3))

    def forward(self,input_ids: torch.Tensor,attention_mask: torch.Tensor,visual: torch.Tensor,
                audio: torch.Tensor,quality: torch.Tensor) -> torch.Tensor:
        """缺模态投影在激活后归零，不能用线性偏置冒充真实观测。"""
        hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        weights=attention_mask.unsqueeze(-1)
        pooled=torch.cat([hidden[:,0],(hidden*weights).sum(1)/weights.sum(1).clamp_min(1)],dim=1)
        q=quality
        if self.training:q=q*(torch.rand_like(q)>.15)
        visual=torch.where(q[:,:1]>0,visual,torch.zeros_like(visual))
        audio=torch.where(q[:,1:]>0,audio,torch.zeros_like(audio))
        return self.head(torch.cat([self.text(pooled),self.visual(visual)*q[:,:1],self.audio(audio)*q[:,1:],q],dim=1))


def logits_for(model: MultimodalBERT,loader: DataLoader) -> np.ndarray:
    """冻结评估模式，不更新Batch或模型统计。"""
    model.eval();values=[]
    with torch.inference_mode():
        for row in loader:
            inputs={k:v.cuda() for k,v in row.items() if k!="labels"}
            with torch.autocast("cuda",dtype=torch.float16):values.append(model(**inputs).float().cpu().numpy())
    return np.concatenate(values)


def choose_bias(labels: np.ndarray,logits: np.ndarray) -> tuple[float,dict[str,Any]]:
    """开发数据内选中性偏置，允许相对自身原始accuracy最多下降2个百分点。"""
    raw=summary(labels,logits);choices=[]
    for bias in [-.5,-.25,0.,.25,.5,.75,1.]:
        adjusted=logits.copy();adjusted[:,1]+=bias;score=summary(labels,adjusted)
        if score["accuracy"]>=raw["accuracy"]-.02:choices.append((score["macro_f1"],score["accuracy"],-abs(bias),bias,score))
    best=max(choices,key=lambda row:row[:3]);return best[3],best[4]


def train(model: MultimodalBERT,rows: Rows,valid_rows: Rows,config: dict[str,Any],epochs: int,
          output: Path,prefix: str) -> tuple[dict,float,dict]:
    """实际应用类别加权损失，按固定开发集锁定checkpoint；不读取历史标签。"""
    loader=DataLoader(rows,batch_size=32,shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(42))
    valid_loader=DataLoader(valid_rows,batch_size=64,collate_fn=collate)
    counts=torch.bincount(rows.y,minlength=3).cuda().float()
    weights=(len(rows)/(3*counts.clamp_min(1))).pow(config["weight_power"]);weights/=weights.mean()
    optimizer=torch.optim.AdamW([{"params":model.encoder.parameters(),"lr":config["lr"]},
                                {"params":[p for name,p in model.named_parameters() if not name.startswith("encoder.")],"lr":config["lr"]*10}],weight_decay=.01)
    steps=max(1,len(loader)*epochs)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/max(1,int(steps*.1)),max(0,(steps-step)/max(1,int(steps*.9)))))
    amp=torch.amp.GradScaler("cuda");best=None;best_state=None;best_bias=0.;history=[];stale=0
    for epoch in range(1,epochs+1):
        model.train();total=0.
        for row in loader:
            inputs={k:v.cuda() for k,v in row.items()};labels=inputs.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                logits=model(**inputs)
                loss=nn.functional.cross_entropy(logits.float(),labels,weight=weights,label_smoothing=.03)
            amp.scale(loss).backward();amp.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.)
            previous_scale=amp.get_scale()
            amp.step(optimizer);amp.update()
            # 溢出时GradScaler会跳过optimizer.step，同步跳过学习率推进。
            if amp.get_scale()>=previous_scale:scheduler.step()
            total+=loss.item()*len(labels)
        values=logits_for(model,valid_loader);bias,score=choose_bias(valid_rows.y.numpy(),values)
        record={"epoch":epoch,"loss":total/len(rows),"validation":score,"bias":bias};history.append(record)
        print(json.dumps({"candidate":prefix,**record}),flush=True)
        key=(score["macro_f1"],score["accuracy"])
        if best is None or key>best:
            best=key;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_bias=bias;stale=0
        else:stale+=1
        save_json(output/(prefix+"_training.json"),history)
        if stale>=3:break
    model.load_state_dict(best_state)
    return best_state,best_bias,{"history":history,"best_macro_f1":best[0],"best_accuracy":best[1]}


def external_data(path: Path) -> dict[str,np.ndarray]:
    """读取明确标注的外部句子；媒体缺失保持掩码，不伪造配对。"""
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    n=len(rows)
    return {"raw_text":np.array([r["text"] for r in rows]),"labels":np.array([r["label"] for r in rows]),
            "sample_ids":np.array([r["id"] for r in rows]),"visual":np.zeros((n,512),np.float32),"audio":np.zeros((n,46),np.float32),
            "modality_mask":np.column_stack([np.ones(n,bool),np.zeros((n,2),bool)]),"quality":np.column_stack([np.ones(n,np.float32),np.zeros((n,2),np.float32)])}


def run(bundle: Path,encoder: Path,external: Path,previous: Path,output: Path) -> dict[str,Any]:
    """先跑全部固定训练，再锁定并读取历史测试与外部验证。"""
    output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():raise ValueError("结果已存在，拒绝覆盖")
    seed_all();start=time.monotonic()
    training=merge([load_split(bundle/(s+".npz")) for s in ["train","selection"]]);valid=load_split(bundle/"calibration.npz")
    if set(training["source_groups"]) & set(valid["source_groups"]):raise ValueError("来源泄漏")
    scaler=FeatureStandardizer().fit({"visual":training["visual"],"audio":training["audio"]},training["modality_mask"][:,[2,1]])
    tokenizer=AutoTokenizer.from_pretrained(encoder,local_files_only=True)
    if tokenizer.pad_token_id!=0:raise ValueError("动态padding实现要求pad_token_id=0")
    tr=Rows(training,tokenizer,scaler);va=Rows(valid,tokenizer,scaler)
    configs=[{"id":f"{stage}_{lr}_{power}","auxiliary":stage=="ecisa","lr":lr,"weight_power":power}
        for stage,lr,power in [("target",1e-5,.5),("target",2e-5,.5),("target",2e-5,1.),("ecisa",1e-5,.5),("ecisa",2e-5,.5)]]
    protocol={"seed":42,"configs":configs,"epochs_max":8,"patience":3,"batch":32,"max_length":128,"pooling":"CLS+masked mean",
        "training":"original train+selection 2876","selection":"original calibration703 repurposed for checkpoint and bias selection; no calibration claim",
        "historical_test":"original821 fixed, already examined in prior rounds; never used to select new checkpoints",
        "external":"ECISA original dev; not used for any selection; label mapping and overlap removal in external/manifest.json",
        "auxiliary":"one epoch ECISA official train only; checkpoint chosen by fixed schedule, not ECISA dev",
        "code_sha256":sha256_file(Path(__file__)),"source_manifest_sha256":sha256_file(bundle/"metadata.json"),
        "external_manifest_sha256":sha256_file(external/"manifest.json"),"baseline_macro_f1":.5657375344573995,"target_macro_f1":.6223112879031395}
    save_json(output/"preregister.json",protocol);save_json(output/"side_standardizer.json",scaler.parameters)
    results={};aux_state=None
    for config in configs:
        seed_all();model=MultimodalBERT(encoder).cuda()
        if config["auxiliary"]:
            if aux_state is None:
                aux=Rows(external_data(external/"train.jsonl"),tokenizer,scaler)
                # 固定一轮辅助训练，不用外部dev选择checkpoint；这里的日志只观察目标开发集。
                aux_state,_,_=train(model,aux,va,{"lr":2e-5,"weight_power":.5},1,output,"auxiliary_one_epoch")
            model.load_state_dict(aux_state)
        state,bias,record=train(model,tr,va,config,8,output,config["id"])
        torch.save(state,output/(config["id"]+".pt"));results[config["id"]]={**record,"bias":bias,"checkpoint_sha256":sha256_file(output/(config["id"]+".pt"))}
        np.savez_compressed(output/(config["id"]+"_validation.npz"),logits=logits_for(model,DataLoader(va,batch_size=64,collate_fn=collate)),labels=valid["labels"])
        del model,state;gc.collect();torch.cuda.empty_cache()
        save_json(output/"development_progress.json",results)
    winner=max(results,key=lambda key:(results[key]["best_macro_f1"],results[key]["best_accuracy"]))
    save_json(output/"selection_locked.json",{"winner":winner,"results":results})
    test=load_split(bundle/"test.npz");te=Rows(test,tokenizer,scaler)
    model=MultimodalBERT(encoder).cuda();model.load_state_dict(torch.load(output/(winner+".pt"),map_location="cpu",weights_only=True))
    scores=logits_for(model,DataLoader(te,batch_size=64,collate_fn=collate));scores[:,1]+=results[winner]["bias"]
    historical=summary(test["labels"],scores)
    with np.load(previous/"concat/test_predictions.npz",allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"],test["sample_ids"]):raise ValueError("历史ID错位")
        old=saved["logits"].argmax(1)
    ci=grouped_delta_ci(test["labels"],scores.argmax(1),old,test["source_groups"],3)
    np.savez_compressed(output/"historical_predictions.npz",logits=scores,labels=test["labels"],sample_ids=test["sample_ids"])
    external_valid=external_data(external/"external_valid.jsonl");ex=Rows(external_valid,tokenizer,scaler)
    ex_scores=logits_for(model,DataLoader(ex,batch_size=64,collate_fn=collate));ex_scores[:,1]+=results[winner]["bias"]
    external_score=summary(external_valid["labels"],ex_scores)
    np.savez_compressed(output/"external_predictions.npz",logits=ex_scores,labels=external_valid["labels"],sample_ids=external_valid["sample_ids"])
    report={"protocol":protocol,"winner":winner,"development":results,"historical":historical,"relative_f1_improvement":historical["macro_f1"]/.5657375344573995-1,
        "macro_f1_delta_ci95":ci,"external_ecisa":external_score,"seconds":time.monotonic()-start,"deployed":False,
        "limitations":["historical test already examined","ECISA is implicit text sentiment, not film multimodal sentiment or clinical outcomes",
                       "ECISA auxiliary train and its dev share dataset domain; external split is held out but not completely unseen domain for augmented model"]}
    save_json(output/"results.json",report);print(json.dumps({"winner":winner,"historical":historical,"relative":report["relative_f1_improvement"],"ci":ci,"external":external_score}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["bundle","encoder","external","previous","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.encoder,args.external,args.previous,args.output)
