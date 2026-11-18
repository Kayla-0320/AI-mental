"""从本地开放预训练文字编码器提取新表征；不使用目标标签训练编码器。"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from perception.knowledge_moe.features import save_json,sha256_file


def encode(texts: list[str], path: Path, batch_size: int=64) -> tuple[np.ndarray,np.ndarray]:
    """提取CLS与掩码均值，屏蔽PAD；每条原文单独编码，无跨样本上下文。"""
    torch.manual_seed(42);torch.set_num_threads(1)
    device="cuda" if torch.cuda.is_available() else "cpu"
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModel.from_pretrained(path,local_files_only=True).to(device).eval()
    cls_values,mean_values=[],[]
    with torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            batch=tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=128,return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device=="cuda"):
                hidden=model(**batch).last_hidden_state.float()
            weights=batch["attention_mask"].unsqueeze(-1)
            cls_values.append(hidden[:,0].cpu().numpy())
            mean_values.append(((hidden*weights).sum(1)/weights.sum(1).clamp_min(1)).cpu().numpy())
    del model;gc.collect()
    if device=="cuda":torch.cuda.empty_cache()
    return np.concatenate(cls_values),np.concatenate(mean_values)


def run(bundle: Path, output: Path, encoders: Path) -> None:
    """编码冻结划分及可选外部文本，保存编码器身份与原始ID。"""
    output.mkdir(parents=True,exist_ok=True)
    parts={}
    for split in ["train","selection","calibration","test"]:
        with np.load(bundle/(split+".npz"),allow_pickle=False) as saved:parts[split]={k:saved[k] for k in saved.files}
    sizes={key:len(value["raw_text"]) for key,value in parts.items()}
    texts=[str(text) for part in parts.values() for text in part["raw_text"]]
    fingerprint={};start=time.monotonic()
    for name,folder in [("roberta","chinese-roberta-wwm-ext"),("bge","bge-base-zh-v1.5")]:
        path=encoders/folder
        weights=sorted([*path.glob("*.safetensors"),*path.glob("pytorch_model*.bin")])
        identity={p.name:sha256_file(p) for p in [path/"config.json",*weights]}
        fingerprint[name]=identity
        cache=output/(name+"_all.npz")
        if cache.exists():
            with np.load(cache,allow_pickle=False) as saved:
                if str(saved["input_digest"])!=hashlib.sha256('\n'.join(texts).encode()).hexdigest():raise ValueError("文本顺序不符")
                if str(saved["model_digest"])!=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest():raise ValueError("编码器摘要不符")
                cls,mean=saved["cls"],saved["mean"]
        else:
            cls,mean=encode(texts,path)
            np.savez_compressed(cache,cls=cls,mean=mean,input_digest=np.array(hashlib.sha256('\n'.join(texts).encode()).hexdigest()),
                                model_digest=np.array(hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()))
        cursor=0
        for split,part in parts.items():
            n=sizes[split];part[name+"_cls"]=cls[cursor:cursor+n];part[name+"_mean"]=mean[cursor:cursor+n];cursor+=n
        print(json.dumps({"encoder":name,"n":len(texts),"seconds":time.monotonic()-start}),flush=True)
    for split,part in parts.items():np.savez_compressed(output/(split+".npz"),**part)
    save_json(output/"manifest.json",{"seed":42,"input_manifest_sha256":sha256_file(bundle/"metadata.json"),"encoders":fingerprint,
              "sizes":sizes,"training":"none_frozen_pretrained_encoders","max_length":128,"pooling":["CLS","attention_mask_mean"],
              "test_labels_used_for_encoding":False,"code_sha256":sha256_file(Path(__file__))})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["bundle","output","encoders"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.output,args.encoders)
