"""原始媒体单请求轻量部署基准，保留完整可调用推理入口。"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
import torch
import cv2
from transformers import AutoModel,AutoTokenizer
from experiments.round11_common import ROOT,OUTPUT,PREVIOUS,digest,save
from experiments.round11_features import alignment_feature
from experiments.round11_final import predict
from perception.knowledge_moe.features import FrozenCLIPEncoder
from round11_sensevoice import build_model,decode,encode
from round11_external_gpu import msp_model,msp_encode
from round11_external_signals import prepare,frames
from round11_external_faces import encode as face_encode


class RawRuntime:
    """常驻轻量版：只去除生成式语言模型，其余输入来自真实媒体。"""
    def __init__(self) -> None:
        torch.manual_seed(42);torch.set_num_threads(1);cv2.setNumThreads(1)
        with (OUTPUT/"final_models/light/model.pkl").open("rb") as handle:self.bundle=pickle.load(handle)
        self.sense,self.sense_kwargs=build_model();gc.collect();torch.cuda.empty_cache()
        self.msp,self.extractor,self.head=msp_model();gc.collect();torch.cuda.empty_cache()
        self.clip=FrozenCLIPEncoder(Path(r"D:\ai\models\encoders\chinese-clip-vit-base-patch16"),"cuda")
        base=Path(r"D:\ai\models\encoders")
        self.bge_tokenizer=AutoTokenizer.from_pretrained(base/"bge-base-zh-v1.5",local_files_only=True)
        self.bge=AutoModel.from_pretrained(base/"bge-base-zh-v1.5",local_files_only=True).cuda().eval()
        self.aux_tokenizer=AutoTokenizer.from_pretrained(base/"chinese-roberta-wwm-ext",local_files_only=True)
        aux=AutoModel.from_pretrained(base/"chinese-roberta-wwm-ext",local_files_only=True)
        weights=torch.load(PREVIOUS/"text/auxiliary_only.pt",map_location="cpu",weights_only=True)
        aux.load_state_dict({k.removeprefix("encoder."):v for k,v in weights.items() if k.startswith("encoder.")},strict=True)
        self.aux=aux.cuda().eval();del weights;gc.collect();torch.cuda.empty_cache()

    def infer(self,path: Path,text: str) -> tuple[np.ndarray,dict[str,float],dict[str,np.ndarray]]:
        """一个请求从原始MP4/已知转录到五种子概率，计时包含解码与所有编码。"""
        if not text.strip():raise ValueError("当前基准入口需要非空转录")
        start=time.perf_counter();timings={}
        def stamp(name: str,began: float) -> float:
            torch.cuda.synchronize();now=time.perf_counter();timings[name]=now-began;return now
        t=time.perf_counter();signal=prepare(path);face=face_encode(path);images,_,_=frames(path);wave,_,available=decode(path)
        if not available or not images:raise ValueError("当前原媒体基准需要可解码音轨和画面")
        t=stamp("decode_MFCC_quality_face",t)
        clip=self.clip.encode(images,[text]*len(images));t=stamp("clip",t)
        sense,query,emo,asr=encode(self.sense,self.sense_kwargs,[wave]);t=stamp("sensevoice",t)
        speech,emotion=msp_encode(self.msp,self.extractor,self.head,[wave]);t=stamp("msp",t)
        with torch.inference_mode():
            row=self.bge_tokenizer([text],padding=True,truncation=True,max_length=128,return_tensors="pt").to("cuda")
            with torch.autocast("cuda",dtype=torch.float16):hidden=self.bge(**row).last_hidden_state.float()
            bge=hidden[:,0].cpu().numpy()
            row=self.aux_tokenizer([text],padding=True,truncation=True,max_length=96,return_tensors="pt").to("cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):hidden=self.aux(**row).last_hidden_state.float()
            mask=row["attention_mask"].unsqueeze(-1);aux=((hidden*mask).sum(1)/mask.sum(1)).cpu().numpy()
        t=stamp("text_encoders",t)
        data={"sample_ids":np.array([path.stem]),"raw_text":np.array([text]),"source_groups":np.array(["runtime"]),
            "audio":signal["audio"][None],"modality_mask":signal["modality_mask"][None],"quality":signal["quality"][None],
            "face_summary":face["face_summary"][None],"deep_face":face["deep_face"][None],"coverage":np.array([float(face["coverage"])]),
            "visual":clip["visual"].mean(0)[None],"bge_cls":bge,"aux_mean":aux,
            "sensevoice":sense,"sensevoice_query":query[:,1,:],"sensevoice_emotion":emo[:,1,:],"sensevoice_available":np.array([available]),
            "speech":speech,"speech_emotion":emotion,"speech_available":np.array([available]),
            "asr_alignment":alignment_feature(text,asr[0])[None]}
        p=predict(self.bundle,data);stamp("fusion",t);timings["total"]=time.perf_counter()-start
        return p,timings,data


def main() -> None:
    """固定12个开发片段、每条两次实测；热不是缓存分类头。"""
    parser=argparse.ArgumentParser();parser.add_argument("--samples",type=int,default=12);args=parser.parse_args()
    folder=OUTPUT/"runtime_benchmark";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"results.json").exists():return
    from experiments.round11_features import load_data
    data=load_data("pts");order=sorted(range(len(data["sample_ids"])),key=lambda i:hashlib.sha256(str(data["sample_ids"][i]).encode()).hexdigest())[:args.samples]
    plan={"mode":"light","n_media":len(order),"repetitions":2,"request_batch":1,"IDs":data["sample_ids"][order].tolist(),
        "condition":"official transcript+raw MP4;all decoders/frozen encoders/fusion run each request;no feature cache reads",
        "startup":"one model initialization;OS disk cache not flushed;not a production cold-start distribution",
        "code_sha256":digest(Path(__file__)),"model_sha256":digest(OUTPUT/"final_models/light/model.pkl")}
    pre=folder/"preregister.json"
    if pre.exists():
        if json.loads(pre.read_text("utf-8"))!=plan:raise ValueError("运行时基准预注册不符")
    else:save(pre,plan)
    # 释放大开发特征，只保留原文与ID；避免误用缓存做推理。
    jobs=[(str(data["sample_ids"][i]),str(data["raw_text"][i])) for i in order];del data;gc.collect()
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();runtime=RawRuntime();torch.cuda.synchronize();startup=time.perf_counter()-start
    rows=[];probabilities=[]
    for repeat in range(2):
        for key,text in jobs:
            p,t,_=runtime.infer(ROOT/f"data/public/ch_sims_v2/supervised/{key}.mp4",text)
            rows.append({"id":key,"repeat":repeat,**t});probabilities.append(p)
            print(json.dumps(rows[-1]),flush=True)
            gc.collect();torch.cuda.empty_cache()
    values=np.array([r["total"] for r in rows]);hot=values[1:]
    raw=np.stack(probabilities);delta=float(np.abs(raw[:len(jobs)]-raw[len(jobs):]).max())
    result={"mode":"light","n_requests":len(rows),"model_startup_seconds":startup,"first_inference_seconds":float(values[0]),
        "model_start_plus_first_inference_seconds":startup+float(values[0]),"hot_p50_seconds":float(np.quantile(hot,.5)),"hot_p95_seconds":float(np.quantile(hot,.95)),
        "cuda_peak_allocated_MiB":torch.cuda.max_memory_allocated()/1024**2,"cuda_peak_reserved_MiB":torch.cuda.max_memory_reserved()/1024**2,
        "repeated_request_probability_max_abs_error":delta,"rows":rows,
        "limits":"single laptop;12 clips/24 requests;shared desktop load;warm OS cache;not network/server/ASR transcription latency;7B/3B raw latency not measured"}
    np.savez_compressed(folder/"predictions.npz",sample_ids=np.array([r["id"] for r in rows]),
        probabilities=np.concatenate(probabilities,axis=1),seeds=np.arange(42,47))
    save(folder/"results.json",result);print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)


if __name__=="__main__":main()
