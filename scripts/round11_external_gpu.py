"""外部真实媒体的冻结语音/图文表征，纯输入编码与目标标签严格分离。"""
from __future__ import annotations

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import torch
from experiments.round11_common import ROOT,OUTPUT,digest,save
from round11_resources import require_memory,available_commit_bytes
from round11_sensevoice import decode,build_model,encode as sense_encode,WEIGHT_SHA256

MEDIA=ROOT/"data/public/emotiontalk_media/clips"


def read_inputs(split: str) -> tuple[np.ndarray,list[str]]:
    """只返回ID和转录，不把标注文件中的目标传给编码器。"""
    rows=[json.loads(line) for line in (OUTPUT/"corpus"/f"{split}.jsonl").read_text("utf-8").splitlines()]
    return np.array([row["id"] for row in rows]),[row["text"] for row in rows]


def msp_model() -> tuple[Any,Any,dict[str,torch.Tensor]]:
    """使用与第十轮相同本地MSP权重与分类头。"""
    from safetensors import safe_open
    from transformers import AutoFeatureExtractor,Wav2Vec2Model
    path=Path(r"D:\ai\models\encoders\wav2vec2-emotion-msp")
    extractor=AutoFeatureExtractor.from_pretrained(path,local_files_only=True)
    model=Wav2Vec2Model.from_pretrained(path,local_files_only=True,attn_implementation="sdpa").cuda().eval()
    with safe_open(path/"model.safetensors",framework="pt") as weights:
        head={name:weights.get_tensor("classifier."+name).cuda() for name in
              ("dense.weight","dense.bias","out_proj.weight","out_proj.bias")}
    return model,extractor,head


def msp_encode(model: Any,extractor: Any,head: dict[str,torch.Tensor],waves: list[np.ndarray]) -> tuple[np.ndarray,np.ndarray]:
    """与冻结基线相同bf16编码、掩码均值/总体标准差与三维情感头。"""
    batch=extractor(waves,sampling_rate=16000,padding=True,return_attention_mask=True,return_tensors="pt").to("cuda")
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        hidden=model(**batch).last_hidden_state.float()
    mask=model._get_feature_vector_attention_mask(hidden.shape[1],batch["attention_mask"]).unsqueeze(-1)
    mean=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
    variance=((hidden-mean[:,None]).square()*mask).sum(1)/mask.sum(1).clamp_min(1)
    with torch.inference_mode():
        emotion=torch.nn.functional.linear(torch.tanh(torch.nn.functional.linear(mean,head["dense.weight"],head["dense.bias"])),
            head["out_proj.weight"],head["out_proj.bias"])
    return torch.cat((mean,variance.clamp_min(0).sqrt()),1).cpu().numpy(),emotion.cpu().numpy()


def main() -> None:
    """每个分量单独运行，缓存原子提交，完成后核对全量ID与有限性。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("kind",choices=("sense","msp","clip"))
    parser.add_argument("--splits",nargs="+",default=["train","valid","test_inputs"])
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if not (OUTPUT/"emotiontalk_acquisition/complete.json").exists():
        raise ValueError("媒体来源尚未验收")
    require_memory(6.)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    if args.kind=="sense":
        model,kwargs=build_model()
        weight_identity=WEIGHT_SHA256
        folder="sensevoice_deterministic_features"
        fields=("sensevoice","query","emotion_logprobs","transcript","available","original_duration")
    elif args.kind=="msp":
        model,extractor,head=msp_model()
        weight_identity=digest(Path(r"D:\ai\models\encoders\wav2vec2-emotion-msp\model.safetensors"))
        folder="speech_features"
        fields=("speech","speech_emotion","available","original_duration")
    else:
        from perception.knowledge_moe.features import FrozenCLIPEncoder
        from round11_external_signals import frames
        model=FrozenCLIPEncoder(Path(r"D:\ai\models\encoders\chinese-clip-vit-base-patch16"),"cuda")
        weight_identity=model.fingerprint
        folder="clip_features"
        fields=("visual","text","available","frame_count")
    output=OUTPUT/"emotiontalk_features"/folder
    cache=output/"samples"
    cache.mkdir(parents=True,exist_ok=True)
    binding=weight_identity+digest(Path(__file__))
    initial_code=OUTPUT/"source/external_gpu_initial.py"
    compatible=weight_identity+digest(initial_code)
    if args.kind=="sense":
        dependency=digest(Path(__file__).with_name("round11_sensevoice.py"))
        binding+=dependency
        compatible+=dependency
    if args.kind=="clip":
        dependency=digest(Path(__file__).with_name("round11_external_signals.py"))
        binding+=dependency
        compatible+=dependency
    allowed_bindings={binding,compatible}
    for intermediate in (OUTPUT/"source/external_gpu_gc.py",OUTPUT/"source/external_gpu_workers8.py"):
        if intermediate.exists():
            value=weight_identity+digest(intermediate)
            if args.kind=="sense":value+=digest(Path(__file__).with_name("round11_sensevoice.py"))
            if args.kind=="clip":value+=digest(Path(__file__).with_name("round11_external_signals.py"))
            allowed_bindings.add(value)
    summaries={}
    started=time.monotonic()
    for split in args.splits:
        identifiers,texts=read_inputs(split)
        destination=output/f"{split}.npz"
        if destination.exists():
            with np.load(destination,allow_pickle=False) as prior:
                if not np.array_equal(prior["sample_ids"],identifiers):
                    raise ValueError("既有外部编码ID不符")
            continue
        missing=[i for i,key in enumerate(identifiers) if not (cache/f"{key}.npz").exists()]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # 两批有界预取，把下一批的解码与当前GPU计算重叠；采样/数值规则不变。
            pending={}
            submitted=0
            def fill() -> None:
                nonlocal submitted
                if args.kind!="clip":return
                while submitted<len(missing) and len(pending)<2*args.batch:
                    i=missing[submitted];pending[i]=pool.submit(frames,MEDIA/f"{identifiers[i]}.mp4");submitted+=1
            fill()
            for start in range(0,len(missing),args.batch):
                if start%128==0 or available_commit_bytes()<2.5*1024**3:
                    before=available_commit_bytes()
                    gc.collect()
                    torch.cuda.empty_cache()
                    if before<2.5*1024**3:
                        print(json.dumps({"allocator_cleanup":True,"commit_before_gib":before/1024**3,
                            "commit_after_gib":available_commit_bytes()/1024**3,
                            "cuda_allocated_mib":torch.cuda.memory_allocated()/1024**2,
                            "cuda_reserved_mib":torch.cuda.memory_reserved()/1024**2}),flush=True)
                require_memory(1.5)
                ix=missing[start:start+args.batch]
                payloads=[]
                if args.kind in ("sense","msp"):
                    decoded=list(pool.map(decode,[MEDIA/f"{identifiers[i]}.mp4" for i in ix]))
                    active=[j for j,row in enumerate(decoded) if row[2]]
                    lookup={index:j for j,index in enumerate(active)}
                    if active and args.kind=="sense":
                        pooled,query,emo,transcripts=sense_encode(model,kwargs,[decoded[j][0] for j in active])
                    elif active:
                        pooled,emo=msp_encode(model,extractor,head,[decoded[j][0] for j in active])
                    for j in range(len(ix)):
                        k=lookup.get(j)
                        row={"available":np.array(k is not None),"original_duration":np.array(decoded[j][1])}
                        if args.kind=="sense":
                            row.update(sensevoice=pooled[k] if k is not None else np.zeros(1024,np.float32),
                                query=query[k] if k is not None else np.zeros((4,512),np.float32),
                                emotion_logprobs=emo[k] if k is not None else np.zeros((4,5),np.float32),
                                transcript=np.array(transcripts[k] if k is not None else ""))
                        else:
                            row.update(speech=pooled[k] if k is not None else np.zeros(2048,np.float32),
                                speech_emotion=emo[k] if k is not None else np.zeros(3,np.float32))
                        payloads.append(row)
                else:
                    decoded=[pending.pop(i).result() for i in ix]
                    fill()
                    images=[image for row in decoded for image in row[0]]
                    repeated=[texts[i] for i,row in zip(ix,decoded) for _ in row[0]]
                    values=model.encode(images,repeated) if images else None
                    cursor=0
                    for j,i in enumerate(ix):
                        count=len(decoded[j][0])
                        row={"visual":values["visual"][cursor:cursor+count].mean(0) if count else np.zeros(512,np.float32),
                             "text":values["text"][cursor] if count else model.encode_text([texts[i]])[0],
                             "available":np.array(count>0),"frame_count":np.array(count)}
                        payloads.append(row)
                        cursor+=count
                for i,row in zip(ix,payloads):
                    for value in row.values():
                        if value.dtype.kind in "fc" and not np.isfinite(value).all():
                            raise ValueError("编码器输出非有限值")
                    target=cache/f"{identifiers[i]}.npz"
                    temporary=target.with_suffix(".tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle,**row,sample_id=identifiers[i],binding=np.array(binding))
                    temporary.replace(target)
                if start%128==0 or start+args.batch>=len(missing):
                    print(json.dumps({"kind":args.kind,"split":split,"encoded":min(len(missing),start+args.batch),
                        "missing_at_start":len(missing),"seconds":round(time.monotonic()-started,1)}),flush=True)
        records=[]
        for key in identifiers:
            with np.load(cache/f"{key}.npz",allow_pickle=False) as saved:
                if str(saved["binding"])not in allowed_bindings or str(saved["sample_id"])!=key:
                    raise ValueError("外部逐样本缓存绑定不符")
                records.append({name:saved[name] for name in fields})
        np.savez_compressed(destination,sample_ids=identifiers,
            **{name:np.stack([row[name] for row in records]) for name in fields})
        summaries[split]={"n":len(identifiers),"available":sum(bool(row["available"]) for row in records),
                          "sha256":digest(destination)}
    save(output/"manifest.json",{"binding":binding,"compatible_initial_binding":compatible,
        "execution_change":"periodic allocator cleanup and configurable bounded decode workers;no numerical algorithm change",
        "accepted_source_bindings":sorted(allowed_bindings),"decode_workers":args.workers,
        "model":weight_identity,"kind":args.kind,
        "splits_completed_this_run":summaries,"target_labels_used":False,"code_sha256":digest(Path(__file__)),
        "speech_max_contiguous_seconds":10 if args.kind!="clip" else None,
        "sense_frontend_dither":0. if args.kind=="sense" else None,
        "MSP_precision":"bf16 encoder,float32 pooling" if args.kind=="msp" else None,
        "representation":"frozen upstream model;native tags and ASR are predictions,not human gold"},immutable=False)


if __name__=="__main__":
    main()
