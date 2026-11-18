"""真实原片上的自动质量处理对照；阈值只用开发样本选择。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import av
import cv2
import librosa
import numpy as np
from PIL import Image
import torch

from data_quality.raw_signals import assess_audio_signal, assess_visual_signal
from experiments.round2_moe_train import forward, load_split, subset, grouped_delta_ci
from experiments.round4_accuracy import summarize
from perception.knowledge_moe.features import FrozenCLIPEncoder, FeatureStandardizer, save_json, sha256_file
from perception.knowledge_moe.model import ConcatenationHead

SCENARIOS=["clean","silence","clipping","white_noise","black_video","blur_video"]


def sample_indices(data: dict[str,np.ndarray], count: int) -> np.ndarray:
    """按ID摘要选样，不根据标签或预测挑容易样本。"""
    order=sorted(range(len(data["labels"])),key=lambda i:hashlib.sha256(("42|"+str(data["sample_ids"][i])).encode()).digest())
    return np.array(order[:count])


def read_clip(path: Path) -> tuple[np.ndarray,list[Image.Image]]:
    """读取真实音轨及三帧；保持与第三轮特征提取约定一致。"""
    parts=[]
    with av.open(str(path)) as container:
        sampler=av.AudioResampler(format="fltp",layout="mono",rate=16000)
        if container.streams.audio:
            for frame in container.decode(audio=0):
                parts.extend(f.to_ndarray().reshape(-1) for f in sampler.resample(frame))
            parts.extend(f.to_ndarray().reshape(-1) for f in sampler.resample(None))
    audio=np.clip(np.concatenate(parts),-1,1).astype(np.float32) if parts else np.empty(0,np.float32)
    cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images=[]
    for fraction in [.2,.5,.8]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,min(count-1,round(count*fraction))))
        ok,frame=cap.read()
        if ok: images.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    cap.release()
    if not images:
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            duration=float(stream.duration*stream.time_base)
            targets=np.array([.2,.5,.8])*duration; distances=np.full(3,np.inf); selected=[None]*3
            for frame in container.decode(video=0):
                position=float(frame.time or 0)
                for j,target in enumerate(targets):
                    if abs(position-target)<distances[j]:
                        distances[j]=abs(position-target);selected[j]=frame.to_image()
            images=[image for image in selected if image is not None]
    if not images or len(audio)<512: raise ValueError("原片缺少有效音视频，不静默删样本")
    return audio,images


def audio_features(signal: np.ndarray) -> np.ndarray:
    """真实故障波形重新提取同版本46维特征，不对缓存向量伪造波形故障。"""
    signal=np.asarray(signal,dtype=np.float32)
    mfcc=librosa.feature.mfcc(y=signal,sr=16000,n_mfcc=20,n_fft=512,hop_length=160)
    rms=librosa.feature.rms(y=signal,frame_length=512,hop_length=160)[0]
    zcr=librosa.feature.zero_crossing_rate(signal,frame_length=512,hop_length=160)[0]
    centroid=librosa.feature.spectral_centroid(y=signal,sr=16000,n_fft=512,hop_length=160)[0]/8000
    return np.concatenate([mfcc.mean(1),mfcc.std(1),[rms.mean(),rms.std(),zcr.mean(),zcr.std(),centroid.mean(),centroid.std()]]).astype(np.float32)


def prepare(data: dict[str,np.ndarray], media: Path, cache: Path, encoder: FrozenCLIPEncoder) -> dict[str,Any]:
    """缓存六种条件的真实特征与检测结果；缓存绑定媒体与编码器摘要。"""
    cache.mkdir(parents=True,exist_ok=True)
    collected={scenario:{"audio":[],"visual":[],"quality":[],"usable":[]} for scenario in SCENARIOS}
    for i,identifier in enumerate(data["sample_ids"]):
        path=media/(str(identifier)+".mp4");target=cache/(str(identifier)+".npz")
        binding=sha256_file(path)+encoder.fingerprint+sha256_file(Path(__file__))
        if target.exists():
            with np.load(target,allow_pickle=False) as old:
                if str(old["binding"])!=binding: raise ValueError("缓存版本不同，不能混用")
                sample={key:old[key].copy() for key in old.files if key!="binding"}
        else:
            audio,images=read_clip(path)
            sample={}
            original_frames=np.stack([np.asarray(image) for image in images])
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(str(identifier).encode()).digest()[:4],"little")^42)
            altered_audio={"silence":np.zeros_like(audio),"clipping":np.clip(audio*16,-1,1),
                           "white_noise":rng.normal(0,.15,len(audio)).clip(-1,1).astype(np.float32)}
            black=[Image.fromarray(np.zeros_like(np.asarray(image))) for image in images]
            blur=[Image.fromarray(cv2.GaussianBlur(np.asarray(image),(51,51),15)) for image in images]
            image_variants={"black_video":black,"blur_video":blur}
            vectors=encoder.encode([image for group in image_variants.values() for image in group],
                                   [str(data["raw_text"][i])]*(2*len(images)))["visual"]
            for scenario in SCENARIOS:
                changed_audio=altered_audio.get(scenario,audio)
                changed_frames=np.stack([np.asarray(image) for image in image_variants[scenario]]) if scenario in image_variants else original_frames
                aq=assess_audio_signal(changed_audio,16000);vq=assess_visual_signal(changed_frames)
                af=audio_features(changed_audio) if scenario in altered_audio else data["audio"][i]
                vf=data["visual"][i]
                if scenario in image_variants:
                    offset=0 if scenario=="black_video" else len(images)
                    vf=vectors[offset:offset+len(images)].mean(0);vf=vf/max(np.linalg.norm(vf),1e-8)
                sample[scenario+"_audio"]=af; sample[scenario+"_visual"]=vf
                sample[scenario+"_quality"]=np.array([1.,aq.quality_score,vq.quality_score],np.float32)
                sample[scenario+"_usable"]=np.array([True,aq.usable,vq.usable])
            np.savez_compressed(target,binding=np.array(binding),**sample)
        for scenario in SCENARIOS:
            for key in collected[scenario]:collected[scenario][key].append(sample[scenario+"_"+key])
        if (i+1)%40==0: print(json.dumps({"stage":"raw_quality","prepared":i+1,"total":len(data["labels"])}),flush=True)
    return {scenario:{key:np.stack(values) for key,values in fields.items()} for scenario,fields in collected.items()}


def apply_condition(data: dict[str,np.ndarray], variants: dict, scenario: str, mode: str, threshold: float) -> dict[str,np.ndarray]:
    """三臂共享同一故障特征；自动臂不读取注入位置或情绪标签。"""
    result={key:value.copy() for key,value in data.items()}
    entry=variants[scenario]
    result["audio"],result["visual"]=entry["audio"],entry["visual"]
    result["quality"]=np.ones_like(data["quality"])
    if mode=="automatic":
        result["quality"]=entry["quality"].copy()
        result["modality_mask"] &= entry["usable"] & (entry["quality"]>threshold)
    elif mode=="oracle" and scenario!="clean":
        result["modality_mask"][:,1 if scenario in ["silence","clipping","white_noise"] else 2]=False
    result["quality"]*=result["modality_mask"]
    return result


def predict_old(data: dict[str,np.ndarray], model: ConcatenationHead, scaler: FeatureStandardizer) -> np.ndarray:
    """固定第三轮拼接头，不重新训练以追逐故障测试结果。"""
    mask=data["modality_mask"];q=data["quality"]
    x=scaler.transform({name:data[name] for name in ["text","audio","visual"]},mask)
    with torch.inference_mode():
        values,_,_=model({k:torch.tensor(v) for k,v in x.items()},torch.tensor(mask),torch.tensor(q))
    return values.numpy()


def run(bundle: Path, previous: Path, media: Path, model_path: Path, output: Path) -> dict[str,Any]:
    """开发128条选质量阈值，历史测试240条仅在锁定后作诊断。"""
    output.mkdir(parents=True,exist_ok=True)
    if (output/"quality_results.json").exists():raise ValueError("拒绝覆盖已完成结果")
    seed=42;torch.manual_seed(seed);torch.set_num_threads(1)
    development=load_split(bundle/"train.npz");development=subset(development,sample_indices(development,128))
    protocol={"seed":seed,"development_n":128,"historical_n":240,"scenarios":SCENARIOS,"threshold_grid":[0.,.25,.5],
              "selection":"maximum mean development macro-F1 subject to clean accuracy decrease <= .01 versus ignore",
              "model":"frozen round3 concat", "historical_test":"previously examined; diagnostic only",
              "sample_policy":"ID SHA256 order; no labels or predictions used","code_sha256":sha256_file(Path(__file__)),
              "clean_intervention_rate":"not false positive rate: natural clips lack independent quality gold"}
    save_json(output/"preregister.json",protocol)
    encoder=FrozenCLIPEncoder(model_path,"cuda" if torch.cuda.is_available() else "cpu")
    manifest=json.loads((previous/"concat"/"manifest.json").read_text(encoding="utf-8"))
    model=ConcatenationHead(manifest["dimensions"],3,manifest["hidden"])
    model.load_state_dict(torch.load(previous/"concat"/"weights.pt",map_location="cpu",weights_only=True));model.eval()
    scaler=FeatureStandardizer();scaler.parameters=manifest["standardizer"]
    dev_variants=prepare(development,media,output/"signal_cache",encoder)
    choices=[];baseline=summarize(development["labels"],predict_old(apply_condition(development,dev_variants,"clean","ignore",0),model,scaler))
    for threshold in protocol["threshold_grid"]:
        scores={s:summarize(development["labels"],predict_old(apply_condition(development,dev_variants,s,"automatic",threshold),model,scaler)) for s in SCENARIOS}
        choices.append({"threshold":threshold,"scores":scores,"eligible":scores["clean"]["accuracy"]>=baseline["accuracy"]-.01,
                        "mean_f1":float(np.mean([v["macro_f1"] for v in scores.values()]))})
    eligible=[choice for choice in choices if choice["eligible"]]
    # 没有阈值满足约束则仍报告固定最宽松诊断，但不得选为可交付方案。
    chosen=max(eligible,key=lambda row:row["mean_f1"]) if eligible else choices[0]
    save_json(output/"selection_locked.json",{"chosen":chosen,"choices":choices,"eligible":bool(eligible)})
    historical=load_split(bundle/"test.npz");historical=subset(historical,sample_indices(historical,240))
    variants=prepare(historical,media,output/"signal_cache",encoder)
    results={}; clean_mask=None
    for scenario in SCENARIOS:
        results[scenario]={}; predictions={}
        for mode in ["ignore","oracle","automatic"]:
            data=apply_condition(historical,variants,scenario,mode,chosen["threshold"])
            logits=predict_old(data,model,scaler);predictions[mode]=logits.argmax(1)
            results[scenario][mode]=summarize(historical["labels"],logits)
            results[scenario][mode]["modality_suppressed_fraction"]=(~data["modality_mask"]).mean(0).tolist()
            if scenario=="clean" and mode=="automatic":clean_mask=data["modality_mask"]
            np.savez_compressed(output/(scenario+"_"+mode+".npz"),sample_ids=historical["sample_ids"],labels=historical["labels"],logits=logits)
        results[scenario]["automatic_minus_ignore"]={"delta_macro_f1":results[scenario]["automatic"]["macro_f1"]-results[scenario]["ignore"]["macro_f1"],
                "ci95":grouped_delta_ci(historical["labels"],predictions["automatic"],predictions["ignore"],historical["source_groups"],3)}
    report={"protocol":protocol,"selected_threshold":chosen["threshold"],"eligible_for_delivery":bool(eligible),"development":choices,
            "historical_diagnostic":results,"source_groups":len(np.unique(historical["source_groups"])),
            "clean_modality_suppression_rate":(~clean_mask).mean(0).tolist(),"real_world_false_positive_rate":None,
            "deployed":False,"limitations":["controlled raw-signal faults, not independently annotated real noise", "only sampled clips; no target-population claims",
                 "quality threshold chosen on training subset, not historical test", "no lip-sync or face-identity quality ground truth"]}
    save_json(output/"quality_results.json",report)
    print(json.dumps({"threshold":chosen["threshold"],"eligible":bool(eligible),"historical":results}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["bundle","previous","media","model-path","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.previous,args.media,args.model_path,args.output)
