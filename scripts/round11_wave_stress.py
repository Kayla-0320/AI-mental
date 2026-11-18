"""固定开发样本的真实波形噪声压力，不把特征扰动冒充音频噪声。"""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
import torch
import librosa
from experiments.round11_common import ROOT,OUTPUT,digest,save
from round11_sensevoice import build_model,encode,decode
from round11_external_gpu import msp_model,msp_encode


def mfcc(signal: np.ndarray) -> np.ndarray:
    """与冻结基线相同MFCC46，噪声施加在波形上。"""
    values=librosa.feature.mfcc(y=signal,sr=16000,n_mfcc=20,n_fft=512,hop_length=160)
    rms=librosa.feature.rms(y=signal,frame_length=512,hop_length=160)[0]
    zcr=librosa.feature.zero_crossing_rate(signal,frame_length=512,hop_length=160)[0]
    centroid=librosa.feature.spectral_centroid(y=signal,sr=16000,n_fft=512,hop_length=160)[0]/8000.
    return np.concatenate((values.mean(1),values.std(1),[rms.mean(),rms.std(),zcr.mean(),zcr.std(),centroid.mean(),centroid.std()])).astype(np.float32)


def main() -> None:
    """全量2876条开发媒体，不按模型是否识别正确删样本。"""
    output=OUTPUT/"wave_stress";output.mkdir(parents=True,exist_ok=True)
    if (output/"complete.json").exists():return
    with np.load(OUTPUT/"evidence_deterministic/family10/oof.npz",allow_pickle=False) as z:
        ids=z["sample_ids"];groups=z["source_groups"]
    selected=ids.astype(str).tolist()
    plan={"n":len(selected),"sample_ids":selected,"selection":"all development IDs in frozen order;no label-based filtering",
        "snr_db":[20,10],"noise":"iid Gaussian;signal RMS/noise RMS scaling;clip[-1,1];not real environmental recordings",
        "duration":"same center10sec waveform as frozen speech encoders;MFCC also recomputed on this interval",
        "quality":"existing observation masks fixed to isolate representation noise;not an automatic quality detector test",
        "seed":"42 xor first32bits sha256(id)","code_sha256":digest(Path(__file__))}
    prereg=output/"preregister.json"
    if prereg.exists():
        if json.loads(prereg.read_text("utf-8"))!=plan:raise ValueError("波形压力预注册不符")
    else:save(prereg,plan)
    torch.manual_seed(42);torch.set_num_threads(1)
    media=ROOT/"data/public/ch_sims_v2/supervised"
    for kind in ("sense","msp"):
        if kind=="sense":model,kwargs=build_model()
        else:model,extractor,head=msp_model()
        for snr in ("clean",20,10):
            target=output/f"{kind}_{snr}.npz"
            if target.exists():continue
            features=[];extras=[];emotions=[];transcripts=[];mfccs=[];clipped=[]
            for start in range(0,len(selected),8):
                waves=[]
                for key in selected[start:start+8]:
                    wave,_,ok=decode(media/f"{key}.mp4")
                    if not ok:raise ValueError("压力审计选中音轨不可解码；禁止替换样本")
                    if snr!="clean":
                        rng=np.random.default_rng(42^int(hashlib.sha256(key.encode()).hexdigest()[:8],16))
                        noise=rng.standard_normal(len(wave)).astype(np.float32)
                        noise*=np.sqrt(np.mean(wave**2)/(np.mean(noise**2)*10**(snr/10)))
                        mixed=wave+noise;clipped.append(float(np.mean(np.abs(mixed)>1)));wave=np.clip(mixed,-1,1)
                    else:clipped.append(0.)
                    waves.append(wave);mfccs.append(mfcc(wave))
                if kind=="sense":
                    pooled,query,emo,text=encode(model,kwargs,waves);features.extend(pooled);extras.extend(query[:,1,:]);emotions.extend(emo[:,1,:]);transcripts.extend(text)
                else:
                    pooled,emo=msp_encode(model,extractor,head,waves);features.extend(pooled);emotions.extend(emo)
                gc.collect();torch.cuda.empty_cache()
            payload={"sample_ids":np.array(selected),"audio":np.stack(mfccs),"features":np.stack(features),"emotion":np.stack(emotions),"clipped_fraction":np.array(clipped)}
            if kind=="sense":payload.update(query=np.stack(extras),transcript=np.array(transcripts))
            np.savez_compressed(target,**payload);print(json.dumps({"wave_stress":kind,"snr":snr,"n":len(selected)}),flush=True)
        del model
        if kind=="msp":del head
        gc.collect();torch.cuda.empty_cache()
    save(output/"complete.json",{"n":len(selected),"SNR":[20,10],"labels_used_in_encoding":False,"preregister_sha256":digest(prereg)})


if __name__=="__main__":main()
