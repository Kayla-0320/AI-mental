"""外部真实媒体的MFCC、三帧原始质量与时间戳；同时保留旧测量对照。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")

import av
import cv2
import librosa
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_quality.raw_signals import assess_audio_signal,assess_visual_signal,assess_av_alignment
from experiments.round11_common import ROOT,OUTPUT,digest,save
from round11_input_audit import presentation_span
from round11_resources import require_memory


def frames(path: Path) -> tuple[list[Image.Image],int,float]:
    """复用冻结三帧位置和OpenCV采样，失败时才按原规则PyAV恢复。"""
    capture=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=float(capture.get(cv2.CAP_PROP_FPS))
    images=[]
    for fraction in (.2,.5,.8):
        capture.set(cv2.CAP_PROP_POS_FRAMES,max(0,min(count-1,round(count*fraction))))
        ok,frame=capture.read()
        if ok:
            images.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    capture.release()
    if not images:
        with av.open(str(path)) as source:
            stream=source.streams.video[0]
            stream.codec_context.thread_count=1
            duration=float(stream.duration*stream.time_base) if stream.duration else (count/fps if fps>0 else 1.)
            targets=[duration*fraction for fraction in (.2,.5,.8)]
            best,distances=[None]*3,[float("inf")]*3
            for i,frame in enumerate(source.decode(video=0)):
                position=float(frame.time) if frame.time is not None else i/max(fps,1.)
                for j,target in enumerate(targets):
                    if abs(position-target)<distances[j]:
                        distances[j]=abs(position-target)
                        best[j]=frame.to_image()
            images=[image for image in best if image is not None]
    return images,count,fps


def prepare(path: Path) -> dict[str,Any]:
    """旧规则与新时轴共享完全相同的波形、RGB质量规则和阈值。"""
    images,count,fps=frames(path)
    audio_times,audio_durations,parts=[],[],[]
    with av.open(str(path)) as source:
        stream=source.streams.audio[0]
        stream.codec_context.thread_count=1
        resampler=av.AudioResampler(format="fltp",layout="mono",rate=16000)
        for frame in source.decode(audio=0):
            if frame.time is not None:
                audio_times.append(float(frame.time))
                audio_durations.append(frame.samples/frame.sample_rate)
            parts.extend(item.to_ndarray().reshape(-1) for item in resampler.resample(frame))
        parts.extend(item.to_ndarray().reshape(-1) for item in resampler.resample(None))
    signal=np.concatenate(parts).astype(np.float32) if parts else np.empty(0,np.float32)
    signal=np.clip(signal,-1,1)
    audio_observed=len(signal)>=512 and np.isfinite(signal).all()
    if audio_observed:
        mfcc=librosa.feature.mfcc(y=signal,sr=16000,n_mfcc=20,n_fft=512,hop_length=160)
        rms=librosa.feature.rms(y=signal,frame_length=512,hop_length=160)[0]
        zcr=librosa.feature.zero_crossing_rate(signal,frame_length=512,hop_length=160)[0]
        centroid=librosa.feature.spectral_centroid(y=signal,sr=16000,n_fft=512,hop_length=160)[0]/8000.
        audio=np.concatenate((mfcc.mean(1),mfcc.std(1),[rms.mean(),rms.std(),zcr.mean(),zcr.std(),centroid.mean(),centroid.std()])).astype(np.float32)
    else:
        audio=np.zeros(46,np.float32)
    aq=assess_audio_signal(signal,16000)
    vq=assess_visual_signal(np.stack([np.asarray(image) for image in images]) if images else np.zeros((0,1,1,3),np.uint8))
    old_sync=assess_av_alignment(len(signal)/16000,count/fps) if len(signal) and count>0 and fps>0 else None
    old_usable=old_sync is not None and old_sync.usable
    old_mask=[True,bool(audio_observed and aq.usable and old_usable),bool(vq.usable and old_usable)]
    old_quality=[1.,aq.quality_score*old_mask[1],vq.quality_score*old_mask[2]]
    video_times,video_durations=[],[]
    with av.open(str(path)) as source:
        stream=source.streams.video[0]
        stream.codec_context.thread_count=1
        duration=float(stream.duration*stream.time_base) if stream.duration else None
        for frame in source.decode(video=0):
            if frame.time is not None:
                video_times.append(float(frame.time))
                video_durations.append(float(frame.duration*frame.time_base) if frame.duration else 0.)
    video=presentation_span(video_times,video_durations)
    a=presentation_span(audio_times,audio_durations)
    verified=video["verified"] and a["verified"]
    if duration is not None:
        verified=verified and abs(video["duration"]-duration)<=max(.25,.1*duration)
    sync=None
    mask,quality=old_mask,old_quality
    if verified:
        origin=min(video["start"],a["start"],0.)
        sync=assess_av_alignment(len(signal)/16000,video["duration"],audio_start=a["start"]-origin,video_start=video["start"]-origin)
        mask=[True,bool(audio_observed and aq.usable and sync.usable),bool(vq.usable and sync.usable)]
        quality=[1.,aq.quality_score*mask[1],vq.quality_score*mask[2]]
    details={"audio":asdict(aq),"visual":asdict(vq),"old_alignment":asdict(old_sync) if old_sync else None,
             "pts_alignment":asdict(sync) if sync else None,"video_pts":video,"audio_pts":a,"pts_verified":bool(verified)}
    return {"audio":audio,"legacy_mask":np.array(old_mask,bool),"legacy_quality":np.array(old_quality,np.float32),
            "modality_mask":np.array(mask,bool),"quality":np.array(quality,np.float32),
            "duration":np.array(len(signal)/16000),"raw_quality_json":np.array(json.dumps(details,ensure_ascii=False))}


def main() -> None:
    """媒体完整性先验收；不在信号工件中混入任一数据集的标签。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    if not (OUTPUT/"emotiontalk_acquisition/complete.json").exists():
        raise ValueError("原始媒体完整性尚未验收")
    cv2.setNumThreads(1)
    output=OUTPUT/"emotiontalk_features/signal_features"
    cache=output/"samples_v2_bounded"
    cache.mkdir(parents=True,exist_ok=True)
    binding=digest(Path(__file__))+digest(Path(__file__).with_name("round11_input_audit.py"))
    archived=OUTPUT/"source/media_initial/round11_external_signals.py"
    compatible=digest(archived)+digest(Path(__file__).with_name("round11_input_audit.py"))

    def process(key: str) -> dict[str,np.ndarray]:
        require_memory()
        target=cache/f"{key}.npz"
        if target.exists():
            with np.load(target,allow_pickle=False) as record:
                if str(record["binding"])!=binding:
                    raise ValueError("信号缓存绑定不符")
                return {name:record[name] for name in record.files if name not in ("binding","sample_id")}
        legacy=output/"samples"/f"{key}.npz"
        if legacy.exists():
            with np.load(legacy,allow_pickle=False) as record:
                details=json.loads(str(record["raw_quality_json"]))
                if str(record["binding"])!=compatible:
                    raise ValueError("初版原始信号缓存身份不符")
                if details["visual"]["metrics"]["frame_count"]==3 and details["pts_verified"]:
                    return {name:record[name] for name in record.files if name not in ("binding","sample_id")}
        result=prepare(ROOT/"data/public/emotiontalk_media/clips"/f"{key}.mp4")
        if json.loads(str(result["raw_quality_json"]))["visual"]["metrics"]["frame_count"]!=3:
            require_memory(6.)
            repeated=prepare(ROOT/"data/public/emotiontalk_media/clips"/f"{key}.mp4")
            if any(not np.array_equal(result[name],repeated[name]) for name in result):
                raise ValueError("不完整信号无法在内存充足时稳定复现，拒绝缓存")
        temporary=target.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle,**result,binding=np.array(binding),sample_id=np.array(key))
        temporary.replace(target)
        return result

    started=time.monotonic()
    for split in ("train","valid","test_inputs"):
        target=output/f"{split}.npz"
        if target.exists():
            continue
        ids=[json.loads(line)["id"] for line in (OUTPUT/"corpus"/f"{split}.jsonl").read_text("utf-8").splitlines()]
        values=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0,len(ids),32):
                values.extend(pool.map(process,ids[start:start+32]))
                if start%256==0 or start+32>=len(ids):
                    print(json.dumps({"signals":len(values),"total":len(ids),"split":split,"seconds":round(time.monotonic()-started,1)}),flush=True)
        np.savez_compressed(target,sample_ids=np.array(ids),**{name:np.stack([row[name] for row in values]) for name in values[0]})
    save(output/"manifest.json",{"binding":binding,"compatible_initial_binding":compatible,
        "execution_change":"one decoder thread;bounded32clip queues;initial caches require3frames and verifiedPTS",
        "labels_used":False,"thresholds":{"seconds":.25,"relative":.1},
        "audio":"same16kHz MFCC46 as frozen baseline","visual_quality":"same3RGB frame observable proxies",
        "quality_change":"decoded presentation timestamp span vs frame_count/fps;other criteria unchanged"})


if __name__=="__main__":
    main()
