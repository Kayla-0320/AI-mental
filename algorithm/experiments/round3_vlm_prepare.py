"""完整公开v2监督原片接入、预先冻结来源组划分和真实冻结图文/音频特征。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.knowledge_moe.features import save_json, sha256_file

ARCHIVE_SIZE = 11588045887
ARCHIVE_SHA = "e2817c4841a74f9e73eed6cf3196442ff0245f999bdfc5f975dcf18e66348f1e"
REVISION = "ff135d32adaac2c3d5b0268e88a7ce6af200f544"


def freeze_plan(tsv_dir: Path, output: Path) -> list[dict[str, Any]]:
    """冻结来源组；测试仅选从未在v1实验及四例前向中出现的来源。"""
    rows: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        with (tsv_dir / f"{split}.tsv").open(encoding="utf-8-sig", newline="") as handle:
            for r in csv.DictReader(handle, delimiter="\t"):
                rows.append({"id": r["id"], "text": r["text"], "label": r["label"].strip(),
                             "source_group": r["id"].rsplit("_", 1)[0], "original_split": split})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("监督样本ID不唯一")
    groups = sorted({r["source_group"] for r in rows})
    previously_seen = {f"video_{i:04d}" for i in range(1, 61)} | {"aqgy5_0016"}
    new_groups = [g for g in groups if g not in previously_seen]
    rng = random.Random(42)
    rng.shuffle(new_groups)
    n_test = max(1, round(len(groups) * .2))
    if len(new_groups) < n_test + 10:
        raise ValueError("新来源组不足以留出独立测试")
    assignment = {g: "test" for g in new_groups[:n_test]}
    remaining = [g for g in groups if g not in assignment]
    rng.shuffle(remaining)
    n_valid = max(4, round(len(groups) * .2))
    for j, group in enumerate(remaining[:n_valid]):
        assignment[group] = "selection" if j % 2 else "calibration"
    for group in remaining[n_valid:]:
        assignment[group] = "train"
    for row in rows:
        row["split"] = assignment[row["source_group"]]
    plan = {"schema_version": "round3_frozen_source_split.v1", "seed": 42, "task": "ch_sims_v2_sentiment_polarity_3",
            "class_names": ["negative", "neutral", "positive"], "samples": rows,
            "test_selection": "only source video IDs absent from v1 video_0001..0060 and prior 4-clip smoke",
            "limitations": ["来源视频互斥，不保证演员/影片/节目互斥", "公开镜像TSV为4400条；原论文4402不能冒充本地实际数量"],
            "tsv_sha256": {s: sha256_file(tsv_dir / f"{s}.tsv") for s in ("train", "dev", "test")}}
    path = output / "frozen_split.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("已有冻结划分不同，不允许覆盖")
    save_json(path, plan)
    print(json.dumps({"stage": "split_frozen", "n": len(rows), "counts": {s: sum(r["split"] == s for r in rows) for s in ["train", "selection", "calibration", "test"]}}), flush=True)
    return rows


def download_archive(path: Path) -> None:
    """匿名下载完整媒体归档并核对发布者LFS摘要，断点续传不接受错位Range。"""
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == ARCHIVE_SIZE and sha256_file(path) == ARCHIVE_SHA:
        return
    part = path.with_suffix(path.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    url = f"https://huggingface.co/datasets/THUIAR/MMLA-Datasets/resolve/{REVISION}/CH-SIMSv2.0/Ch-simsv2_video.tar.gz?download=true&round3=full"
    from concurrent.futures import ThreadPoolExecutor, as_completed
    start = time.monotonic()
    chunks = path.parent / "download_chunks"
    chunks.mkdir(exist_ok=True)
    block = 64 * 1024 * 1024
    ranges = [(lo,min(lo+block,ARCHIVE_SIZE)-1) for lo in range(offset,ARCHIVE_SIZE,block)]

    def fetch(bounds: tuple[int,int]) -> Path:
        lo,hi=bounds;target=chunks/f"{lo}-{hi}.bin"
        if target.exists() and target.stat().st_size==hi-lo+1:
            return target
        for attempt in range(3):
            try:
                response=requests.get(url+f"&range_start={lo}",headers={"Range":f"bytes={lo}-{hi}"},timeout=(20,40))
                response.raise_for_status()
                if response.status_code!=206 or response.headers.get("content-range")!=f"bytes {lo}-{hi}/{ARCHIVE_SIZE}" or len(response.content)!=hi-lo+1:
                    raise ValueError("分片响应起止或长度不符")
                temp=target.with_suffix(".part");temp.write_bytes(response.content);temp.replace(target)
                return target
            except (requests.RequestException,ValueError):
                if attempt==2:raise
        raise ValueError("下载分片失败")
    completed=offset
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,bounds):bounds for bounds in ranges}
        for future in as_completed(futures):
            downloaded=future.result();completed+=downloaded.stat().st_size
            print(json.dumps({"stage":"range_download","verified_range_bytes":completed,"total":ARCHIVE_SIZE,"seconds":round(time.monotonic()-start,1)}),flush=True)
    with part.open("ab" if offset else "wb") as handle:
        for lo,hi in ranges:
            with (chunks/f"{lo}-{hi}.bin").open("rb") as source:
                while piece:=source.read(4*1024*1024):handle.write(piece)
    if sha256_file(part) != ARCHIVE_SHA:
        raise ValueError("完整归档摘要与发布者不符")
    part.replace(path)
    save_json(path.with_suffix(".provenance.json"), {"revision": REVISION, "bytes": ARCHIVE_SIZE, "sha256": ARCHIVE_SHA, "authenticated": False, "complete_archive": True})


def extract_supervised(archive: Path, rows: list[dict[str, Any]], destination: Path) -> None:
    """只提取清单内的普通媒体成员，不执行归档路径或链接。"""
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {r["id"] for r in rows}
    found: set[str] = set()
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.startswith("Ch-sims_") or not name.endswith(".mp4"):
                continue
            sample_id = name.removeprefix("Ch-sims_").removesuffix(".mp4")
            if sample_id not in wanted:
                continue
            target = destination / (sample_id + ".mp4")
            if not target.exists() or target.stat().st_size != member.size:
                stream = handle.extractfile(member)
                if stream is None:
                    raise ValueError("媒体成员不可读取")
                with target.open("wb") as out:
                    while chunk := stream.read(1024 * 1024):
                        out.write(chunk)
            found.add(sample_id)
    if found != wanted:
        raise ValueError(f"监督媒体不完整，缺少{len(wanted-found)}条: {sorted(wanted-found)[:5]}")
    print(json.dumps({"stage": "media_complete", "count": len(found)}), flush=True)


def encode_all(rows: list[dict[str, Any]], media: Path, model_path: Path, output: Path) -> None:
    """有界CPU预取与单GPU批量编码；算法与单条缓存的三帧/音频约定一致。"""
    import cv2
    import librosa
    import numpy as np
    import torch
    import av
    from PIL import Image
    from dataclasses import asdict
    from concurrent.futures import ThreadPoolExecutor
    from data_quality.raw_signals import assess_audio_signal, assess_visual_signal, assess_av_alignment
    from perception.knowledge_moe.features import FrozenCLIPEncoder
    from shared import RawSignalQuality
    torch.manual_seed(42);torch.set_num_threads(1)
    encoder=FrozenCLIPEncoder(model_path,'cuda' if torch.cuda.is_available() else 'cpu')
    feature_dir=output/'sample_features';feature_dir.mkdir(parents=True,exist_ok=True)
    version='clip3_mfcc46_raw_quality_saturation_v1'

    def prepare(row: dict[str,Any]) -> tuple:
        path=media/(row['id']+'.mp4')
        cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        images=[]
        for fraction in (.2,.5,.8):
            cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,min(count-1,round(count*fraction))));ok,frame=cap.read()
            if ok:images.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
        cap.release()
        fallback=0;decoded=0
        if not images:
            fallback=1
            recovery=av.open(str(path));stream=recovery.streams.video[0]
            duration=float(stream.duration*stream.time_base) if stream.duration else (count/fps if fps>0 else 1.)
            targets=[duration*f for f in (.2,.5,.8)];best=[None,None,None];distances=[float('inf')]*3
            for frame in recovery.decode(video=0):
                position=float(frame.time) if frame.time is not None else decoded/max(fps,1.)
                for j,target in enumerate(targets):
                    if abs(position-target)<distances[j]:
                        distances[j]=abs(position-target);best[j]=frame.to_image()
                decoded+=1
            recovery.close();images=[image for image in best if image is not None]
        container=av.open(str(path));resampler=av.audio.resampler.AudioResampler(format='fltp',layout='mono',rate=16000)
        parts=[]
        if container.streams.audio:
            for frame in container.decode(audio=0):
                parts.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
            parts.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
        container.close()
        signal=np.concatenate(parts).astype(np.float32) if parts else np.empty(0,np.float32)
        peak=float(np.max(np.abs(signal))) if len(signal) else 0.
        overshoot=float((np.abs(signal)>1).mean()) if len(signal) else 0.
        signal=np.clip(signal,-1,1)
        observed=len(signal)>=512 and np.isfinite(signal).all()
        if observed:
            mfcc=librosa.feature.mfcc(y=signal,sr=16000,n_mfcc=20,n_fft=512,hop_length=160)
            rms=librosa.feature.rms(y=signal,frame_length=512,hop_length=160)[0]
            zcr=librosa.feature.zero_crossing_rate(signal,frame_length=512,hop_length=160)[0]
            centroid=librosa.feature.spectral_centroid(y=signal,sr=16000,n_fft=512,hop_length=160)[0]/8000.
            audio=np.concatenate([mfcc.mean(axis=1),mfcc.std(axis=1),[rms.mean(),rms.std(),zcr.mean(),zcr.std(),centroid.mean(),centroid.std()]]).astype(np.float32)
        else:audio=np.zeros(46,np.float32)
        aq=assess_audio_signal(signal,16000);aq.metrics.update(decoded_peak_before_saturation=peak,decoded_overshoot_fraction=overshoot)
        vq=assess_visual_signal(np.stack([np.asarray(image) for image in images]) if images else np.zeros((0,1,1,3),np.uint8))
        vq.metrics.update(pyav_sampling_fallback=float(fallback),fallback_decoded_frames=float(decoded))
        if len(signal) and count>0 and fps>0:
            sync=assess_av_alignment(len(signal)/16000,count/fps)
        else:
            sync=RawSignalQuality('alignment',False,0.,{},['missing_duration_metadata'],['container_metadata_consistency_only_not_lip_sync_ground_truth'])
        mask=np.array([True,observed and aq.usable and sync.usable,vq.usable and sync.usable])
        quality=np.array([1.,aq.quality_score*float(mask[1]),vq.quality_score*float(mask[2])],np.float32)
        details=json.dumps({'audio':asdict(aq),'visual':asdict(vq),'synchrony':asdict(sync)},ensure_ascii=False)
        return row,images,audio,mask,quality,details,sha256_file(path)

    remaining=[]
    for row in rows:
        path=feature_dir/(row['id']+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as cache:
                if 'pipeline_version' in cache and str(cache['pipeline_version'])==version and str(cache['encoder_fingerprint'])==encoder.fingerprint:continue
        remaining.append(row)
    start=time.monotonic();completed=len(rows)-len(remaining)
    print(json.dumps({'stage':'encode_resume','cached':completed,'remaining':len(remaining),'cpu_workers':4,'gpu_batch_clips':4}),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={i:pool.submit(prepare,row) for i,row in enumerate(remaining[:8])};submitted=len(pending);next_index=0
        while next_index<len(remaining):
            prepared=[]
            for _ in range(min(4,len(remaining)-next_index)):
                prepared.append(pending.pop(next_index).result());next_index+=1
                if submitted<len(remaining):
                    pending[submitted]=pool.submit(prepare,remaining[submitted]);submitted+=1
            images=[image for item in prepared for image in item[1]]
            texts=[item[0]['text'] for item in prepared for _ in item[1]]
            vectors=encoder.encode(images,texts) if images else {'text':np.zeros((0,512),np.float32),'visual':np.zeros((0,512),np.float32)};cursor=0
            for row,frames,audio,mask,q,details,media_sha in prepared:
                end=cursor+len(frames)
                text_vector=vectors['text'][cursor] if frames else encoder.encode_text([row['text']])[0]
                visual_vector=vectors['visual'][cursor:end].mean(axis=0) if frames else np.zeros(512,np.float32)
                np.savez_compressed(feature_dir/(row['id']+'.npz'),text=text_vector,visual=visual_vector,audio=audio,
                    modality_mask=mask,quality=q,raw_quality_json=np.array(details),pipeline_version=np.array(version),media_sha256=np.array(media_sha),encoder_fingerprint=np.array(encoder.fingerprint))
                cursor=end;completed+=1
            if completed%40<4 or completed==len(rows):
                print(json.dumps({'stage':'encode','done':completed,'total':len(rows),'seconds_since_resume':round(time.monotonic()-start,1)}),flush=True)
    canonical=['text','audio','visual'];records=[]
    for row in rows:
        with np.load(feature_dir/(row['id']+'.npz'),allow_pickle=False) as z:
            if str(z['encoder_fingerprint'])!=encoder.fingerprint:raise ValueError('缓存编码器身份不同')
            records.append({k:z[k] for k in z.files})
    for split in ('train','selection','calibration','test'):
        indices=[i for i,r in enumerate(rows) if r['split']==split]
        payload={name:np.stack([records[i][name] for i in indices]) for name in canonical+['modality_mask','quality']}
        payload.update(labels=np.array([['negative','neutral','positive'].index(rows[i]['label']) for i in indices],np.int64),
            sample_ids=np.array([rows[i]['id'] for i in indices]),source_groups=np.array([rows[i]['source_group'] for i in indices]),
            raw_text=np.array([rows[i]['text'] for i in indices]),original_split=np.array([rows[i]['original_split'] for i in indices]))
        np.savez_compressed(output/(split+'.npz'),**payload)
    audio_fingerprint='librosa_mfcc20_mean_std_rms_zcr_centroid_16k_fft512_hop160_saturation_v1'
    save_json(output/'metadata.json',{'schema_version':'paired_features_v1','task':'ch_sims_v2_sentiment_polarity_3','dataset':'CH-SIMS-v2',
        'class_names':['negative','neutral','positive'],'modality_order':canonical,'feature_dims':{'text':512,'audio':46,'visual':512},
        'encoder_fingerprints':{'text':encoder.fingerprint+';component=text_l2_v1','audio':audio_fingerprint,'visual':encoder.fingerprint+';component=vision_l2_mean3_v1'},
        'live_text_encoder_fingerprint':encoder.fingerprint,'visual_frames':[.2,.5,.8],'frozen_split_sha256':sha256_file(output/'frozen_split.json'),
        'media_archive_sha256':ARCHIVE_SHA,'n':len(rows),'audio_kind':'real_signal_handcrafted_not_pretrained_speech_model','device':encoder.device,
        'encoding_seconds_since_last_resume':time.monotonic()-start,'pipeline_version':version,
        'raw_quality_code_sha256':sha256_file(Path(__file__).resolve().parents[1]/'data_quality/raw_signals.py'),
        'limitations':['三帧池化不是完整时序视频理解','手工音频特征不是预训练语音编码器','来源视频隔离不保证演员/影片隔离']})
    print(json.dumps({'stage':'feature_bundle_complete','n':len(rows)}),flush=True)


def main() -> None:
    """顺序完成冻结、媒体获取与全量特征，允许断点恢复。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", type=Path, required=True);p.add_argument("--output", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True);p.add_argument("--model", type=Path, required=True)
    p.add_argument("--encode-only",action="store_true")
    a = p.parse_args();a.output.mkdir(parents=True, exist_ok=True)
    rows = freeze_plan(a.tsv, a.output)
    archive = a.media_root / "Ch-simsv2_video.tar.gz"
    download_archive(archive)
    if not a.encode_only:
        extract_supervised(archive, rows, a.media_root / "supervised")
    elif any(not (a.media_root/"supervised"/(r["id"]+".mp4")).is_file() for r in rows):
        raise ValueError("encode-only要求全部监督原片已存在")
    encode_all(rows, a.media_root / "supervised", a.model, a.output)


if __name__ == "__main__":
    main()
