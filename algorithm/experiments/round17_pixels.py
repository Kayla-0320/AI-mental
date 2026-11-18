"""并行测量真实原片质量及可观察中心遮挡；不读取标签、音轨或语义模型。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from shared.dataclasses import RawSignalQuality
from data_quality.raw_signals import assess_visual_signal
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round13_train import ch_data
from experiments.round17_quality import MEDIA,OUT


def quality_from_frames(frames: np.ndarray) -> RawSignalQuality:
    """补充中心大块平坦遮挡代理；仅为图像质量证据，不判断情绪或真实性。"""
    original=assess_visual_signal(frames);result=RawSignalQuality(**asdict(original))
    if not len(frames):return result
    blocked=[]
    for frame in frames:
        gray=cv2.cvtColor(cv2.resize(frame.astype(np.uint8),(128,128)),cv2.COLOR_RGB2GRAY)
        middle=gray[32:96,32:96];ring=gray.copy();ring[24:104,24:104]=0
        outside=np.concatenate((gray[:24].ravel(),gray[104:].ravel(),gray[24:104,:24].ravel(),gray[24:104,104:].ravel()))
        blocked.append(float(np.std(middle)<2 and 20<float(middle.mean())<235 and np.std(outside)>10))
    fraction=float(np.mean(blocked));result.metrics["uniform_center_block_fraction"]=fraction
    if fraction>=.8:
        result.quality_score=min(result.quality_score,.25);result.reason_codes.append("uniform_center_occlusion_proxy")
    result.limitations.append("uniform_center_proxy_not_all_occlusions_or_human_quality_gold")
    return result


def frames(path: Path) -> np.ndarray:
    """单解码线程逐片提取与旧测量同位置的三个缩略帧。"""
    options=[cv2.CAP_PROP_N_THREADS,1] if hasattr(cv2,"CAP_PROP_N_THREADS") else []
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,options);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));values=[]
    for fraction in (.2,.5,.8):
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,round((total-1)*fraction)));ok,frame=cap.read()
        if ok:values.append(cv2.cvtColor(cv2.resize(frame,(128,128)),cv2.COLOR_BGR2RGB))
    cap.release();return np.stack(values) if values else np.empty((0,128,128,3),np.uint8)


def run() -> None:
    """保存两种质量分数及真实像素故障探针，线程只加速解码不参与模型选择。"""
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/"pixels_protocol.json",{"seed":42,"workers":6,"decode_threads":1,"source_sha256":sha256(Path(__file__)),
        "center_rule":"center64std<2, mean20..235, outside std>10 in >=80% frames; soft score capped.25",
        "probes":"48 ID-hash-selected clips, actual pixel transformations, detection only"})
    data=ch_data();ids=data["sample_ids"].astype(str).tolist()
    selected=set(sorted(ids,key=lambda x:hashlib.sha256(("42|"+x).encode()).digest())[:48])
    path=OUT/"raw_quality_v2.json";records=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old_path=OUT/"raw_quality.json";old=json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
    probes_path=OUT/"detector_probes_v2.json";probes=json.loads(probes_path.read_text(encoding="utf-8")) if probes_path.exists() else []
    def worker(identifier: str) -> tuple[str,dict,dict,list[dict]]:
        """每个线程只读取一个视频并返回质量，未接收标签或预测。"""
        image=frames(MEDIA/(identifier+".mp4"));items=[]
        if not len(image):
            row={"measured":False,"score":None,"reason":"decode_unavailable_not_assumed_bad"};return identifier,row,row,items
        first=assess_visual_signal(image);second=quality_from_frames(image);digest=hashlib.sha256(image.tobytes()).hexdigest()
        a={"measured":True,"score":first.quality_score,"metrics":first.metrics,"reasons":first.reason_codes,"frame_sha256":digest}
        b={"measured":True,"score":second.quality_score,"metrics":second.metrics,"reasons":second.reason_codes,"frame_sha256":digest}
        if identifier in selected:
            variants={"clean":image,"black":np.zeros_like(image),"dark":(image*.03).astype(np.uint8),"blur":np.stack([cv2.GaussianBlur(x,(31,31),9) for x in image]),"center_occlusion":image.copy()}
            variants["center_occlusion"][:,24:104,24:104]=127
            for condition,value in variants.items():
                q1,q2=assess_visual_signal(value),quality_from_frames(value)
                items.append({"id":identifier,"condition":condition,"old_score":q1.quality_score,"new_score":q2.quality_score,"new_reasons":q2.reason_codes})
        return identifier,a,b,items
    work=[i for i in ids if i not in records]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for n,(identifier,a,b,items) in enumerate(pool.map(worker,work),1):
            old[identifier]=a;records[identifier]=b;probes.extend(items)
            if n%100==0:
                save(path,records);save(old_path,old);save(probes_path,probes);print(json.dumps({"pixel_clips":len(records),"total":len(ids)}),flush=True)
    save(path,records);save(old_path,old);save(probes_path,probes)
    save(OUT/"detector_probes.json",[{"id":r["id"],"condition":r["condition"],"score":r["old_score"]} for r in probes])
    print(json.dumps({"pixel_clips":len(records),"probes":len(probes)}),flush=True)


if __name__=="__main__":run()
