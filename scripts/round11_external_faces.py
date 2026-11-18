"""相同五帧、检测器、裁剪与HSEmotion权重的外部人脸特征。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS","1")
import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import ROOT,OUTPUT,PREVIOUS,digest,save
from round11_resources import require_memory

DETECTOR=ROOT/"_model_cache/round5/yunet.onnx"
BACKBONE=PREVIOUS/"deep_face_features/hsemotion_embedding.onnx"
LOCAL=threading.local()


def encode(path: Path) -> dict[str,np.ndarray]:
    """严格复用最大脸规则；没有身份验证，不把最大脸声称为确认说话人。"""
    if not hasattr(LOCAL,"detector"):
        LOCAL.detector=cv2.FaceDetectorYN.create(str(DETECTOR),"",(320,320),.8,.3,5000)
        options=ort.SessionOptions()
        options.intra_op_num_threads=1
        options.inter_op_num_threads=1
        LOCAL.session=ort.InferenceSession(str(BACKBONE),options,providers=["CPUExecutionProvider"])
    capture=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    scores,embeddings,areas,confidences,face_counts=[],[],[],[],[]
    decoded=0
    for fraction in (.1,.3,.5,.7,.9):
        capture.set(cv2.CAP_PROP_POS_FRAMES,max(0,min(count-1,round(count*fraction))))
        ok,frame=capture.read()
        if not ok:
            continue
        decoded+=1
        height,width=frame.shape[:2]
        scaled=cv2.resize(frame,(320,max(32,round(height*320/width))))
        LOCAL.detector.setInputSize((scaled.shape[1],scaled.shape[0]))
        _,faces=LOCAL.detector.detect(scaled)
        face_counts.append(0 if faces is None else len(faces))
        if faces is None or not len(faces):
            continue
        face=max(faces,key=lambda row:row[2]*row[3])
        x,y,fw,fh=face[:4]
        sx,sy=width/scaled.shape[1],height/scaled.shape[0]
        left,top=max(0,int((x-.1*fw)*sx)),max(0,int((y-.1*fh)*sy))
        right,bottom=min(width,int((x+1.1*fw)*sx)),min(height,int((y+1.1*fh)*sy))
        if right<=left or bottom<=top:
            continue
        crop=cv2.cvtColor(cv2.resize(frame[top:bottom,left:right],(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        values=((crop-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None]
        outputs=LOCAL.session.run(None,{LOCAL.session.get_inputs()[0].name:values})
        score=outputs[0].reshape(-1)
        embedding=outputs[-1].reshape(-1)
        if score.shape!=(8,) or embedding.shape!=(1280,) or not np.isfinite(embedding).all():
            raise ValueError("表情网络输出维数或数值不符")
        scores.append(score)
        embeddings.append(embedding)
        areas.append(float(fw*fh/(scaled.shape[0]*scaled.shape[1])))
        confidences.append(float(face[-1]))
    capture.release()
    if scores:
        values=np.stack(scores)
        values-=values.mean(1,keepdims=True)
        summary=np.concatenate((values.mean(0),values.std(0),values.max(0),
            [len(scores)/5,float(np.mean(face_counts)),float(np.mean(areas)),float(np.std(areas)),float(np.mean(confidences))])).astype(np.float32)
        values=np.stack(embeddings)
        deep=np.concatenate((values.mean(0),values.std(0))).astype(np.float32)
    else:
        summary=np.zeros(29,np.float32)
        deep=np.zeros(2560,np.float32)
    return {"face_summary":summary,"deep_face":deep,"coverage":np.array(len(scores)/5),
            "decoded_frames":np.array(decoded)}


def verify() -> None:
    """在固定三个开发片段上核对新组合实现与既有缓存的数值一致性。"""
    with np.load(ROOT/"_国赛优化/round5/face_features/selection.npz",allow_pickle=False) as source:
        ids=source["sample_ids"].astype(str)
        indices=[0,len(ids)//2,len(ids)-1]
        summary=source["face_summary"][indices]
    with np.load(PREVIOUS/"deep_face_features/selection.npz",allow_pickle=False) as source:
        deep=source["deep_face"][indices]
    reports=[]
    for j,i in enumerate(indices):
        value=encode(ROOT/"data/public/ch_sims_v2/supervised"/f"{ids[i]}.mp4")
        error=float(np.abs(value["face_summary"]-summary[j]).max())
        deep_error=float(np.abs(value["deep_face"]-deep[j]).max())
        reports.append({"id":ids[i],"face_summary_max_error":error,"deep_face_max_error":deep_error})
        if error>1e-4 or deep_error>1e-3:
            raise ValueError("组合人脸提取器与冻结基线不一致")
    save(OUTPUT/"external_face_encoder_verification.json",{"checks":reports,"detector_sha256":digest(DETECTOR),
        "backbone_sha256":digest(BACKBONE),"code_sha256":digest(Path(__file__))})
    print(json.dumps(reports),flush=True)


def main() -> None:
    """原始媒体整包核验通过后才提取，逐样本缓存支持断点。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    if args.verify:
        verify()
        return
    if not (OUTPUT/"emotiontalk_acquisition/complete.json").exists():
        raise ValueError("真实媒体尚未通过完整来源核验")
    output=OUTPUT/"emotiontalk_features/face_features"
    cache=output/"samples_v2_bounded"
    cache.mkdir(parents=True,exist_ok=True)
    binding=digest(DETECTOR)+digest(BACKBONE)+digest(Path(__file__))
    archived=OUTPUT/"source/media_initial/round11_external_faces.py"
    compatible=digest(DETECTOR)+digest(BACKBONE)+digest(archived)

    def process(key: str) -> dict[str,np.ndarray]:
        require_memory()
        target=cache/f"{key}.npz"
        if target.exists():
            with np.load(target,allow_pickle=False) as saved:
                if str(saved["binding"])!=binding:
                    raise ValueError("外部人脸缓存版本不符")
                return {name:saved[name] for name in ("face_summary","deep_face","coverage")}
        legacy=output/"samples"/f"{key}.npz"
        if legacy.exists():
            with np.load(legacy,allow_pickle=False) as saved:
                # 仅兼容同一数值算法下完整解码的旧缓存；OOM附近未完成五帧的条目重算。
                if str(saved["binding"])!=compatible:
                    raise ValueError("初版媒体缓存身份不符")
                if int(saved["decoded_frames"])==5:
                    return {name:saved[name] for name in ("face_summary","deep_face","coverage")}
        value=encode(ROOT/"data/public/emotiontalk_media/clips"/f"{key}.mp4")
        if int(value["decoded_frames"])!=5:
            require_memory(6.)
            repeated=encode(ROOT/"data/public/emotiontalk_media/clips"/f"{key}.mp4")
            if any(not np.array_equal(value[name],repeated[name]) for name in value):
                raise ValueError("不完整解码无法在内存充足时稳定复现，拒绝写缓存")
            value["partial_sampling_repeat_verified"]=np.array(True)
        temporary=target.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle,**value,binding=np.array(binding),sample_id=np.array(key))
        temporary.replace(target)
        return value

    started=time.monotonic()
    for split in ("train","valid","test_inputs"):
        destination=output/f"{split}.npz"
        if destination.exists():
            continue
        identifiers=[json.loads(line)["id"] for line in (OUTPUT/"corpus"/f"{split}.jsonl").read_text("utf-8").splitlines()]
        results=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0,len(identifiers),32):
                results.extend(pool.map(process,identifiers[start:start+32]))
                if start%256==0 or start+32>=len(identifiers):
                    print(json.dumps({"faces":len(results),"total":len(identifiers),"split":split,
                                      "seconds":round(time.monotonic()-started,1)}),flush=True)
        np.savez_compressed(destination,sample_ids=np.array(identifiers),
            **{name:np.stack([row[name] for row in results]) for name in ("face_summary","deep_face","coverage")})
    save(output/"manifest.json",{"binding":binding,"compatible_initial_binding":compatible,
        "execution_change":"one decoder thread;bounded32clip queues;only complete5frame initial caches reused",
        "frames":[.1,.3,.5,.7,.9],
        "identity_verified":False,"training_labels_used":False,"backbone":"frozen HSEmotion EfficientNet-B0",
        "limits":"largest face proxy,not a verified active speaker;face missing is not neutral"})


if __name__=="__main__":
    main()
