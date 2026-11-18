"""真实原片五帧人脸表情表征；最大脸不等于已确认说话人，未检出不填中性。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import cv2
import numpy as np
import onnxruntime as ort
from perception.knowledge_moe.features import save_json,sha256_file

_local=threading.local()


def process(identifier: str,media: Path,output: Path,detector: Path,emotion: Path,binding: str) -> np.ndarray:
    """每线程独立检测器；固定统计汇聚，不使用任何情感目标标签。"""
    target=output/(identifier+".npz")
    if target.exists():
        with np.load(target,allow_pickle=False) as saved:
            if str(saved["binding"])!=binding:raise ValueError("人脸缓存版本不同")
            return saved["features"].copy()
    if not hasattr(_local,"detector"):
        _local.detector=cv2.FaceDetectorYN.create(str(detector),"",(320,320),.8,.3,5000)
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        _local.emotion=ort.InferenceSession(str(emotion),sess_options=options,providers=["CPUExecutionProvider"])
    path=media/(identifier+".mp4");cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logits=[];areas=[];confidence=[];face_counts=[];decoded=0
    for fraction in [.1,.3,.5,.7,.9]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,min(count-1,round(count*fraction))))
        ok,frame=cap.read()
        if not ok:continue
        decoded+=1;h,w=frame.shape[:2];scaled=cv2.resize(frame,(320,max(32,round(h*320/w))))
        _local.detector.setInputSize((scaled.shape[1],scaled.shape[0]));_,faces=_local.detector.detect(scaled)
        face_counts.append(0 if faces is None else len(faces))
        if faces is None or not len(faces):continue
        face=max(faces,key=lambda row:row[2]*row[3]);x,y,fw,fh=face[:4]
        sx=w/scaled.shape[1];sy=h/scaled.shape[0]
        left=max(0,int((x-.1*fw)*sx));top=max(0,int((y-.1*fh)*sy))
        right=min(w,int((x+1.1*fw)*sx));bottom=min(h,int((y+1.1*fh)*sy))
        if right<=left or bottom<=top:continue
        crop=cv2.cvtColor(cv2.resize(frame[top:bottom,left:right],(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        value=((crop-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None]
        pred=_local.emotion.run(None,{_local.emotion.get_inputs()[0].name:value})[0].reshape(-1)
        if pred.shape!=(8,) or not np.isfinite(pred).all():raise ValueError("表情模型输出无效")
        logits.append(pred);areas.append(float(fw*fh/(scaled.shape[0]*scaled.shape[1])));confidence.append(float(face[-1]))
    cap.release()
    if logits:
        scores=np.stack(logits)
        # logits居中，避免逐帧公共偏置被分类器当作情绪。
        scores-=scores.mean(1,keepdims=True)
        feature=np.concatenate([scores.mean(0),scores.std(0),scores.max(0),
                [len(logits)/5,float(np.mean(face_counts)),float(np.mean(areas)),float(np.std(areas)),float(np.mean(confidence))]]).astype(np.float32)
    else:feature=np.zeros(29,np.float32)
    np.savez_compressed(target,features=feature,binding=np.array(binding),media_sha256=np.array(sha256_file(path)),decoded_frames=np.array(decoded))
    return feature


def run(bundle: Path,media: Path,detector: Path,emotion: Path,output: Path) -> None:
    """逐ID缓存五帧人脸统计，保留原划分和可用率。"""
    output.mkdir(parents=True,exist_ok=True);cache=output/"samples";cache.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    binding=sha256_file(detector)+sha256_file(emotion)+sha256_file(Path(__file__))
    parts={}
    for split in ["train","selection","calibration","test"]:
        with np.load(bundle/(split+".npz"),allow_pickle=False) as saved:parts[split]={k:saved[k] for k in saved.files}
    identifiers=[str(identifier) for part in parts.values() for identifier in part["sample_ids"]]
    values=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index,feature in enumerate(pool.map(lambda identifier:process(identifier,media,cache,detector,emotion,binding),identifiers)):
            values.append(feature)
            if (index+1)%200==0:print(json.dumps({"face_features":index+1,"total":len(identifiers)}),flush=True)
    values=np.stack(values);cursor=0
    for split,part in parts.items():
        n=len(part["labels"]);part["face_summary"]=values[cursor:cursor+n];cursor+=n
        np.savez_compressed(output/(split+".npz"),**part)
    save_json(output/"manifest.json",{"input_sha256":sha256_file(bundle/"manifest.json"),"binding":binding,"n":len(identifiers),
        "frames":[.1,.3,.5,.7,.9],"dimension":29,"detected_clip_fraction":float((values[:,24]>0).mean()),
        "expression_class_order":["Anger","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"],
        "limitations":["largest detected face is not confirmed speaker identity","expression neutral not equivalent to multimodal sentiment neutral",
                       "face absence represented by zero values and zero coverage, never a neutral prediction"],"code_sha256":sha256_file(Path(__file__))})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["bundle","media","detector","emotion","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.media,args.detector,args.emotion,args.output)
