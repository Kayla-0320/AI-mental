"""冻结视觉编码器加独立画质头；KonIQ按图像严格留出，真人MOS仅作既有公开标签。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import pickle
import struct
from typing import Any
import zipfile
import zlib
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import requests
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round18_images import MODEL
from experiments.round18_natural_quality import DATA,API
from perception.knowledge_moe.features import FrozenCLIPEncoder

OUT=ROOT/"round18/quality_model"


def descriptors(rgb: np.ndarray) -> np.ndarray:
    """补充曝光、对比度、梯度与频率统计，均从真实像素计算。"""
    gray=cv2.cvtColor(cv2.resize(rgb,(128,128)),cv2.COLOR_RGB2GRAY).astype(float)/255
    lap=cv2.Laplacian(gray,cv2.CV_64F);gx=cv2.Sobel(gray,cv2.CV_64F,1,0);gy=cv2.Sobel(gray,cv2.CV_64F,0,1)
    hist=np.bincount((gray*255).astype(np.uint8).ravel(),minlength=256)/gray.size
    return np.array([gray.mean(),gray.std(),np.mean(gray<.05),np.mean(gray>.95),np.log1p(lap.var()*1000),np.mean(np.hypot(gx,gy)),
        -np.sum(hist*np.log2(hist+1e-12))/8,*np.asarray(rgb,float).std((0,1))/255],float)


def metrics(y: np.ndarray,p: np.ndarray) -> dict[str,Any]:
    """画质MOS相关性和误差，不表示心理语义或项目真人评价。"""
    return {"n":len(y),"spearman":float(spearmanr(y,p).statistic) if np.ptp(p)>0 and np.ptp(y)>0 else None,"mae_MOS_1_5":float(np.abs(y-p).mean()),"rmse_MOS_1_5":float(np.sqrt(np.mean((y-p)**2))),
        "auc_MOS_below_2_5":float(roc_auc_score(y<2.5,-p)) if len(np.unique(y<2.5))==2 else None}


def summarize_saved() -> None:
    """只重新序列化已有留出预测，常数相关性写null，不重训、不重新选型。"""
    z=np.load(OUT/"test_predictions.npz",allow_pickle=False);y=z["truth"];p=z["selected"];baseline=z["training_mean"]
    ix=np.random.default_rng(42).integers(0,len(y),(2000,len(y)));delta=np.abs(p-y)-np.abs(baseline-y)
    selected=json.loads((OUT/"selection_lock.json").read_text(encoding="utf-8"))["selected"]
    save(OUT/"results.json",{"selected":selected,"test":metrics(y,p),"training_mean":metrics(y,baseline),"mae_delta_bootstrap_ci95":np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),"test_scope":"new images of same public natural-quality source, not cross-source or emotion benefit","exact_image_hashes_unique":True,"project_human_blind_ratings":0,"summary_repair_does_not_change_predictions":True})
    print(json.dumps(json.loads((OUT/"results.json").read_text(encoding="utf-8"))),flush=True)


def run() -> None:
    """768训练、192验证选型、256新测试；先封存策略，再计算测试指标。"""
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    spec={"seed":42,"splits":[768,192,256],"exclude":"all128 prior natural-quality probe names","features":"frozen ChineseCLIP visual512 +10 observed RGB descriptors","candidates":["ridge1","ridge10","ridge100","trees"],
        "select":"validation Spearman, fit winner on train+validation, then test once","targets":"published MOS1-5; no project-human ratings","source_sha256":sha256(Path(__file__)),"no_test_tuning":True,"pretraining_image_overlap_unknown":True}
    if (OUT/"protocol.json").exists():
        original=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
        freeze(OUT/"protocol_v2.json",{**original,"source_sha256":sha256(Path(__file__)),"original_protocol_sha256":sha256(OUT/"protocol.json"),"repair":"constant correlation null serialization only, reuse saved predictions without retuning"})
        if (OUT/"test_predictions.npz").exists():summarize_saved();return
    else:freeze(OUT/"protocol.json",spec)
    meta=json.loads((ROOT/"round18/koniq_metadata.json").read_text(encoding="utf-8"))["data"]["latestVersion"]
    table=pd.read_csv(DATA/"scores.tab",sep="\t",dtype={"image_name":str});excluded=set(json.loads((ROOT/"round18/natural_quality/selected_ids.json").read_text(encoding="utf-8")))
    if table.image_name.duplicated().any() or not table.MOS.between(1,5).all():raise ValueError("公开标签重复或量表无效")
    names=sorted(set(table.image_name)-excluded,key=lambda n:hashlib.sha256(("18quality|42|"+n).encode()).digest())[:1216]
    splits={"train":names[:768],"validation":names[768:960],"test":names[960:]};freeze(OUT/"split_lock.json",splits)
    cache=OUT/"features.npz"
    if not cache.exists():
        size=next(f["dataFile"]["filesize"] for f in meta["files"] if f["dataFile"]["id"]==72286)
        r=requests.get(API+"72286",headers={"Range":f"bytes={size-65558}-{size-1}"},timeout=(15,45));r.raise_for_status()
        if r.status_code!=206:raise ValueError("分段下载不可用")
        url=r.url;tail=r.content;at=tail.rfind(b"PK\x05\x06");end=tail[at:];fields=struct.unpack("<4s4H2IH",end[:22]);cd_size,offset=fields[5:7]
        def ranged(a: int,b: int) -> bytes:
            for attempt in range(3):
                try:
                    response=requests.get(url,headers={"Range":f"bytes={a}-{b}"},timeout=(15,45));response.raise_for_status()
                    if response.status_code!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {a}-"):raise ValueError("分段不符")
                    return response.content
                except requests.RequestException:
                    if attempt==2:raise
            raise RuntimeError("分段重试失败")
        directory=ranged(offset,offset+cd_size-1);patched=end[:16]+struct.pack("<I",0)+end[20:]
        with zipfile.ZipFile(io.BytesIO(directory+patched)) as archive:infos={Path(i.filename).name:i for i in archive.infolist() if not i.is_dir()}
        def download(name: str) -> dict[str,str]:
            info=infos[name];path=DATA/name
            if not path.exists():
                content=ranged(info.header_offset,min(size-1,info.header_offset+30+len(info.filename.encode())+info.compress_size+4096));head=struct.unpack("<4s5H3I2H",content[:30])
                if head[0]!=b"PK\x03\x04":raise ValueError("本地头不符")
                start=30+head[-2]+head[-1];raw=content[start:start+info.compress_size];decoded=zlib.decompress(raw,-15) if info.compress_type==8 else raw
                if len(decoded)!=info.file_size or zlib.crc32(decoded)!=info.CRC:raise ValueError("CRC或长度不符")
                path.write_bytes(decoded)
            if zlib.crc32(path.read_bytes())!=info.CRC:raise ValueError("缓存CRC不符")
            return {"image_name":name,"sha256":sha256(path)}
        with ThreadPoolExecutor(max_workers=8) as pool:
            hashes=[]
            for i,item in enumerate(pool.map(download,names),1):
                hashes.append(item)
                if i%128==0:print(json.dumps({"downloaded_quality_images":i}),flush=True)
        save(OUT/"image_hashes.json",hashes)
        # 不以重复图像跨折虚增留出表现；检测到精确重复则停止协议，不能暗中换样。
        if len({x["sha256"] for x in hashes})!=len(hashes):raise ValueError("发现重复图像，停止而不重选测试样本")
        torch.set_num_threads(1);encoder=FrozenCLIPEncoder(MODEL,"cuda" if torch.cuda.is_available() else "cpu");features=[]
        for start in range(0,len(names),48):
            images=[]
            for name in names[start:start+48]:
                with Image.open(DATA/name) as im:images.append(im.convert("RGB"))
            vectors=encoder.encode(images,[""]*len(images))["visual"];features.append(np.column_stack((vectors,np.stack([descriptors(np.asarray(im)) for im in images]))))
        np.savez_compressed(cache,image_names=np.array(names),X=np.concatenate(features))
        save(OUT/"encoder.json",{"fingerprint":encoder.fingerprint,"audio_decoded":False,"images":len(names),"visual_features":512,"pixel_features":10})
    z=np.load(cache,allow_pickle=False);assert np.array_equal(z["image_names"],names);x=z["X"];lookup=table.set_index("image_name");y=np.array([lookup.loc[n,"MOS"] for n in names],float)
    candidates={f"ridge{a}":make_pipeline(StandardScaler(),Ridge(alpha=a)) for a in (1,10,100)}
    candidates["trees"]=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=5,max_features=.5,random_state=42,n_jobs=1)
    scores={}
    for name,model in candidates.items():
        model.fit(x[:768],y[:768]);scores[name]=metrics(y[768:960],np.clip(model.predict(x[768:960]),1,5))
    selected=max(scores,key=lambda n:scores[n]["spearman"] if scores[n]["spearman"] is not None else -1)
    freeze(OUT/"selection_lock.json",{"selected":selected,"validation_metrics":scores,"no_test_labels_used":True,"source_sha256":sha256(Path(__file__))})
    model=candidates[selected].fit(x[:960],y[:960]);p=np.clip(model.predict(x[960:]),1,5);baseline=np.full(256,float(y[:960].mean()))
    with (OUT/"model.pkl").open("wb") as stream:pickle.dump(model,stream)
    save(OUT/"manifest.json",{"model_sha256":sha256(OUT/"model.pkl"),"feature_dimension":522,"semantic_reliability_or_emotion_accuracy":False,"production_promoted":False})
    np.savez_compressed(OUT/"test_predictions.npz",image_names=np.array(names[960:]),truth=y[960:],selected=p,training_mean=baseline)
    summarize_saved()


if __name__=="__main__":run()
