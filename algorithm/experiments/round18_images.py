"""真实RGB变换后重新编码的图文质量实验，不复用未受损视觉向量冒充故障。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
from typing import Any
import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round13_train import ch_data
from experiments.round10_fusion import folds
from experiments.round17_pixels import frames,quality_from_frames
from experiments.round17_quality import MEDIA,score
from perception.knowledge_moe.features import FrozenCLIPEncoder

OUT=ROOT/"round18/images"
CONDITIONS=("clean","black","dark","blur","center_occlusion")
MODEL=Path(r"D:\ai\models\encoders\chinese-clip-vit-base-patch16")


def altered(image: np.ndarray,condition: str) -> np.ndarray:
    """固定真实像素变换，所有比较方法共享同一重新编码结果。"""
    if condition=="clean":return image
    if condition=="black":return np.zeros_like(image)
    if condition=="dark":return (image*.03).astype(np.uint8)
    if condition=="blur":return np.stack([cv2.GaussianBlur(f,(31,31),9) for f in image])
    if condition=="center_occlusion":
        value=image.copy();value[:,24:104,24:104]=127;return value
    raise ValueError("未知图像条件")


def run() -> None:
    """ID哈希固定512片、来源分组四折、三个头共享相同训练样本和特征。"""
    import hashlib
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/"protocol.json",{"seed":42,"n":512,"conditions":CONDITIONS,"model":str(MODEL),"quality_rule":"visible quality<=.51 drops V; fixed before classification, not tuned on challenge",
        "heads":"T,V,TV linear C1 balanced, train clean only, same rows; per-fold scalers/PCA train only",
        "scope":"old CH source development, controlled RGB corruption, not natural occlusion labels or full old model comparison",
        "feature_schema":"existing frozen7B text + actual ChineseCLIP 512 visual mean; no old face block substituted","source_sha256":sha256(Path(__file__))})
    data=ch_data();order=sorted(range(len(data["labels"])),key=lambda i:hashlib.sha256(("18|42|"+str(data["sample_ids"][i])).encode()).digest())[:512]
    ix=np.array(order);ids=data["sample_ids"][ix];texts=data["raw_text"][ix];target=data["labels"][ix];groups=data["source_groups"][ix];T=data["T"][ix]
    cache=OUT/"reencoded.npz"
    if not cache.exists():
        with ThreadPoolExecutor(max_workers=6) as pool:images=list(pool.map(lambda i:frames(MEDIA/(str(i)+".mp4")),ids))
        torch.set_num_threads(1);encoder=FrozenCLIPEncoder(MODEL,"cuda" if torch.cuda.is_available() else "cpu")
        encoded={};qualities={};observed=np.array([len(x)>0 for x in images]);counts=np.array([len(x) for x in images])
        for condition in CONDITIONS:
            visual=np.zeros((len(ids),512),np.float32);q=np.zeros(len(ids));flat=[];flat_text=[];owners=[]
            for i,image in enumerate(images):
                if not len(image):continue
                value=altered(image,condition);q[i]=quality_from_frames(value).quality_score
                for f in value:flat.append(Image.fromarray(f));flat_text.append(str(texts[i]));owners.append(i)
            pieces=[]
            for start in range(0,len(flat),48):pieces.append(encoder.encode(flat[start:start+48],flat_text[start:start+48])["visual"])
            vectors=np.concatenate(pieces);owners=np.array(owners)
            for i in np.flatnonzero(observed):
                value=vectors[owners==i].mean(0);visual[i]=value/max(float(np.linalg.norm(value)),1e-9)
            encoded[condition]=visual;qualities[condition+"_quality"]=q
            print(json.dumps({"RGB_reencoded":condition,"frames":len(flat)}),flush=True)
        np.savez_compressed(cache,sample_ids=ids,labels=target,groups=groups,T=T,observed=observed,frame_counts=counts,**encoded,**qualities)
        save(OUT/"encoder.json",{"fingerprint":encoder.fingerprint,"actual_image_frames":int(counts.sum()),"conditions":5,"audio_decoded":False})
    z=np.load(cache,allow_pickle=False);assert np.array_equal(z["sample_ids"],ids)
    outputs={c:{m:np.zeros((len(ids),3)) for m in ("ordinary_TV","uniform","quality")} for c in CONDITIONS};fold_audit=[]
    for fold,(tr,va) in enumerate(folds({"labels":target,"source_groups":groups})):
        scalers={};projectors={};train_blocks={};held={}
        for name,raw in (("T",z["T"]),("V",z["clean"])):
            scaler=StandardScaler().fit(raw[tr]);scaled=scaler.transform(raw[tr]);pca=PCA(n_components=min(64,len(tr)-1,scaled.shape[1]),svd_solver="randomized",random_state=42).fit(scaled)
            scalers[name]=scaler;projectors[name]=pca;train_blocks[name]=pca.transform(scaled)/np.sqrt(raw.shape[1])
        matrices=[train_blocks["T"],train_blocks["V"],np.column_stack((train_blocks["T"],train_blocks["V"]))]
        heads=[LogisticRegression(C=1.,class_weight="balanced",max_iter=1000,random_state=42).fit(x,target[tr]) for x in matrices]
        for condition in CONDITIONS:
            t=projectors["T"].transform(scalers["T"].transform(z["T"][va]))/np.sqrt(z["T"].shape[1])
            v=projectors["V"].transform(scalers["V"].transform(z[condition][va]))/np.sqrt(z[condition].shape[1])
            bank=np.stack([h.predict_proba(x) for h,x in zip(heads,(t,v,np.column_stack((t,v))),strict=True)],1)
            outputs[condition]["ordinary_TV"][va]=bank[:,2];outputs[condition]["uniform"][va]=bank.mean(1)
            automatic=bank.mean(1);drop=z[condition+"_quality"][va]<=.51;automatic[drop]=bank[drop,0];outputs[condition]["quality"][va]=automatic
        with (OUT/f"fold{fold}.pkl").open("wb") as stream:pickle.dump({"scalers":scalers,"projectors":projectors,"heads":heads},stream)
        fold_audit.append({"fold":fold,"train_groups":np.unique(groups[tr]).tolist(),"held_groups":np.unique(groups[va]).tolist()})
    result={}
    for c,methods in outputs.items():
        result[c]={m:score(target,p) for m,p in methods.items()};result[c]["visual_drop_count"]=int(np.sum(z[c+"_quality"]<=.51))
        np.savez_compressed(OUT/(c+"_predictions.npz"),truth=target,groups=groups,**methods)
    save(OUT/"fold_audits.json",fold_audit);save(OUT/"results.json",{"conditions":result,"n":len(ids),"raw_decode_unavailable":int(np.sum(~z["observed"])),"classification_visual_reencoded":True,
        "natural_bad_quality_gold":False,"comparison_with_round17_strong_classifier_valid":False})
    print(json.dumps(result),flush=True)


if __name__=="__main__":run()
