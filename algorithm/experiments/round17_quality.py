"""真实原片的非语义质量测量与来源内锁定的专家降级，不把情绪分歧当质量。"""
from __future__ import annotations
import hashlib
import json
import pickle
from pathlib import Path
import re
from typing import Any
import cv2
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round13_train import ch_data
from experiments.round10_fusion import folds
from data_quality.raw_signals import assess_visual_signal

OUT=ROOT/"round17/quality"
MEDIA=Path(r"D:\Develop\AIC\data\public\ch_sims_v2\supervised")


def text_usable(text: str) -> bool:
    """只识别空白、链接和标点，不按长度、情绪、否定或图文一致性判断质量。"""
    value=re.sub(r"https?://\S+","",str(text))
    return any(c.isalnum() for c in value)


def availability(mask: np.ndarray,texts: np.ndarray,quality: np.ndarray,threshold: float) -> np.ndarray:
    """保持观测资格；质量未测得不擅自认定损坏；阈值不涉及风险或升级。"""
    result=mask.copy()
    if threshold<0:return result
    result[:,0]&=np.array([text_usable(t) for t in texts])
    result[:,2]&=~(np.isfinite(quality)&(quality<=threshold))
    result[:,1]=False
    return result


def route(bank: np.ndarray,mask: np.ndarray,prior: np.ndarray) -> np.ndarray:
    """只混合实际合格的T/V/TV专家，不基于预测分歧或测试标签切换。"""
    code=mask.astype(int)@np.array([1,2,4]);allowed=np.column_stack([((code&p)==p) for p in (1,4,5)])
    weights=allowed/np.maximum(allowed.sum(1,keepdims=True),1)
    p=(weights[:,:,None]*bank).sum(1);p[~allowed.any(1)]=prior
    return p


def read_frames(path: Path) -> np.ndarray:
    """只解码三帧RGB，不打开音轨；返回128像素缩略帧用于可观察质量测量。"""
    if not path.exists():return np.empty((0,128,128,3),np.uint8)
    cap=cv2.VideoCapture(str(path));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=[]
    for fraction in (.2,.5,.8):
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,round((total-1)*fraction)));ok,frame=cap.read()
        if ok:frames.append(cv2.cvtColor(cv2.resize(frame,(128,128)),cv2.COLOR_BGR2RGB))
    cap.release();return np.stack(frames) if frames else np.empty((0,128,128,3),np.uint8)


def measured_quality(data: dict[str,np.ndarray]) -> np.ndarray:
    """测量全开发原片并保留逐条记录；小规模真实帧故障探针仅评价检测器。"""
    path=OUT/"raw_quality.json";records=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    probes=[];probe_ids=set(sorted(data["sample_ids"].astype(str),key=lambda x:hashlib.sha256(("42|"+x).encode()).digest())[:48])
    for i,identifier in enumerate(data["sample_ids"].astype(str)):
        if identifier in records:continue
        frames=read_frames(MEDIA/(identifier+".mp4"))
        if not len(frames):records[identifier]={"measured":False,"score":None,"reason":"raw_frames_unavailable_not_assumed_bad"}
        else:
            q=assess_visual_signal(frames);records[identifier]={"measured":True,"score":q.quality_score,
                "frame_sha256":hashlib.sha256(frames.tobytes()).hexdigest(),"metrics":q.metrics,"reasons":q.reason_codes}
            if identifier in probe_ids:
                variants={"clean":frames,"black":np.zeros_like(frames),"dark":(frames.astype(float)*.03).astype(np.uint8),
                    "blur":np.stack([cv2.GaussianBlur(f,(31,31),9) for f in frames]),"center_occlusion":frames.copy()}
                variants["center_occlusion"][:,24:104,24:104]=127
                for condition,value in variants.items():
                    qp=assess_visual_signal(value);probes.append({"id":identifier,"condition":condition,"score":qp.quality_score,"reasons":qp.reason_codes})
        if (i+1)%100==0:
            save(path,records);print(json.dumps({"measured":i+1,"total":len(data["sample_ids"])}),flush=True)
    save(path,records)
    if probes:save(OUT/"detector_probes.json",probes)
    return np.array([records[str(identifier)]["score"] if records[str(identifier)]["measured"] else np.nan for identifier in data["sample_ids"]],float)


def score(y: np.ndarray,p: np.ndarray) -> dict[str,float]:
    """固定三类宏F1和全覆盖准确率，不删除困难样本。"""
    return {"macro_f1":float(f1_score(y,p.argmax(1),labels=[0,1,2],average="macro",zero_division=0)),"accuracy":float(accuracy_score(y,p.argmax(1)))}


def run() -> None:
    """像素质量仅作门控证据，分类评测使用未改动原片表征；故障探针不伪装分类实验。"""
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/"protocol.json",{"seed":42,"thresholds":[-1.,0.,.25,.51],"threshold_fit":"source-inner-OOF only; clean accuracy >= baseline-.005; prefer less switching on ties",
        "classification":"natural original clips and existing source-held predictions; previously exposed development diagnostic",
        "fault_probes":"48 hash-selected original clips, actual RGB transformations; detector only, NOT re-encoded classifier results",
        "disagreement_is_quality":False,"audio":"no audio decoded or experts used","source_sha256":sha256(Path(__file__))})
    data=ch_data();data["mask"][:,1]=False;qualities=measured_quality(data)
    old={c:np.load(ROOT/f"round16/moe/{c}_predictions.npz",allow_pickle=False) for c in ("TV","T_only","V_only")}
    bank=np.stack((old["T_only"]["uniform"],old["V_only"]["uniform"],old["TV"]["ordinary_TV"]),1)
    output=np.zeros((len(data["labels"]),3));locks=[];used_mask=data["mask"].copy()
    for fold,(tr,va) in enumerate(folds(data)):
        cache=np.load(ROOT/f"round13/neural/ch/fold{fold}/inner_oof.npz",allow_pickle=False)["missing_A"][:,[0,3,4],:]
        with (ROOT/f"round16/moe/export/fold{fold}/experts.pkl").open("rb") as stream:prior=pickle.load(stream)["prior"]
        base=score(data["labels"][tr],route(cache,data["mask"][tr],prior));scores=[]
        for threshold in (-1.,0.,.25,.51):
            m=availability(data["mask"][tr],data["raw_text"][tr],qualities[tr],threshold);p=route(cache,m,prior)
            metric=score(data["labels"][tr],p);switch=float(np.mean(np.any(m!=data["mask"][tr],1)))
            scores.append({"threshold":threshold,**metric,"switch_rate":switch})
        eligible=[r for r in scores if r["accuracy"]>=base["accuracy"]-.005]
        winner=max(eligible,key=lambda r:(r["macro_f1"]-.001*r["switch_rate"],-r["switch_rate"]))
        m=availability(data["mask"][va],data["raw_text"][va],qualities[va],winner["threshold"])
        output[va]=route(bank[va],m,prior);used_mask[va]=m
        locks.append({"fold":fold,"winner":winner,"scores":scores,"train_groups":np.unique(data["source_groups"][tr]).tolist(),"held_groups":np.unique(data["source_groups"][va]).tolist()})
    save(OUT/"selection_locks.json",locks)
    baseline=old["TV"]["uniform"];y=data["labels"]
    np.savez_compressed(OUT/"predictions.npz",truth=y,groups=data["source_groups"],sample_ids=data["sample_ids"],baseline=baseline,selected=output,used_mask=used_mask,original_mask=data["mask"],quality=qualities)
    changed=np.any(used_mask!=data["mask"],1)
    result={"baseline":score(y,baseline),"selected":score(y,output),"observations_measured":int(np.isfinite(qualities).sum()),"switch_count":int(changed.sum()),
        "rescued":int(np.sum((baseline.argmax(1)!=y)&(output.argmax(1)==y))),"introduced_errors":int(np.sum((baseline.argmax(1)==y)&(output.argmax(1)!=y))),
        "all_samples_kept":len(y),"fault_probe_classifier_reencoded":False,"quality_not_semantic_consistency":True}
    save(OUT/"results.json",result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":run()
