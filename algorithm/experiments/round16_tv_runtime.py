"""仅文本/视觉的冻结专家离线运行时，导出时剔除全部音频投影和专家。"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np
from experiments.round14_common import ROOT,save,sha256
from experiments.round16_moe import PATTERNS,mix
from experiments.round13_models import probabilities,matrix

OUT=ROOT/"round16/moe"


def predict_tv(text: np.ndarray,visual: np.ndarray,mask: np.ndarray,folder: Path,method: str="uniform") -> np.ndarray:
    """输入为匹配冻结编码器的表征和T/V可用性；不接收音频、标签或来源ID。"""
    if mask.shape!=(len(text),2) or len(visual)!=len(text):raise ValueError("T/V输入形状不一致")
    if mask.dtype!=bool:raise ValueError("可用性必须是布尔值")
    for values,active in ((text,mask[:,0]),(visual,mask[:,1])):
        if values.ndim!=2 or not np.isfinite(values[active]).all():raise ValueError("可见表征包含无效值")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    for name,digest in manifest["files"].items():
        if sha256(folder/name)!=digest:raise ValueError("离线模型文件摘要不一致")
    with (folder/"projection.pkl").open("rb") as stream:projection=pickle.load(stream)
    with (folder/"experts.pkl").open("rb") as stream:bundle=pickle.load(stream)
    with (folder/"gate.pkl").open("rb") as stream:gate=pickle.load(stream)
    if projection.members["A"] is not None or set(bundle["models"])!=set(PATTERNS):raise ValueError("导出包含不允许的专家或模态")
    availability=np.column_stack((mask[:,0],np.zeros(len(mask),bool),mask[:,1]))
    packed={"T":text,"V":visual,"A":np.zeros((len(mask),1)),"mask":availability}
    projected=projection.transform(packed)
    if method=="ordinary_TV":return probabilities(bundle["models"][5],matrix(projected,5,None))
    if method not in {"uniform","learned","anchored"}:raise ValueError("未知融合方法")
    bank=np.stack([probabilities(bundle["models"][p],matrix(projected,p,None)) for p in PATTERNS],axis=1)
    return mix(bank,availability,gate,method,bundle["prior"])


def export_verify() -> dict[str,Any]:
    """裁剪导出四折模型并复核三条件全部已存预测，证明无音频依赖。"""
    from experiments.round13_train import ch_data
    from experiments.round10_fusion import folds
    import shutil
    data=ch_data();expected={c:np.load(OUT/(c+"_predictions.npz"),allow_pickle=False) for c in ("TV","T_only","V_only")}
    result={}
    for fold,(_,va) in enumerate(folds(data)):
        folder=OUT/f"export/fold{fold}";folder.mkdir(parents=True,exist_ok=True)
        with (ROOT/f"round13/training/ch/fold{fold}/projection.pkl").open("rb") as stream:projection=pickle.load(stream)
        projection.members["A"]=None
        with (ROOT/f"round13/training/ch/linear_1/fold{fold}/model.pkl").open("rb") as stream:bundle=pickle.load(stream)
        bundle={**bundle,"models":{p:bundle["models"][p] for p in PATTERNS},"full":5}
        with (folder/"projection.pkl").open("wb") as stream:pickle.dump(projection,stream)
        with (folder/"experts.pkl").open("wb") as stream:pickle.dump(bundle,stream)
        shutil.copy2(OUT/f"gate_fold{fold}.pkl",folder/"gate.pkl")
        save(folder/"manifest.json",{"files":{name:sha256(folder/name) for name in ("projection.pkl","experts.pkl","gate.pkl")},"experts":PATTERNS,
            "audio_present":False,"semantics":"M3 negative/neutral/positive; matched frozen features only","input_dimensions":{"T":data["T"].shape[1],"V":data["V"].shape[1]}})
        for condition in expected:
            mask=data["mask"][va][:,[0,2]].copy()
            if condition=="T_only":mask[:,1]=False
            if condition=="V_only":mask[:,0]=False
            for method in ("ordinary_TV","uniform","learned","anchored"):
                p=predict_tv(data["T"][va],data["V"][va],mask,folder,method)
                delta=float(np.max(np.abs(p-expected[condition][method][va])))
                if delta>1e-12:raise ValueError("导出运行时预测与研究结果不一致")
                result[f"fold{fold}/{condition}/{method}"]={"rows":len(va),"max_error":delta}
        if fold==0:
            sample=va[:16];mask=data["mask"][sample][:,[0,2]]
            np.savez_compressed(OUT/"export/smoke_input.npz",T=data["T"][sample],V=data["V"][sample],mask=mask,
                expected_uniform=predict_tv(data["T"][sample],data["V"][sample],mask,folder))
    save(OUT/"export_verification.json",result);print(json.dumps({"comparisons":len(result),"max_error":max(v["max_error"] for v in result.values())}),flush=True)
    return result


if __name__=="__main__":export_verify()
