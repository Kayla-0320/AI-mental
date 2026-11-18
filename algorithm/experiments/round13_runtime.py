"""可调用的中文缺模态退化路径，原联合主分类器按可用性保留。"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any
import numpy as np
import torch
from scipy.special import softmax
from experiments.round11_common import OUTPUT as ROUND11,digest
from experiments.round13_data import OUTPUT
from experiments.round13_models import SentimentLexicon,predict_heads
from experiments.round13_neural import FrozenBankRouter,dispatch


def pack_raw(data: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    """从真实冻结编码器输出构造联合表示；函数不接收或读取情绪目标。"""
    from experiments.round11_features import observed
    groups = dict(T=("llm_last","aux_mean","bge_cls"),
        A=("audio","speech","speech_emotion","sensevoice","sensevoice_query","sensevoice_emotion"),
        V=("face_summary","deep_face","visual"))
    result = {k:data[k] for k in ("sample_ids","raw_text")}
    for name,fields in groups.items():
        result[name] = np.column_stack([data[f]/np.sqrt(data[f].shape[1]) for f in fields]).astype(np.float32)
    result["mask"] = np.column_stack((data["modality_mask"][:,0] & (data["quality"][:,0]>0),
        observed(data,"audio"),observed(data,"face_summary")|observed(data,"visual")))
    return result


def route(original: np.ndarray,fallback: np.ndarray,mask: np.ndarray,patterns: list[int]) -> np.ndarray:
    """仅按真实观测的可用性路由，不按预测结果或人工标签选择。"""
    if original.shape != fallback.shape or original.shape != (len(mask),3) or mask.shape != (len(mask),3):
        raise ValueError("退化路由形状不一致")
    codes = mask.astype(int) @ np.array([1,2,4])
    result = original.copy()
    active = np.isin(codes,patterns)
    result[active] = fallback[active]
    return result


class CHPredictor:
    """加载完整原7B模型与新专用头；提供可用性路由和概率重放接口。"""

    def __init__(self,folder: Path = OUTPUT/"final_ch") -> None:
        self.folder = folder
        self.manifest = json.loads((folder/"model_manifest.json").read_text("utf-8"))
        if digest(folder/"model.pkl") != self.manifest["model_sha256"]:
            raise ValueError("新增分支模型摘要不符")
        baseline = ROUND11/"final_models/7b/model.pkl"
        if digest(baseline) != self.manifest["original_model_sha256"]:
            raise ValueError("原7B模型变化，拒绝静默混用")
        with baseline.open("rb") as handle:
            self.original = pickle.load(handle)
        with (folder/"model.pkl").open("rb") as handle:
            self.bundle = pickle.load(handle)
        self.network = None
        if self.bundle["family"] == "neural":
            weights = torch.load(folder/"head.pt",map_location="cpu",weights_only=False)
            self.network = FrozenBankRouter(weights["dimensions"],weights["config"],weights["full"],weights["prior"])
            self.network.load_state_dict(weights["state_dict"],strict=True)
            self.network.eval()
        self.lexicon = SentimentLexicon()

    def fallback(self,packed: dict[str,np.ndarray]) -> np.ndarray:
        """新分支仅使用其确实可见的模态，推理无目标标签入口。"""
        projected = self.bundle["projection"].transform(packed)
        if self.network is not None:
            return dispatch(self.network,self.bundle["heads"],projected)[0]
        return predict_heads(self.bundle["heads"],projected,self.lexicon.encode(packed["raw_text"]))

    def predict(self,data: dict[str,np.ndarray]) -> tuple[np.ndarray,dict[str,Any]]:
        """从配套冻结表征到最终概率；原主模型五种子概率均值与新单头分开记录。"""
        from experiments.round11_final import predict
        original = predict(self.original,data).mean(0)
        packed = pack_raw(data)
        fallback = self.fallback(packed)
        codes = packed["mask"].astype(int) @ np.array([1,2,4])
        for code,bias in self.manifest["bias_by_pattern"].items():
            active = codes==int(code)
            fallback[active] = softmax(np.log(np.maximum(fallback[active],1e-12))+np.array(bias),axis=1)
        output = route(original,fallback,packed["mask"],self.manifest["enabled_patterns"])
        return output,dict(availability=codes.tolist(),fallback_used=np.isin(codes,self.manifest["enabled_patterns"]).tolist(),
            output_semantics="class decision probabilities; no new calibration guarantee")
