"""开发侧新旧语音与输入质量的公平对照，折内拟合全部数值变换。"""
from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from experiments import round10_large_speech as previous
from experiments.round11_common import OUTPUT

AUDIO_PACKS = {
    "msp": ("audio", "speech", "speech_emotion"),
    "sense": ("audio", "sensevoice", "sensevoice_query", "sensevoice_emotion"),
    "both": ("audio", "speech", "speech_emotion", "sensevoice", "sensevoice_query", "sensevoice_emotion"),
}
TEXT = ("llm_last", "aux_mean", "bge_cls")
VISUAL = ("face_summary", "deep_face", "visual")


def alignment_feature(reference: str, prediction: str) -> np.ndarray:
    """文字与真实ASR的表面一致度代理，不作情感或同步真值。"""
    clean = re.sub(r"<\|.*?\|>", "", prediction)
    left = "".join(c.lower() for c in reference if c.isalnum())
    right = "".join(c.lower() for c in clean if c.isalnum())
    ratio = SequenceMatcher(None, left, right, autojunk=False).ratio() if left and right else 0.
    return np.array([ratio, min(len(left),len(right))/max(1,len(left),len(right)),
                     float(bool(left and right))], np.float64)


def load_data(quality: str = "legacy") -> dict[str, np.ndarray]:
    """仅加载train和selection，编码存在与质量放行分别保存。"""
    data = previous.data_for(("train", "selection"))
    keys = data["sample_ids"].astype(str)
    for directory in ("msp_complete_features", "sensevoice_deterministic_features"):
        pieces = []
        for split in ("train", "selection"):
            with np.load(OUTPUT/directory/f"{split}.npz", allow_pickle=False) as source:
                pieces.append({key: source[key] for key in source.files})
        if not np.array_equal(np.concatenate([part["sample_ids"] for part in pieces]), keys):
            raise ValueError("第十一轮语音ID未对齐")
        if directory == "msp_complete_features":
            data["speech"] = np.concatenate([part["speech"] for part in pieces])
            data["speech_emotion"] = np.concatenate([part["speech_emotion"] for part in pieces])
            data["speech_available"] = np.concatenate([part["available"] for part in pieces])
        else:
            data["sensevoice"] = np.concatenate([part["sensevoice"] for part in pieces])
            data["sensevoice_query"] = np.concatenate([part["query"][:,1,:] for part in pieces])
            data["sensevoice_emotion"] = np.concatenate([part["emotion_logprobs"][:,1,:] for part in pieces])
            data["sensevoice_available"] = np.concatenate([part["available"] for part in pieces])
            text = np.concatenate([part["transcript"] for part in pieces])
            data["asr_alignment"] = np.stack([alignment_feature(str(a),str(b)) for a,b in zip(data["raw_text"], text)])
    if quality == "pts":
        with np.load(OUTPUT/"input_audit/development_masks.npz", allow_pickle=False) as source:
            if not np.array_equal(keys, source["sample_ids"]):
                raise ValueError("修正质量ID未对齐")
            data["modality_mask"] = source["modality_mask"]
            data["quality"] = source["quality"]
    elif quality != "legacy":
        raise ValueError("未知质量方案")
    return data


def observed(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    """冻结编码缓存存在不能覆盖媒体质量拒绝。"""
    if name.startswith("sensevoice"):
        return data["modality_mask"][:,1] & (data["quality"][:,1] > 0) & data["sensevoice_available"]
    if name.startswith("speech"):
        return data["modality_mask"][:,1] & (data["quality"][:,1] > 0) & data["speech_available"]
    if name == "audio":
        return data["modality_mask"][:,1] & (data["quality"][:,1] > 0)
    if name == "visual":
        return data["modality_mask"][:,2] & (data["quality"][:,2] > 0)
    if name == "deep_face":
        return data["coverage"] > 0
    if name == "face_summary":
        return data["face_summary"][:,24] > 0
    if name == "asr_alignment":
        return observed(data, "sensevoice") & data["modality_mask"][:,0]
    return np.ones(len(data["sample_ids"]), bool)


def transform(data: dict[str, np.ndarray], fields: tuple[str, ...],
              fitted: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """均值填补、标准化、FP64 PCA均只见训练折，缺失模态输出显式零与mask。"""
    fitting = fitted is None
    fitted = {} if fitting else fitted
    blocks = []
    for name in fields:
        mask = observed(data, name)
        values = np.asarray(data[name], np.float64)
        if values.ndim != 2 or not np.isfinite(values[mask]).all():
            raise ValueError("有效表征必须为有限二维数组")
        if fitting:
            if not mask.any():
                raise ValueError("训练折不存在有效模态")
            scaler = StandardScaler().fit(values[mask])
            pca = None
            if name == "deep_face":
                pca = PCA(n_components=min(32,mask.sum()-1,values.shape[1]), whiten=True,
                          svd_solver="randomized", random_state=42).fit(scaler.transform(values[mask]))
            fitted[name] = {"scaler": scaler, "pca": pca}
        scaler, pca = fitted[name]["scaler"], fitted[name]["pca"]
        block = scaler.transform(np.where(mask[:,None], values, scaler.mean_))
        if pca is not None:
            block = pca.transform(block)
        block = np.where(mask[:,None], block, 0.)/np.sqrt(block.shape[1])
        blocks.extend((block,mask[:,None].astype(np.float64)*.1))
    return np.column_stack(blocks), fitted
