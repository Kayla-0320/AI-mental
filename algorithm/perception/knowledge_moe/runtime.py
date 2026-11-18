"""指纹匹配后才能使用的独立任务推理器，不接管线上五类情绪。"""
from __future__ import annotations

import json
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from perception.knowledge_moe.features import FeatureStandardizer, sha256_file
from perception.knowledge_moe.model import ConcatenationHead, QualityAwareMoE
from perception.knowledge_moe.sparse import SparseQualityAwareMoE
from perception.knowledge_moe.knowledge import KnowledgeIndex
from shared import MultimodalFeatureBatch, MultimodalPrediction


def _validate_manifest(value: Any) -> dict[str, Any]:
    """在分配模型内存之前验证工件形状、语义和资源范围。"""
    if not isinstance(value, dict) or value.get("schema_version") != "knowledge_moe.v1":
        raise ValueError("模型工件必须是 knowledge_moe.v1 对象")
    m = value
    if not isinstance(m.get("task"), str) or not m["task"].strip():
        raise ValueError("模型工件缺少非空任务名")
    classes = m.get("class_names")
    if (not isinstance(classes, list) or not 2 <= len(classes) <= 256
            or not all(isinstance(c, str) and c.strip() for c in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("类别必须是2至256个唯一非空名称")
    dimensions = m.get("dimensions")
    if (not isinstance(dimensions, dict) or not 1 <= len(dimensions) <= 8
            or not all(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
                       and type(size) is int and 1 <= size <= 4096 for name, size in dimensions.items())
            or sum(dimensions.values()) > 16384):
        raise ValueError("模型模态/维度工件非法或超出研究头资源范围")
    if type(m.get("hidden")) is not int or not 2 <= m["hidden"] <= 512:
        raise ValueError("非法隐藏维度")
    if m.get("architecture") not in ("moe", "sparse_moe", "concat"):
        raise ValueError("未知模型架构")
    knowledge_dim = m.get("knowledge_dim", 0)
    if type(knowledge_dim) is not int or not 0 <= knowledge_dim <= 4096:
        raise ValueError("非法知识维度")
    if knowledge_dim and (m.get("knowledge_review_mode") != "user_attested_only" or "text" not in dimensions
                          or not isinstance(m.get("knowledge_sha256"), str)
                          or not re.fullmatch(r"[0-9a-f]{64}", m["knowledge_sha256"])):
        raise ValueError("知识实验工件尚不可用于业务推理，需完成专家审核与知识输入契约")
    count = len(dimensions)
    experts = count + count * (count - 1) // 2 + int(count > 2) + int(knowledge_dim > 0)
    top_k = m.get("top_k", 0)
    if type(top_k) is not int or not 0 <= top_k <= experts or (m["architecture"] == "concat" and top_k):
        raise ValueError("非法路由 top_k 工件")
    if m["architecture"] == "sparse_moe" and top_k < 1:
        raise ValueError("稀疏分派必须明确top_k")
    fingerprints = m.get("encoder_fingerprints")
    if (not isinstance(fingerprints, dict) or set(fingerprints) != set(dimensions)
            or not all(isinstance(v, str) and v.strip() for v in fingerprints.values())):
        raise ValueError("编码器身份必须与模态逐项对应")
    digest = m.get("checkpoint_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("非法权重摘要")
    parameters = m.get("standardizer")
    if not isinstance(parameters, dict) or set(parameters) != set(dimensions):
        raise ValueError("归一化工件必须与模态逐项对应")
    for name, size in dimensions.items():
        p = parameters[name]
        if not isinstance(p, dict):
            raise ValueError("归一化工件必须包含mean/scale对象")
        for field in ("mean", "scale"):
            values = p.get(field)
            if (not isinstance(values, list) or len(values) != size
                    or not all(type(v) in (int, float) and -3.4028234663852886e38 <= v <= 3.4028234663852886e38 for v in values)):
                raise ValueError("归一化工件的维度或有限值非法")
            if field == "scale" and any(v < 1.1754943508222875e-38 for v in values):
                raise ValueError("归一化工件scale必须为正数")
    temperature = m.get("temperature")
    # 当前训练器的拟合范围是[0.1,10]；此处保留更宽的可表示研究范围。
    if type(temperature) not in (int, float) or not .01 <= temperature <= 100:
        raise ValueError("非法温度或超出数值适用范围")
    calibration = m.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("status") not in ("calibrated", "uncalibrated"):
        raise ValueError("校准状态必须明确且合法")
    if calibration["status"] == "calibrated":
        if calibration.get("fit_split") != "calibration":
            raise ValueError("温度必须在独立校准集拟合")
        if type(calibration.get("n")) is not int or calibration["n"] <= 0:
            raise ValueError("已校准工件必须记录正整数校准样本量")
    elif temperature != 1:
        raise ValueError("未校准工件温度必须为1，不能静默套用缩放")
    if "text_only_calibration" in m:
        text_cal=m["text_only_calibration"]
        if not isinstance(text_cal,dict) or text_cal.get("status") not in ("calibrated","uncalibrated","not_applicable"):
            raise ValueError("非法文字条件校准状态")
        if text_cal["status"]=="calibrated":
            t=text_cal.get("temperature")
            if (text_cal.get("fit_split")!="calibration" or type(text_cal.get("n")) is not int or text_cal["n"]<=0
                    or type(t) not in (int,float) or not .01<=t<=100
                    or text_cal.get("condition")!="text_present_audio_visual_missing" or "text" not in dimensions):
                raise ValueError("文字条件温度缺少独立校准证据或适用条件")
    return m


class FusionRuntime:
    """校验任务、类别、冻结编码器及权重来源；全缺失时显式弃权。"""

    def __init__(self, artifact_dir: Path) -> None:
        try:
            raw = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("无法读取合法模型工件JSON") from exc
        self.meta = _validate_manifest(raw)
        m = self.meta
        weights = artifact_dir / "weights.pt"
        try:
            actual_digest = sha256_file(weights)
        except OSError as exc:
            raise ValueError("无法读取模型权重") from exc
        if actual_digest != m["checkpoint_sha256"]:
            raise ValueError("权重摘要不匹配")
        self.knowledge = None
        if m.get("knowledge_dim", 0):
            hybrid=m.get("knowledge_vectors_sha256")
            self.knowledge = KnowledgeIndex(artifact_dir / "knowledge.json", dimensions=32 if hybrid else m["knowledge_dim"], allow_user_attested=True)
            if self.knowledge.sha256 != m["knowledge_sha256"]:
                raise ValueError("知识工件指纹不匹配")
            if not self.knowledge.attestation_verified:
                raise ValueError("知识工件没有与节点绑定的用户背书")
            if hybrid:
                if (not isinstance(hybrid,str) or not re.fullmatch(r"[0-9a-f]{64}",hybrid)
                        or m.get("knowledge_vector_encoder_fingerprint")!=m.get("live_text_encoder_fingerprint")
                        or not isinstance(m.get("live_text_encoder_fingerprint"),str)
                        or m["encoder_fingerprints"]["text"]!=m["live_text_encoder_fingerprint"]+";component=text_l2_v1"):
                    raise ValueError("知识向量未绑定同一冻结文字编码器")
                path=artifact_dir/"knowledge_vectors.npz"
                if sha256_file(path)!=hybrid:raise ValueError("知识语义向量摘要不符")
                try:
                    with np.load(path,allow_pickle=False) as z:
                        ids,values=z["node_ids"],z["vectors"]
                        if ids.ndim!=1 or values.ndim!=2 or len(ids)!=len(values) or len(set(ids.tolist()))!=len(ids):
                            raise ValueError("知识向量未与唯一节点ID对齐")
                        vectors={str(k):v for k,v in zip(ids,values)}
                    self.knowledge.attach_semantic_vectors(vectors)
                except (OSError,ValueError,TypeError,KeyError,zipfile.BadZipFile) as exc:
                    raise ValueError("知识向量工件损坏") from exc
                if self.knowledge.dimensions!=m["knowledge_dim"]:
                    raise ValueError("知识语义向量维度与模型不符")
        try:
            if m["architecture"] == "moe":
                self.model = QualityAwareMoE(m["dimensions"], len(m["class_names"]), m["hidden"], m.get("knowledge_dim", 0), top_k=m.get("top_k", 0))
                self.expert_names = self.model.expert_names
            elif m["architecture"] == "sparse_moe":
                self.model = SparseQualityAwareMoE(m["dimensions"], len(m["class_names"]), m["hidden"], m.get("knowledge_dim", 0), top_k=m["top_k"])
                self.expert_names = self.model.expert_names
            else:
                self.model = ConcatenationHead(m["dimensions"], len(m["class_names"]), m["hidden"], m.get("knowledge_dim", 0))
                self.expert_names = ["concatenation"]
            self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError, AssertionError,
                UnicodeError, OverflowError, EOFError, pickle.UnpicklingError) as exc:
            raise ValueError("模型权重内容与工件架构不匹配或损坏") from exc
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("模型权重包含非有限值")
        self.model.eval()
        self.standardizer = FeatureStandardizer()
        self.standardizer.parameters = m["standardizer"]
        self.temperature = float(m["temperature"])

    def predict(self, batch: MultimodalFeatureBatch) -> MultimodalPrediction:
        """独立研究任务预测；任务或编码器不符时拒绝，缺失样本不给假概率。"""
        m = self.meta
        if batch.task != m["task"] or batch.class_names != m["class_names"]:
            raise ValueError("任务或类别口径不匹配")
        if batch.encoder_fingerprints != m["encoder_fingerprints"]:
            raise ValueError("冻结编码器身份不匹配")
        names = list(m["dimensions"])
        n = len(batch.sample_ids)
        if not n or len(set(batch.sample_ids)) != n:
            raise ValueError("样本ID必须非空且唯一")
        if any(set(x) != set(names) for x in [batch.modalities, batch.observed, batch.quality]):
            raise ValueError("输入模态集合不匹配")
        observed_values = np.array([batch.observed[name] for name in names]).T
        if observed_values.dtype != np.bool_ or observed_values.shape != (n, len(names)):
            raise ValueError("观测掩码必须是与样本对齐的布尔值")
        quality = np.array([batch.quality[name] for name in names], dtype=np.float32).T
        if quality.shape != observed_values.shape or not np.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
            raise ValueError("非法模态质量")
        observed = observed_values & (quality > 0)
        features = {name: np.asarray(batch.modalities[name], dtype=np.float32) for name in names}
        if any(x.shape != (n, m["dimensions"][name]) for name, x in features.items()):
            raise ValueError("冻结特征维度不匹配")
        valid = observed.any(axis=1)
        probs: list[list[float]] = [[] for _ in range(n)]
        gates: list[list[float]] = [[] for _ in range(n)]
        if valid.any():
            x = self.standardizer.transform({name: value[valid] for name, value in features.items()}, observed[valid])
            knowledge_values = knowledge_present = None
            if self.knowledge is not None:
                raw_text = getattr(batch, "raw_text", [])
                if len(raw_text) != n or any(not isinstance(t, str) or len(t) > 8192 for t in raw_text):
                    raise ValueError("知识工件需要与样本对齐且有界的原文")
                values, present = self.knowledge.encode(raw_text)
                present &= observed[:, names.index("text")]
                knowledge_values, knowledge_present = torch.tensor(values[valid]), torch.tensor(present[valid])
            with torch.inference_mode():
                logits, weights, _ = self.model({name: torch.tensor(value) for name, value in x.items()},
                                               torch.tensor(observed[valid]), torch.tensor(quality[valid]),
                                               knowledge_values, knowledge_present)
                if not torch.isfinite(logits).all() or not torch.isfinite(weights).all():
                    raise ValueError("模型计算产生非有限输出，拒绝发布概率")
                p = torch.softmax(logits / self.temperature, dim=1).tolist()
            for i, row, gate in zip(np.where(valid)[0], p, weights.tolist()):
                probs[i], gates[i] = row, gate
        return MultimodalPrediction(task=batch.task, class_names=batch.class_names, probabilities=probs,
                                   expert_weights=gates, expert_names=self.expert_names, sample_ids=batch.sample_ids,
                                   calibration_status=m.get("calibration", {}).get("status", "uncalibrated"),
                                   model_id=m["checkpoint_sha256"], abstained=(~valid).tolist(),
                                   warnings=["独立语料任务结果，不能解释为心理诊断或临床预警"])
