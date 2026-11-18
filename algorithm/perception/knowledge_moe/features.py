"""训练集归一化和冻结 VLM 适配；显式保留编码器身份。"""
from __future__ import annotations

import hashlib
import json
import functools
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class FeatureStandardizer:
    """只用训练样本的有效观测拟合，缺失项不参与均值和方差。"""

    def __init__(self) -> None:
        self.parameters: dict[str, dict[str, list[float]]] = {}

    def fit(self, features: Mapping[str, np.ndarray], observed: np.ndarray) -> FeatureStandardizer:
        """记录每个冻结编码器维度的训练均值和尺度。"""
        if observed.dtype != np.bool_ or observed.shape != (len(next(iter(features.values()))), len(features)):
            raise ValueError("归一化掩码维数不符")
        for j, (name, x) in enumerate(features.items()):
            values = np.asarray(x, dtype=np.float32)[observed[:, j]]
            if not len(values) or not np.isfinite(values).all():
                raise ValueError("每个模态至少需要一个有限训练观测")
            self.parameters[name] = {"mean": values.mean(axis=0).tolist(),
                                     "scale": np.maximum(values.std(axis=0), 1e-5).tolist()}
        return self

    def transform(self, features: Mapping[str, np.ndarray], observed: np.ndarray) -> dict[str, np.ndarray]:
        """使用已拟合参数转换，忽略缺失值，拒绝有效观测中的 NaN。"""
        if list(features) != list(self.parameters):
            raise ValueError("模态名称或顺序改变")
        if observed.dtype != np.bool_ or observed.shape != (len(next(iter(features.values()))), len(features)):
            raise ValueError("归一化掩码维数不符")
        output = {}
        for j, (name, x) in enumerate(features.items()):
            x = np.asarray(x, dtype=np.float32)
            mean = np.asarray(self.parameters[name]["mean"], dtype=np.float32)
            scale = np.asarray(self.parameters[name]["scale"], dtype=np.float32)
            if mean.ndim != 1 or scale.shape != mean.shape or not len(mean) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("归一化工件参数非法")
            if x.ndim != 2 or x.shape[1] != len(mean) or not np.isfinite(x[observed[:, j]]).all():
                raise ValueError("特征维度改变或有效观测包含非有限值")
            clean = np.where(observed[:, j:j+1], x, mean)
            output[name] = np.where(observed[:, j:j+1], (clean - mean) / scale, 0).astype(np.float32)
        return output


@functools.lru_cache(maxsize=1)
def get_clip_encoder(model_path: Path, device: str = "cpu") -> Any:
    """业务图文和本人文本参考共享冻结编码器，避免重复占用内存。"""
    return FrozenCLIPEncoder(model_path, device)


class FrozenCLIPEncoder:
    """可选的真实联合视觉语言预训练编码器，显式加载本地权重。

    原有 BERT/FACET/COVAREP 特征不会被改称 VLM。该适配器只有实际
    加载 CLIP 权重并对图像/文字前向后才构成 VLM 编码证据。CLIP 为
    视觉语言对比预训练模型，不将独立输出向量伪称对话推理大模型。
    """

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor, ChineseCLIPModel, ChineseCLIPProcessor

        self.model_path = model_path.resolve()
        config = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") == "chinese_clip":
            processor_cls, model_cls = ChineseCLIPProcessor, ChineseCLIPModel
        elif config.get("model_type") == "clip":
            processor_cls, model_cls = CLIPProcessor, CLIPModel
        else:
            raise ValueError("仅接受已明确适配的 CLIP / Chinese-CLIP 配置")
        self.processor = processor_cls.from_pretrained(str(self.model_path), local_files_only=True, use_fast=False)
        self.model = model_cls.from_pretrained(str(self.model_path), local_files_only=True).to(device).eval()
        self.device = device
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        digest = hashlib.sha256()
        weights = sorted([*self.model_path.glob("*.safetensors"), *self.model_path.glob("pytorch_model*.bin")])
        if not weights:
            raise ValueError("未找到可指纹校验的 VLM 权重")
        preprocessing_files = sorted(p for p in self.model_path.iterdir()
                                     if p.name in {"config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "vocab.json", "merges.txt"})
        for path in [*preprocessing_files, *weights]:
            digest.update(path.name.encode() + b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        self.fingerprint = "clip_local_sha256:" + digest.hexdigest()

    def encode(self, images: list[Any], texts: list[str]) -> dict[str, np.ndarray]:
        """对齐图像与文字的冻结前向，文字截断长度由模型决定。"""
        import torch

        if not images or len(images) != len(texts):
            raise ValueError("需要等长且非空的原始图像与文本")
        batch = self.processor(images=images, text=texts, padding=True, truncation=True,
                               max_length=self.model.config.text_config.max_position_embeddings, return_tensors="pt")
        with torch.inference_mode():
            output = self.model(**batch.to(self.device))
        return {"visual": output.image_embeds.cpu().numpy(), "text": output.text_embeds.cpu().numpy()}

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """仅文字也用同一投影与L2归一化，和图文前向的text输出一致。

        不调用上游`get_text_features`：transformers 4.57 的
        `ChineseCLIPTextModel` 不再构建pooler，该接口会对`None`做投影并报错。
        这里直接复现`ChineseCLIPModel.forward`的文字分支（CLS位置池化 →
        `text_projection` → L2），与`encode`的text输出逐位相同。
        """
        import torch
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("需要非空的文字列表")
        batch = self.processor.tokenizer(texts, padding=True, truncation=True,
                                         max_length=self.model.config.text_config.max_position_embeddings,
                                         return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.text_model(**batch)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                # 与上游forward一致：无pooler时取首个token（CLS）的隐藏状态。
                pooled = outputs.last_hidden_state[:, 0, :]
            vectors = self.model.text_projection(pooled)
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        return vectors.cpu().numpy()


def sha256_file(path: Path) -> str:
    """流式计算大文件指纹，避免将整个权重读入内存。"""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value: Any) -> None:
    """写入严格 JSON，禁止把 NaN/Infinity 伪装成有效指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

