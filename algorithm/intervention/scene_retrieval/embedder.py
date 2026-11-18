"""句子向量编码器 —— 场景检索的底座。

设计要点：
    1. **延迟加载**：BGE 权重 409MB，import 时不加载，首次 ``encode`` 才载入。
       避免 FastAPI 启动时被 409MB 模型拖慢。
    2. **零新增依赖**：只用项目 venv 已有的 ``transformers`` + ``torch``，
       不引入 ``sentence-transformers``。CLS pooling 手工实现，与官方
       ``1_Pooling/config.json`` 的 ``pooling_mode_cls_token=true`` 一致。
    3. **可插拔**：调用方只依赖 :class:`Embedder` 协议，本地 BGE 与
       DashScope embedding 可互换。
"""
from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Optional, Protocol, runtime_checkable

import numpy as np

# 本地 BGE 权重路径。可用环境变量覆盖，便于换模型或换机器。
_DEFAULT_BGE_PATH = r"D:\ai\models\encoders\bge-base-zh-v1.5"
DEFAULT_MODEL_PATH = os.environ.get("SCENE_EMBED_MODEL", _DEFAULT_BGE_PATH)

# BGE-zh 官方为非对称检索（短查询 → 长文档）推荐的查询指令前缀。
# 本项目两侧都是短句（用户发言 → 场景/想法例句），属于对称语义相似，
# 因此默认关闭；是否启用由离线评测决定（见 tools/eval_scene_retrieval.py）。
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

DASHSCOPE_EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"


@runtime_checkable
class Embedder(Protocol):
    """向量编码器协议。

    任何实现只需提供 ``dim`` 属性与 ``encode`` 方法即可被检索器使用。
    """

    @property
    def dim(self) -> int:
        """向量维度。"""
        ...

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        is_query: bool = False,
    ) -> np.ndarray:
        """把文本编码为 L2 归一化后的向量矩阵。

        Args:
            texts: 待编码文本。
            batch_size: 批大小。
            is_query: 是否为查询侧（可影响指令前缀）。

        Returns:
            shape ``(len(texts), dim)`` 的 float32 矩阵，每行已 L2 归一化。
        """
        ...


class BGEEmbedder:
    """本地 BGE 编码器（CLS pooling + L2 归一化）。

    默认加载 ``bge-base-zh-v1.5``（12 层 / 768 维）。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        use_query_instruction: bool = False,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_path: 本地权重目录，默认取 ``SCENE_EMBED_MODEL`` 或内置路径。
            use_query_instruction: 查询侧是否加 BGE 官方指令前缀。
            max_length: 截断长度。默认 512（该模型 max_position_embeddings）。
            device: 推理设备。本机无 CUDA，默认 cpu。
        """
        self._model_path = str(model_path or DEFAULT_MODEL_PATH)
        self._use_query_instruction = use_query_instruction
        self._max_length = max_length
        self._device = device

        self._tokenizer = None
        self._model = None
        self._dim = 768  # config.json 的 hidden_size，加载后被真实值覆盖

    @property
    def dim(self) -> int:
        """向量维度。"""
        return self._dim

    @property
    def model_path(self) -> str:
        """当前使用的权重目录。"""
        return self._model_path

    def _ensure_loaded(self) -> None:
        """首次调用时加载权重（延迟加载）。"""
        if self._model is not None:
            return

        # 延迟 import：避免 import 本模块就拉起 torch
        import torch  # noqa: F401  (供下方 forward 使用)
        from transformers import AutoModel, AutoTokenizer
        from shared.model_lock import MODEL_LOAD_LOCK

        if not os.path.isdir(self._model_path):
            raise FileNotFoundError(
                f"未找到本地编码器权重目录: {self._model_path}。"
                f"请设置环境变量 SCENE_EMBED_MODEL 指向 BGE 权重目录。"
            )

        with MODEL_LOAD_LOCK:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            model = AutoModel.from_pretrained(self._model_path)
        model.eval()
        model.to(self._device)
        self._model = model
        self._dim = int(model.config.hidden_size)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        is_query: bool = False,
    ) -> np.ndarray:
        """编码为 L2 归一化向量。

        Args:
            texts: 待编码文本。
            batch_size: 批大小。
            is_query: 查询侧标记（配合 ``use_query_instruction``）。

        Returns:
            shape ``(len(texts), dim)`` 的 float32 矩阵。
        """
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        self._ensure_loaded()

        import torch

        prefix = BGE_QUERY_INSTRUCTION if (is_query and self._use_query_instruction) else ""
        chunks: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = [prefix + str(t) for t in texts[start:start + batch_size]]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(**enc)
            # CLS pooling（与 1_Pooling/config.json 的 pooling_mode_cls_token=true 一致）
            cls = out.last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            chunks.append(cls.cpu().numpy().astype(np.float32))

        return np.vstack(chunks)


class DashScopeEmbedder:
    """阿里云 DashScope embedding（``text-embedding-v3``，1024 维）。

    作为本地 BGE 的可插拔备选：无需本地 GPU/权重，但引入网络往返与调用成本。
    批大小受服务端限制，``text-embedding-v3`` 单次上限 10 条。
    """

    def __init__(
        self,
        model: str = "text-embedding-v3",
        *,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """
        Args:
            model: 模型名。
            api_key: 显式 key；默认读环境变量 ``DASHSCOPE_API_KEY``（禁止硬编码）。
            timeout: 单次请求超时（秒）。
            max_retries: 失败重试次数。
        """
        self._model = model
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._timeout = timeout
        self._max_retries = max_retries
        self._dim = 1024 if "v3" in model else 1536

    @property
    def dim(self) -> int:
        """向量维度。"""
        return self._dim

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 10,
        is_query: bool = False,
    ) -> np.ndarray:
        """调用 DashScope 编码（返回已 L2 归一化向量）。

        Args:
            texts: 待编码文本。
            batch_size: 批大小（服务端上限 10）。
            is_query: 该后端不使用指令前缀，仅为满足协议。

        Returns:
            shape ``(len(texts), dim)`` 的 float32 矩阵。
        """
        del is_query  # 协议要求，此后端忽略
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self._api_key:
            raise RuntimeError(
                "未配置 DASHSCOPE_API_KEY，无法使用 DashScope embedding。"
                "请在 .env 或环境变量中设置。"
            )

        import requests

        vectors: list[list[float]] = []
        batch_size = min(batch_size, 10)

        for start in range(0, len(texts), batch_size):
            batch = [str(t)[:2000] for t in texts[start:start + batch_size]]
            last_err: Optional[Exception] = None

            for attempt in range(self._max_retries):
                try:
                    resp = requests.post(
                        DASHSCOPE_EMBED_URL,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self._api_key}",
                        },
                        json={"model": self._model, "input": batch},
                        timeout=self._timeout,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        vectors.extend(d["embedding"] for d in data["data"])
                        last_err = None
                        break
                    last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                except Exception as e:  # noqa: BLE001
                    last_err = e

                time.sleep(0.5 * (attempt + 1))

            if last_err is not None:
                raise RuntimeError(f"DashScope embedding 调用失败: {last_err}") from last_err

        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def get_default_embedder() -> Embedder:
    """返回默认编码器（本地 BGE）。"""
    return BGEEmbedder()
