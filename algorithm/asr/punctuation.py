# -*- coding: utf-8 -*-
"""中文标点恢复 —— sherpa-onnx CT-Transformer 标点模型

为什么需要单独一个模块：
    流式 zipformer 与离线 Paraformer **都不输出标点**。没有这一段，连续几段语音
    会连成一片："首先呢就是这一轮全球金融动量的表现想谈三个问题…"，读起来不像
    手机语音输入。实测补标点后：

        我最近老是睡不着觉白天特别累有时候觉得活着没什么意思
          → 我最近老是睡不着觉，白天特别累，有时候觉得活着没什么意思。

位置：只在**定稿**时跑一次，不参与流式路径 —— 单句 1~5 ms，对"边说边出字"的
延迟没有任何影响。中间结果保持原样（手机语音输入也是这样：上屏时先出字，
停顿后才补标点）。

代价：模型 294 MB（比两个 ASR 模型加起来还大）。这是目前唯一可用的中文标点
模型，取舍是"多 294 MB 磁盘换可读的定稿文本"。若磁盘紧张，删掉该模型即可 ——
本模块会如实报告不可用，识别本身不受影响（只是没有标点）。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PUNCT_MODEL_NAME = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
PUNCT_REQUIRED_FILES = ("model.onnx",)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PUNCT_ROOT = _REPO_ROOT / "algorithm" / "models" / "punct"


def _usable(d: Path) -> bool:
    """目录是否含必需文件"""
    return d.is_dir() and all((d / f).is_file() for f in PUNCT_REQUIRED_FILES)


def resolve_punct_model_dir(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """定位标点模型目录（顺序：显式 → 环境变量 → 默认 → 扫描子目录）"""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("ASR_PUNCT_MODEL_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(_DEFAULT_PUNCT_ROOT / PUNCT_MODEL_NAME)

    for c in candidates:
        if _usable(c):
            return c

    if _DEFAULT_PUNCT_ROOT.is_dir():
        for sub in sorted(_DEFAULT_PUNCT_ROOT.iterdir()):
            if _usable(sub):
                return sub
    return None


class PunctuationRestorer:
    """标点恢复器的进程级单例（懒加载）。

    设计原则：**标点是增强，不是依赖**。模型缺失/加载失败时 `restore()` 原样返回
    输入文本，绝不抛异常、也绝不把文本变成空 —— 识别结果本身必须保住。
    """

    _instance: Optional["PunctuationRestorer"] = None
    _instance_lock = threading.Lock()

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir if model_dir is not None else resolve_punct_model_dir()
        self._punct = None
        self._load_error: str = ""
        self._load_attempted = False
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "PunctuationRestorer":
        """取得进程级单例"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        """模型目录存在且已成功加载"""
        return self.model_dir is not None and self.load()

    @property
    def load_error(self) -> str:
        """最近一次加载失败原因"""
        return self._load_error

    def load(self) -> bool:
        """懒加载模型；只在首次调用时真正加载"""
        if self._punct is not None:
            return True
        if self._load_attempted and self._punct is None:
            return False
        with self._instance_lock:
            if self._punct is not None:
                return True
            if self.model_dir is None:
                self._load_attempted = True
                self._load_error = (
                    f"未找到标点模型目录。请运行 "
                    f"`python algorithm/tools/download_asr_model.py --preset punct-zh-en`，"
                    f"或设置 ASR_PUNCT_MODEL_DIR。识别仍可用，只是定稿文本没有标点。"
                )
                logger.info("[ASR:punct] %s", self._load_error)
                return False
            try:
                import sherpa_onnx
            except ImportError as exc:
                self._load_attempted = True
                self._load_error = f"未安装 sherpa-onnx：{exc}"
                return False

            self._load_attempted = True
            try:
                t0 = time.perf_counter()
                model_config = sherpa_onnx.OfflinePunctuationModelConfig(
                    ct_transformer=str(self.model_dir / "model.onnx"),
                    num_threads=min(2, os.cpu_count() or 1),
                    debug=False,
                    provider="cpu",
                )
                self._punct = sherpa_onnx.OfflinePunctuation(
                    sherpa_onnx.OfflinePunctuationConfig(model=model_config)
                )
                logger.info(
                    "[ASR:punct] 标点模型加载完成（%.0f ms）：%s",
                    (time.perf_counter() - t0) * 1000,
                    self.model_dir.name,
                )
                self._load_error = ""
                return True
            except Exception as exc:  # noqa: BLE001 —— 标点失败不影响识别
                self._punct = None
                self._load_error = f"标点模型加载失败：{exc}"
                logger.warning("[ASR:punct] %s", self._load_error)
                return False

    def restore(self, text: str) -> str:
        """给一段文本补标点。

        失败、模型缺失、或文本过短时**原样返回**输入 —— 调用方不需要 try。
        """
        stripped = (text or "").strip()
        if not stripped:
            return text
        if not self.load():
            return text
        try:
            with self._lock:  # CT-Transformer 的会话不保证并发安全，串行化代价极小
                out = self._punct.add_punctuation(stripped)
            out = (out or "").strip()
            return out if out else text
        except Exception:  # noqa: BLE001
            logger.exception("[ASR:punct] 补标点失败，原样返回")
            return text

    def status(self) -> dict[str, object]:
        """供 /asr/status 使用的状态摘要"""
        return {
            "available": self.available,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "error": self._load_error,
        }
