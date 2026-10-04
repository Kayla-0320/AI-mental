# -*- coding: utf-8 -*-
"""本地语音识别引擎 —— sherpa-onnx Paraformer（中文，CPU 推理）

设计目标：把「语音转文字」从浏览器 Web Speech API 换成完全本地的能力。

为什么换掉 Web Speech API：
    它并不是本地 API。微软官方策略文档 `SpeechRecognitionEnabled` 原文：
    "The Microsoft Edge implementation of the Web Speech API uses Azure
    Cognitive Services, so voice data leaves the machine"；Chrome 走谷歌云。
    实测抓到的端点是：
        wss://speech.platform.bing.com/speech/recognition/edge/interactive/v1
    后果有三：① 依赖外网，校园网/防火墙下随时失效（且旧实现会在 onend 里
    零延迟重启，把一次瞬时失败放大成永久报错风暴）；② 未成年人音频离开设备，
    与「敏感原始数据不出设备」冲突；③ 浏览器差异大（Firefox 没有该 API）。

本实现：
    sherpa-onnx 1.13.8 + Paraformer-zh-small（int8，约 78 MB），纯 CPU onnxruntime。
    实测 RTF ≈ 0.008（5.6 秒音频识别约 46 ms），延迟远低于分片间隔。

模型获取：
    本机 huggingface.co 与 GitHub release 资产均不可达，必须走 hf-mirror 镜像，
    且该镜像对非浏览器 UA 返回 403。用 `algorithm/tools/download_asr_model.py`
    下载（已内置 UA 处理）。

注意：
    sherpa-onnx 的 `OfflineRecognizer.from_paraformer()` **没有 am.mvn 参数**
    —— CMVN 已烘焙进导出的 ONNX。模型仓库里那份 `am.mvn` 是 ModelScope 原始
    导出遗留，本引擎不消费（保留它无害，仅为与上游仓库一致）。
"""
from __future__ import annotations

import logging
import os
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from shared.dataclasses import AsrEngine, AsrResult

logger = logging.getLogger(__name__)

# ── 常量 ────────────────────────────────────────────────────────────────────
MODEL_NAME = "sherpa-onnx-paraformer-zh-small-2024-03-09"
SAMPLE_RATE = 16000
FEATURE_DIM = 80
#: 判定模型目录「可用」的最小文件集（am.mvn 非必需，见模块 docstring）
REQUIRED_FILES = ("model.int8.onnx", "tokens.txt")

#: <repo>/algorithm/asr/engine.py -> parents[2] == <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_ROOT = _REPO_ROOT / "algorithm" / "models" / "asr"


# ── 模型定位 ────────────────────────────────────────────────────────────────
def _usable(d: Path) -> bool:
    """目录是否含全部必需文件"""
    return d.is_dir() and all((d / f).is_file() for f in REQUIRED_FILES)


def resolve_model_dir(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """定位 Paraformer 模型目录。

    查找顺序：
        1. 显式传入的路径
        2. 环境变量 ``ASR_MODEL_DIR``
        3. ``<repo>/algorithm/models/asr/<MODEL_NAME>``
        4. ``<repo>/algorithm/models/asr/`` 下第一个含必需文件的子目录
           （便于更换模型而不用改代码）

    Returns:
        可用的模型目录；都找不到时返回 None（调用方应据此报告「不可用」，
        而不是静默降级 —— 静默降级正是本项目此前多处问题的根源）。
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("ASR_MODEL_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(_DEFAULT_MODEL_ROOT / MODEL_NAME)

    for c in candidates:
        if _usable(c):
            return c

    root = _DEFAULT_MODEL_ROOT
    if root.is_dir():
        for sub in sorted(root.iterdir()):
            if _usable(sub):
                return sub
    return None


# ── WAV 解码（只用标准库，与前端 encodeWav16 的输出格式对应）────────────────
def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """解析 WAV 字节流 → (float32 单声道 [-1,1], 采样率)。

    支持 16bit / 32bit-int / 32bit-float PCM，单声道与多声道（多声道取均值）。
    前端 `localAsr.encodeWav16()` 产出的是 16bit 单声道，此处多分支只为兼容
    外部工具产生的样本（如模型自带的 test_wavs）。
    """
    with wave.open(_BytesReader(data), "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    if sampwidth == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        # WAVE 格式码 3 = IEEE float32，1 = PCM int32；用 wave 时拿不到格式码，
        # 这里按数值范围启发式判断：int32 的绝对值普遍远大于 1。
        as_i4 = np.frombuffer(raw, dtype="<i4")
        as_f4 = np.frombuffer(raw, dtype="<f4")
        peak = float(np.max(np.abs(as_i4))) if as_i4.size else 0.0
        pcm = as_f4.astype(np.float32) if peak <= 2 ** 24 else as_i4.astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        # 8bit WAV 是无符号
        pcm = (np.frombuffer(raw, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"不支持的位深: {sampwidth * 8}bit")

    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), rate


class _BytesReader:
    """让标准库 `wave` 能直接读内存里的 WAV 字节流。"""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._data) - self._pos
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        else:
            self._pos = len(self._data) + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def close(self) -> None:
        pass


# ── 识别器 ──────────────────────────────────────────────────────────────────
class ParaformerRecognizer:
    """sherpa-onnx Paraformer 识别器的进程级单例。

    模型约 78 MB，加载一次后常驻；首次调用时才加载（懒加载），避免拖慢服务启动。
    `transcribe()` 加锁串行化 —— 单次识别 RTF ≈ 0.008，串行化的代价可以忽略，
    换来的是不必假设底层 recognizer 对并发 decode 安全。
    """

    _instance: Optional["ParaformerRecognizer"] = None
    _instance_lock = threading.Lock()

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir if model_dir is not None else resolve_model_dir()
        self._recognizer = None
        self._load_error: str = ""
        self._decode_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ParaformerRecognizer":
        """取得进程级单例"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ---- 状态 ----
    @property
    def available(self) -> bool:
        """模型目录存在且引擎可加载"""
        if self.model_dir is None:
            return False
        return self.load()

    @property
    def load_error(self) -> str:
        """最近一次加载失败原因"""
        return self._load_error

    @property
    def engine(self) -> AsrEngine:
        """当前实际可用的引擎标识"""
        return AsrEngine.PARAFORMER if self.available else AsrEngine.UNAVAILABLE

    # ---- 加载 ----
    def load(self) -> bool:
        """懒加载模型；返回是否可用。失败原因记在 `load_error`。"""
        if self._recognizer is not None:
            return True
        with self._instance_lock:
            if self._recognizer is not None:
                return True
            if self.model_dir is None:
                self._load_error = (
                    f"未找到 ASR 模型目录。请运行 "
                    f"`python algorithm/tools/download_asr_model.py`，"
                    f"或设置环境变量 ASR_MODEL_DIR 指向含 {REQUIRED_FILES} 的目录。"
                )
                logger.warning("[ASR] %s", self._load_error)
                return False
            try:
                import sherpa_onnx  # 延迟导入：未装 sherpa-onnx 时服务仍可启动
            except ImportError as exc:
                self._load_error = f"未安装 sherpa-onnx：{exc}"
                logger.warning("[ASR] %s", self._load_error)
                return False

            try:
                t0 = time.perf_counter()
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    paraformer=str(self.model_dir / "model.int8.onnx"),
                    tokens=str(self.model_dir / "tokens.txt"),
                    num_threads=min(4, os.cpu_count() or 2),
                    sample_rate=SAMPLE_RATE,
                    feature_dim=FEATURE_DIM,
                    decoding_method="greedy_search",
                    debug=False,
                    provider="cpu",
                )
                logger.info(
                    "[ASR] Paraformer 加载完成（%.0f ms）：%s",
                    (time.perf_counter() - t0) * 1000,
                    self.model_dir.name,
                )
                self._load_error = ""
                return True
            except Exception as exc:  # noqa: BLE001 —— 加载失败必须可诊断，不能吞
                self._recognizer = None
                self._load_error = f"Paraformer 加载失败：{exc}"
                logger.exception("[ASR] %s", self._load_error)
                return False

    # ---- 识别 ----
    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> AsrResult:
        """识别一段音频。

        Args:
            samples: float32 单声道波形，取值 [-1, 1]
            sample_rate: 采样率；非 16k 时交给 sherpa-onnx 内部重采样

        Returns:
            AsrResult —— 失败时 `engine` 为 UNAVAILABLE 且 `error` 非空，
            绝不返回看似成功的空结果。
        """
        duration_ms = float(len(samples)) / float(sample_rate or SAMPLE_RATE) * 1000.0

        if not self.load():
            return AsrResult(
                text="",
                engine=AsrEngine.UNAVAILABLE,
                duration_ms=duration_ms,
                error=self._load_error or "ASR 引擎不可用",
            )

        if samples.size == 0:
            return AsrResult(text="", engine=AsrEngine.PARAFORMER, duration_ms=0.0)

        try:
            t0 = time.perf_counter()
            with self._decode_lock:
                stream = self._recognizer.create_stream()
                stream.accept_waveform(int(sample_rate), samples)
                self._recognizer.decode_stream(stream)
                text = stream.result.text or ""
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return AsrResult(
                text=text.strip(),
                engine=AsrEngine.PARAFORMER,
                duration_ms=duration_ms,
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[ASR] 识别失败")
            return AsrResult(
                text="",
                engine=AsrEngine.PARAFORMER,
                duration_ms=duration_ms,
                error=f"识别失败：{exc}",
            )

    def status(self) -> dict[str, object]:
        """供 /health 与调试端点使用的状态摘要"""
        ok = self.load() if self.model_dir is not None else False
        return {
            "available": ok,
            "engine": self.engine.value,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "model_name": self.model_dir.name if self.model_dir else None,
            "sample_rate": SAMPLE_RATE,
            "error": self._load_error,
        }
