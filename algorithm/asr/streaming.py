# -*- coding: utf-8 -*-
"""本地流式语音识别 —— sherpa-onnx streaming zipformer（中文，CPU）

与 `engine.py`（离线整段 Paraformer）的分工：
    engine.py     整段识别：VAD 攒完一句再送，延迟 = 句末 + ~40ms，准一点
    本模块         流式识别：边说边出字，延迟 ~200-300ms，像手机语音输入

为什么换流式：非流式模型（OfflineRecognizer）物理上出不了中间结果 ——
必须先攒完整段才能解码。要做到"边说边出字"，模型本身必须是 Online 的。

模型选择：
    `sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23`，共 25.8 MB，中文专用。
    对比 streaming-paraformer-bilingual-zh-en（226 MB）与
    streaming-zipformer-bilingual-zh-en（189 MB），这个体积小一个数量级，
    实测每 100ms 音频块解码 1.6 ms（RTF≈0.016），完全跟得上实时。

端点检测（定稿时机）：
    sherpa-onnx 内置三条规则，构造时配置：
        rule1  一直没有语音时，静音多久算一段        → 1.6s
        rule2  说过话之后，静音多久定稿（手机语音输入约 1s）→ 0.8s
        rule3  单段最长，防止无限增长                → 15s
    实测：5.15s 语音 + 尾部静音，端点在 5.90s 触发，即静音 0.75s 后定稿。

线程模型：
    每个 WebSocket 连接一个 `StreamingSession`（各持一条 OnlineStream）；
    `OnlineRecognizer` 进程内共享。onnxruntime 的会话对并发 Run() 是安全的，
    per-stream 状态在 stream 对象里，因此不同会话可以并发解码，不需要全局锁。
    模型加载本身加锁（只做一次）。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from asr.engine import ParaformerRecognizer
from asr.punctuation import PunctuationRestorer
from shared.dataclasses import AsrEngine, AsrEventType, StreamingAsrUpdate

logger = logging.getLogger(__name__)

# ── 模型 ────────────────────────────────────────────────────────────────────
STREAMING_MODEL_NAME = "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
ENCODER_FILE = "encoder-epoch-99-avg-1.int8.onnx"
DECODER_FILE = "decoder-epoch-99-avg-1.int8.onnx"
JOINER_FILE = "joiner-epoch-99-avg-1.int8.onnx"
TOKENS_FILE = "tokens.txt"

STREAMING_REQUIRED_FILES = (ENCODER_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE)

SAMPLE_RATE = 16000
FEATURE_DIM = 80

#: 定稿纠错的最短音频：太短的片段复识别收益低，还白付一次推理
MIN_REFINE_SAMPLES = int(SAMPLE_RATE * 0.5)

# ── 端点规则（秒）────────────────────────────────────────────────────────────
RULE1_MIN_TRAILING_SILENCE = 1.6   # 一直没有语音：静音多久算一段
RULE2_MIN_TRAILING_SILENCE = 0.8   # 说过话之后：静音多久定稿
RULE3_MIN_UTTERANCE_LENGTH = 15.0  # 单段最长

# ── 整段录入模式（聊天页语音输入）────────────────────────────────────────────
#: 中间结果（边说边上字）的最小刷新间隔
PARTIAL_MIN_INTERVAL_S = 0.7
#: 两次中间结果之间至少要新增这么多音频，避免同一段音频反复解码
PARTIAL_MIN_NEW_AUDIO_S = 0.4
#: 中间结果最多解码到多长：更长的音频中间结果收益低，纯烧 CPU
PARTIAL_MAX_AUDIO_S = 30.0
#: 一次定稿最多解码多长音频（超出只取尾部，避免异常长的连接把内存吃光）
FINALIZE_MAX_AUDIO_S = 300.0
#: 判定"这段音频里确实有人在说话"的单块能量阈值（100ms 块 RMS）
SPEECH_BLOCK_RMS = 0.02
#: 需要多少个 100ms 块超阈才算有语音（0.3s）
SPEECH_MIN_BLOCKS = 3
#: 块长（样本数）：16kHz × 100ms，与前端 worklet 一致
SPEECH_BLOCK_SAMPLES = SAMPLE_RATE // 10

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_ROOT = _REPO_ROOT / "algorithm" / "models" / "asr"


def _usable(d: Path) -> bool:
    """目录是否含全部必需文件"""
    return d.is_dir() and all((d / f).is_file() for f in STREAMING_REQUIRED_FILES)


def resolve_streaming_model_dir(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """定位流式模型目录。

    查找顺序：
        1. 显式传入
        2. 环境变量 ``ASR_STREAMING_MODEL_DIR``
        3. ``<repo>/algorithm/models/asr/<STREAMING_MODEL_NAME>``
        4. ``<repo>/algorithm/models/asr/`` 下第一个含全部必需文件的子目录

    Returns:
        可用目录；找不到返回 None（调用方据此上报「不可用」，不静默降级）。
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("ASR_STREAMING_MODEL_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(_DEFAULT_MODEL_ROOT / STREAMING_MODEL_NAME)

    for c in candidates:
        if _usable(c):
            return c

    root = _DEFAULT_MODEL_ROOT
    if root.is_dir():
        for sub in sorted(root.iterdir()):
            if _usable(sub):
                return sub
    return None


class StreamingRecognizer:
    """`OnlineRecognizer` 的进程级单例（懒加载）"""

    _instance: Optional["StreamingRecognizer"] = None
    _instance_lock = threading.Lock()

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir if model_dir is not None else resolve_streaming_model_dir()
        self._recognizer = None
        self._load_error: str = ""

    @classmethod
    def instance(cls) -> "StreamingRecognizer":
        """取得进程级单例"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

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
        return AsrEngine.ZIPFORMER_STREAMING if self.available else AsrEngine.UNAVAILABLE

    def load(self) -> bool:
        """懒加载模型；返回是否可用"""
        if self._recognizer is not None:
            return True
        with self._instance_lock:
            if self._recognizer is not None:
                return True
            if self.model_dir is None:
                self._load_error = (
                    f"未找到流式 ASR 模型目录。请运行 "
                    f"`python algorithm/tools/download_asr_model.py "
                    f"--preset streaming-zipformer-zh-14m`，"
                    f"或设置 ASR_STREAMING_MODEL_DIR 指向含 {STREAMING_REQUIRED_FILES} 的目录。"
                )
                logger.warning("[ASR:stream] %s", self._load_error)
                return False
            try:
                import sherpa_onnx  # 延迟导入：未装 sherpa-onnx 时服务仍可启动
            except ImportError as exc:
                self._load_error = f"未安装 sherpa-onnx：{exc}"
                logger.warning("[ASR:stream] %s", self._load_error)
                return False

            try:
                t0 = time.perf_counter()
                md = self.model_dir
                self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=str(md / TOKENS_FILE),
                    encoder=str(md / ENCODER_FILE),
                    decoder=str(md / DECODER_FILE),
                    joiner=str(md / JOINER_FILE),
                    num_threads=min(4, os.cpu_count() or 2),
                    sample_rate=SAMPLE_RATE,
                    feature_dim=FEATURE_DIM,
                    enable_endpoint_detection=True,
                    rule1_min_trailing_silence=RULE1_MIN_TRAILING_SILENCE,
                    rule2_min_trailing_silence=RULE2_MIN_TRAILING_SILENCE,
                    rule3_min_utterance_length=RULE3_MIN_UTTERANCE_LENGTH,
                    decoding_method="greedy_search",
                    provider="cpu",
                )
                logger.info(
                    "[ASR:stream] 流式 zipformer 加载完成（%.0f ms）：%s",
                    (time.perf_counter() - t0) * 1000,
                    md.name,
                )
                self._load_error = ""
                return True
            except Exception as exc:  # noqa: BLE001 —— 加载失败必须可诊断
                self._recognizer = None
                self._load_error = f"流式 zipformer 加载失败：{exc}"
                logger.exception("[ASR:stream] %s", self._load_error)
                return False

    def new_session(self, finalize_on_endpoint: bool = True) -> Optional["StreamingSession"]:
        """为一条新连接创建会话；引擎不可用时返回 None

        Args:
            finalize_on_endpoint: True = 说话停顿即自动定稿（通话页，一句一轮）；
                False = 整段录入（聊天页语音输入），只有用户主动结束才定稿。
        """
        if not self.load():
            return None
        return StreamingSession(
            self._recognizer, self.engine, finalize_on_endpoint=finalize_on_endpoint
        )

    def status(self) -> dict[str, object]:
        """供 /asr/status 与调试端点使用的状态摘要"""
        ok = self.load() if self.model_dir is not None else False
        return {
            "available": ok,
            "engine": self.engine.value,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "model_name": self.model_dir.name if self.model_dir else None,
            "sample_rate": SAMPLE_RATE,
            "endpoint_rules": {
                "rule1_trailing_silence_s": RULE1_MIN_TRAILING_SILENCE,
                "rule2_trailing_silence_s": RULE2_MIN_TRAILING_SILENCE,
                "rule3_max_utterance_s": RULE3_MIN_UTTERANCE_LENGTH,
            },
            "error": self._load_error,
        }


class StreamingSession:
    """一条流式识别会话 —— 对应一个 WebSocket 连接。

    两种结束语义（由 `finalize_on_endpoint` 选择）：

    **端点模式**（`finalize_on_endpoint=True`，通话页）
        说话停顿即自动定稿，一句一轮。用法：
            sess = recognizer.new_session()
            for chunk in mic_chunks:
                for upd in sess.accept_pcm16(chunk):
                    push_to_client(upd)
            upd = sess.finalize()             # 挂断时的残余

    **整段录入模式**（`finalize_on_endpoint=False`，聊天页语音输入）
        说话停顿**不定稿**，整段音频一直累积；中间结果改由离线大模型对已累积的
        音频周期性解码（边说边上字），只有用户主动结束才定稿。
        为什么：12s、句间停顿 1.2s 的实测里，端点切段的每一段都错
        （"重点呢"→"重点来"、"动荡"→"动量"），整段识别则全对 —— 见
        `tools/_probe_asr_mode.py`。断句不是识别需求，只是通话页的交互需求。
    """

    def __init__(
        self,
        recognizer,
        engine: AsrEngine = AsrEngine.ZIPFORMER_STREAMING,
        *,
        finalize_on_endpoint: bool = True,
    ) -> None:
        self._rec = recognizer
        self._engine = engine
        self._finalize_on_endpoint = finalize_on_endpoint
        self._stream = None
        self._last_text = ""
        self._segment_samples = 0   # 当前这一段已喂入的样本数（端点后归零）
        self._total_samples = 0     # 本次连接累计
        # 当前这一段的原始波形，用于定稿时交给离线大模型复识别。
        # 端点模式下每个端点后清空；整段录入模式下从开始累积到结束。
        # 内存代价：60s × 16kHz × float32 ≈ 3.8 MB/连接，可接受。
        self._segment_audio: list[np.ndarray] = []

        # ── 整段录入模式的中间结果节流 ──
        self._last_partial_at = 0.0
        self._partial_audio_samples = 0
        #: 超阈块累计与音频不足一块时的余数，供 `_has_speech()` 判定"有没有人说话"
        self._speech_blocks = 0
        self._rms_carry = np.zeros(0, dtype=np.float32)
        #: 上一次中间结果文本来自离线模型，因此 `self._stream` 的上下文是"另一条路"，
        #: 定稿时必须用离线结果，否则会回退到较差的流式文本。
        self._prefer_offline = False

    # ---- 内部 ----
    def _ensure_stream(self):
        """确保本段有流。**不清音频缓冲** —— 见 `_start_new_segment`。"""
        if self._stream is None:
            self._stream = self._rec.create_stream()
            self._last_text = ""
            self._prefer_offline = False
            self._last_partial_at = 0.0
            self._partial_audio_samples = 0
        return self._stream

    def _start_new_segment(self) -> None:
        """开一段全新的音频：清空这一段的波形与计数（端点定稿后调用）。

        ⚠️ 与 `_ensure_stream` 分开是必须的：整段录入模式下音频是**先 append
        再建流**，若建流顺带清缓冲，第一块音频会被当场抹掉。
        """
        self._stream = None
        self._last_text = ""
        self._segment_samples = 0
        self._segment_audio = []
        self._prefer_offline = False
        self._speech_blocks = 0
        self._rms_carry = np.zeros(0, dtype=np.float32)

    def _decode_offline(self, audio: np.ndarray) -> tuple[str, bool]:
        """用离线大模型解码一段音频。

        Returns:
            (文本, 本次是否真的用上了离线模型)。
            区分这两种失败很重要：不可用时要回落流式结果并留痕，而"模型在、
            只是这段没解出字"不该被当成模型缺失。
        """
        try:
            rec = ParaformerRecognizer.instance()
        except Exception:  # noqa: BLE001 —— 离线模型不可用不应影响识别链路
            return "", False
        if not rec.available:
            return "", False
        try:
            result = rec.transcribe(audio, SAMPLE_RATE)
        except Exception:  # noqa: BLE001 —— 解码异常不能让整条链路失败
            logger.exception("[ASR:stream] 离线解码异常")
            return "", False
        if result.error:
            logger.info("[ASR:stream] 离线解码失败：%s", result.error)
            return "", False
        return (result.text or ""), True

    def _refine(self, stream_text: str) -> tuple[str, AsrEngine, bool]:
        """定稿后处理流水线：离线复识别纠错 → 标点恢复。

        两段都可失败而不影响定稿：
          * 离线模型缺失 → 保留流式文本
          * 标点模型缺失 → 保留无标点文本
        任一环节都不会把文本变成空。
        """
        text, engine, refined = self._refine_asr(stream_text)
        # 两个 ASR 模型都不输出标点，定稿时补一次（实测 1~5 ms）
        return PunctuationRestorer.instance().restore(text), engine, refined

    def _refine_asr(self, stream_text: str) -> tuple[str, AsrEngine, bool]:
        """定稿纠错：把整段音频交给离线大模型（78 MB Paraformer）重新识别。

        为什么值得做：流式用的是 14M 小模型（低延迟的代价是精度），离线用的是
        78 MB 模型，实测同一段音频后者更准，而且整段一次性解码能保住被停顿切开
        会丢掉的上下文（"动荡"→"动量"）。两个模型都已加载，纠错只是多一次
        RTF≈0.008 的推理（12 秒音频约 90 ms），发生在定稿那一刻。

        `_prefer_offline`：中间结果已经用离线模型出过文本时，即使这次离线解码
        失败也不能回落到流式文本 —— 那会让用户看到刚上屏的字被换成另一句。

        Returns:
            (文本, 实际产出该文本的引擎, 是否用上了离线模型)。
        """
        has_audio = bool(self._segment_audio) and self._segment_samples >= MIN_REFINE_SAMPLES
        if not has_audio:
            return stream_text, self._engine, self._prefer_offline

        # 没有人说话就不要解码：纯静音会被模型编出一句流利的假话
        # （实测 3s 数字静音 →「我们我我的我啊。」），绝不能让它进输入框
        if not self._has_speech():
            logger.info("[ASR:stream] 本段未检测到语音（%.2fs），跳过识别",
                        self._segment_samples / SAMPLE_RATE)
            return "", AsrEngine.PARAFORMER, True

        try:
            audio = np.concatenate(self._segment_audio)
        except ValueError:
            return stream_text, self._engine, self._prefer_offline

        limit = int(FINALIZE_MAX_AUDIO_S * SAMPLE_RATE)
        if audio.size > limit:
            logger.warning(
                "[ASR:stream] 本段音频 %.1fs 超过上限 %.0fs，只取尾部解码",
                audio.size / SAMPLE_RATE,
                FINALIZE_MAX_AUDIO_S,
            )
            audio = audio[-limit:]

        text, used_offline = self._decode_offline(audio)
        if not used_offline:
            # 离线不可用：如实回落，并在日志里留痕
            logger.info("[ASR:stream] 离线复识别不可用，使用流式结果")
            return stream_text, self._engine, self._prefer_offline
        if not text:
            # 模型在、但整段没解出字。若中间结果也是离线出的，说明流式那条路
            # 根本没收过音频（整段录入模式不喂流式模型），此时回落到 `stream_text`
            # 等于把用户说的话整段丢掉 —— 宁可返回空（前端会当成"没说话"）。
            if self._prefer_offline:
                return "", AsrEngine.PARAFORMER, True
            return stream_text, self._engine, False

        if text != stream_text:
            logger.debug("[ASR:stream] 定稿纠错: %r -> %r", stream_text, text)
        return text, AsrEngine.PARAFORMER, True

    def _drain(self) -> float:
        """把当前可解码的帧全部解掉，返回耗时（毫秒）"""
        t0 = time.perf_counter()
        stream = self._stream
        while self._rec.is_ready(stream):
            self._rec.decode_stream(stream)
        return (time.perf_counter() - t0) * 1000.0

    def _accumulated(self, max_seconds: float | None = None) -> np.ndarray | None:
        """已累积的整段波形；`max_seconds` 限制解码长度（只取尾部）。"""
        if not self._segment_audio:
            return None
        try:
            audio = np.concatenate(self._segment_audio)
        except ValueError:
            return None
        if max_seconds is not None:
            limit = int(max_seconds * SAMPLE_RATE)
            if audio.size > limit:
                audio = audio[-limit:]
        return audio

    def _has_speech(self) -> bool:
        """本段音频里是否确实有人在说话（至少 SPEECH_MIN_BLOCKS 个 100ms 块超阈）。

        为什么需要它：声学模型**不是**"有没有人说话"的判别器 —— 把纯静音喂给
        离线 Paraformer，它会解码出一句流利的假话（实测 3s 数字静音 →
        「我们我我的我啊。」）。整段录入模式下音频是由用户点结束才定稿的，
        误点/开麦不说话都会走到这里，所以定稿前必须自己判一次有没有语音，
        不能指望模型吐空串。

        阈值标定见 `tools/_probe_asr_energy.py`：真人语音块 RMS 峰值 0.20~0.23、
        -35dBFS 白噪 0.018，取 0.02 作单块阈值并要求 3 块（0.3s）超阈 ——
        随机底噪不会连着 3 块超阈，而真实语音有 10 倍余量。
        """
        return self._speech_blocks >= SPEECH_MIN_BLOCKS

    def _note_audio_for_speech(self, samples: np.ndarray) -> None:
        """按 100ms 分块累计超阈块数（与前端 worklet 的分块口径一致）"""
        if samples.size:
            self._rms_carry = np.concatenate([self._rms_carry, samples])
        n_blocks = self._rms_carry.size // SPEECH_BLOCK_SAMPLES
        if n_blocks == 0:
            return
        blocks = self._rms_carry[: n_blocks * SPEECH_BLOCK_SAMPLES].reshape(
            n_blocks, SPEECH_BLOCK_SAMPLES
        )
        rms = np.sqrt((blocks.astype(np.float32) ** 2).mean(axis=1))
        self._speech_blocks += int((rms >= SPEECH_BLOCK_RMS).sum())
        self._rms_carry = self._rms_carry[n_blocks * SPEECH_BLOCK_SAMPLES :].copy()

    def _utterance_partials(self) -> list["StreamingAsrUpdate"]:
        """整段录入模式：用离线大模型对已累积音频出中间结果（边说边上字）。

        刻意**不用**流式模型出中间结果：一是要 25MB 的 zipformer 与 78MB 的
        Paraformer 同时常驻（本机实测会撞上 OpenBLAS 内存分配失败），二是长停顿会
        把流式解码状态原地拖坏。离线模型对 12s 音频只要 ~90ms，节流到 0.7s 一次
        完全跟得上，而且中间结果与最终定稿出自同一个模型，上屏的字不会在定稿时
        被整句换掉。
        """
        now = time.monotonic()
        # 还没听到人说话就别解码：纯静音会被模型编成一句假话，而中间结果是直接上屏的
        if not self._has_speech():
            return []
        if now - self._last_partial_at < PARTIAL_MIN_INTERVAL_S:
            return []
        if self._segment_samples - self._partial_audio_samples < int(
            PARTIAL_MIN_NEW_AUDIO_S * SAMPLE_RATE
        ):
            return []

        audio = self._accumulated(PARTIAL_MAX_AUDIO_S)
        if audio is None:
            return []

        self._last_partial_at = now
        self._partial_audio_samples = self._segment_samples
        text, decoded = self._decode_offline(audio)
        if not decoded:
            # 离线模型不可用：本会话不再重复尝试（每次都重新加载只会白烧 CPU）
            self._prefer_offline = False
            return []
        # 离线模型不出标点，但中间结果会随音频增长被整体替换，不值得每 0.7s 补一次标点
        if not text or text == self._last_text:
            return []
        self._last_text = text
        self._prefer_offline = True
        return [
            StreamingAsrUpdate(
                event=AsrEventType.PARTIAL,
                text=text,
                elapsed_ms=self._segment_samples / SAMPLE_RATE * 1000.0,
                latency_ms=0.0,
                engine=AsrEngine.PARAFORMER,
            )
        ]

    # ---- 对外 ----
    def accept(self, samples: np.ndarray) -> list[StreamingAsrUpdate]:
        """喂入一段 float32 单声道波形 [-1,1]，返回本块产生的更新（0~2 条）。

        端点模式可能同时返回 partial 与 final（先上屏、再定稿）；
        整段录入模式只返回 partial，定稿只发生在 `finalize()`。
        """
        if samples.size == 0:
            return []

        self._segment_samples += int(samples.size)
        self._total_samples += int(samples.size)
        self._segment_audio.append(np.ascontiguousarray(samples, dtype=np.float32))
        self._note_audio_for_speech(samples)

        # 整段录入与端点模式都要先建流：`finalize()` 用"流是否存在"判断本段有没有
        # 内容，不建流会让定稿直接返回 None（用户说的话整段丢掉）。建流不清缓冲。
        stream = self._ensure_stream()

        if not self._finalize_on_endpoint:
            # 整段录入：不把音频喂给流式模型 —— 它在这儿唯一的产出是中间结果，
            # 而中间结果已经由离线大模型承担（见 `_utterance_partials`）。
            return self._utterance_partials()

        stream.accept_waveform(SAMPLE_RATE, samples)

        latency_ms = self._drain()
        elapsed_ms = self._segment_samples / SAMPLE_RATE * 1000.0

        updates: list[StreamingAsrUpdate] = []
        text = self._rec.get_result(stream) or ""

        # 只在文本真的变了的时候推中间结果，避免刷屏
        if text and text != self._last_text:
            self._last_text = text
            updates.append(
                StreamingAsrUpdate(
                    event=AsrEventType.PARTIAL,
                    text=text,
                    elapsed_ms=elapsed_ms,
                    latency_ms=latency_ms,
                    engine=self._engine,
                )
            )

        if self._rec.is_endpoint(stream):
            # 定稿前先用离线大模型复识别纠错；纠错耗时也计入 latency_ms
            t0 = time.perf_counter()
            final_text, engine, refined = self._refine(text)
            latency_ms += (time.perf_counter() - t0) * 1000.0
            updates.append(
                StreamingAsrUpdate(
                    event=AsrEventType.FINAL,
                    text=final_text,
                    elapsed_ms=elapsed_ms,
                    latency_ms=latency_ms,
                    reason="endpoint",
                    engine=engine,
                    refined=refined,
                )
            )
            # reset 后这一段从头开始；_ensure_stream 会在下次 accept 时重建
            self._start_new_segment()

        return updates

    def accept_pcm16(self, raw: bytes) -> list[StreamingAsrUpdate]:
        """喂入 int16 小端 PCM 字节流（WebSocket 直接收到的格式）"""
        if not raw:
            return []
        # int16 -> float32 [-1, 1]，与 normalize_samples=True 的期望一致
        pcm = np.frombuffer(raw, dtype="<i2")
        if pcm.size == 0:
            return []
        return self.accept(pcm.astype(np.float32) / 32768.0)

    def finalize(self) -> Optional[StreamingAsrUpdate]:
        """用户主动结束：冲刷残余帧并定稿（同样走离线复识别纠错）。

        Returns:
            定稿更新；本段没有内容时返回 None。
        """
        if self._stream is None:
            return None
        stream = self._stream
        try:
            stream.input_finished()
            latency_ms = self._drain()
        except Exception:  # noqa: BLE001 —— 冲刷失败不应影响连接收尾
            logger.exception("[ASR:stream] finalize 冲刷失败")
            latency_ms = 0.0

        text = self._rec.get_result(stream) or ""
        elapsed_ms = self._segment_samples / SAMPLE_RATE * 1000.0

        # ⚠️ 顺序要紧：`_refine` 靠 `_segment_samples` / `_segment_audio` 判断
        # 这段音频值不值得复识别，必须先纠错再清计数。反了会让整段录入模式
        # （它只靠离线结果）永远拿到空文本。
        t0 = time.perf_counter()
        final_text, engine, refined = self._refine(text)
        latency_ms += (time.perf_counter() - t0) * 1000.0

        self._start_new_segment()

        if not final_text.strip():
            return None
        return StreamingAsrUpdate(
            event=AsrEventType.FINAL,
            text=final_text,
            elapsed_ms=elapsed_ms,
            latency_ms=latency_ms,
            reason="finalize",
            engine=engine,
            refined=refined,
        )

    def reset(self) -> None:
        """丢弃当前段（用户清空/取消）"""
        self._start_new_segment()

    @property
    def total_samples(self) -> int:
        """本次连接累计送入的样本数"""
        return self._total_samples

    @property
    def engine(self) -> AsrEngine:
        """本会话的流式引擎标识（整段录入走离线模型，此处仅作回落标识）"""
        return self._engine
