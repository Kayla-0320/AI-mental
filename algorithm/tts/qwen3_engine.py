# -*- coding: utf-8 -*-
"""Qwen3-TTS 引擎 —— 阿里 2026 开源的对话级语音合成

替换 VITS（2021 架构）以解决"发音机械、断句生硬、音色不像真人"三个问题。
Qwen3-TTS 在对话级韵律、自然停顿、中文发音准确度上都显著优于 VITS，
并原生支持 **Voice Design** —— 用自然语言描述目标音色，无需任何参考音频。

依赖（懒导入，缺失只影响本模块）：

    uv pip install qwen-tts torch soundfile accelerate

模型（VoiceDesign 1.7B，约 3.4 GB FP16）：

    python algorithm/tools/download_asr_model.py --preset qwen3-tts-voicedesign

协议差异（与 VITS 引擎对比）：

    * ``sid`` 参数对本引擎**无效**（Qwen3-TTS 不是多说话人查表，而是 prompt 驱动），
      传入会被忽略并在日志里留一行提示。
    * 输出采样率由模型决定（加载后从返回值读取）。
    * 无流式输出 —— 整句合成完才交付，但音质远优于 VITS。

⚠️ RTX 5060 8GB 上 FP16 约占 3–4 GB 显存，与流式 ASR 共存时需注意显存占用。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from shared.dataclasses import TtsEngine, TtsSynthesisResult

logger = logging.getLogger(__name__)

#: 默认模型仓库 ID（HuggingFace）。可被 ``TTS_MODEL_DIR`` 覆盖为本地路径。
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

#: 默认"知心大姐姐"音色描述（voice design）。
#: 可被 ``QWEN3_VOICE_PROMPT`` 环境变量覆盖；``say`` 控制帧也可逐句指定 ``voice``
#: 字段临时覆盖（见 ``api/tts.py``）。
DEFAULT_VOICE_PROMPT = (
    "温和、共情、声音柔而有韧的成熟女性，语速自然流畅，"
    "停顿得当，像知心大姐姐在耐心倾听与回应。"
)

#: 单次合成的文本上限。**这是显存定的，不是对齐 VITS 定的**：
#: 实测 80 字会让 ``generate_voice_design`` 在 RTX 5060 8GB 上抛
#: ``CUDA out of memory``，而 CUDA 一旦 OOM 就污染整个上下文 ——
#: 之后**每一次**合成都会以 ``invalid resource handle`` 失败，服务只能重启。
#: 29 字实测安全，留一倍余量取 60；客户端 ``splitSpeakable`` 按 36 字切，正常打不满。
MAX_TEXT_CHARS = 60


def default_voice_prompt() -> str:
    """取环境变量 ``QWEN3_VOICE_PROMPT``，缺省回落到 :data:`DEFAULT_VOICE_PROMPT`。"""
    return os.environ.get("QWEN3_VOICE_PROMPT", DEFAULT_VOICE_PROMPT)


def _resolve_model_path(explicit: Optional[str | Path]) -> str:
    """定位模型：显式 > ``TTS_MODEL_DIR`` > ``QWEN3_TTS_MODEL`` > 默认仓库 ID。

    优先级与 VITS 的 :func:`tts.engine.resolve_tts_model_dir` 同构：显式路径不可用
    时**不**静默回落到默认（避免"换模型没生效"那种本项目踩过的坑）。
    """
    if explicit:
        return str(explicit)
    env_dir = os.environ.get("TTS_MODEL_DIR")
    if env_dir:
        return env_dir
    env_model = os.environ.get("QWEN3_TTS_MODEL")
    if env_model:
        return env_model
    return DEFAULT_MODEL_ID


class Qwen3TtsEngine:
    """Qwen3-TTS 单例引擎（与 :class:`tts.engine.VitsTtsEngine` 同构）

    懒加载、失败原因如实上报在 ``load_error``，不抛异常。
    并发安全与 VITS 引擎一致：由调用方串行化（``api/tts.py`` 的全局单 worker 池）。
    """

    _instance: Optional["Qwen3TtsEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        voice_prompt: Optional[str] = None,
    ) -> None:
        self._model_path_input = model_path
        self._model_path: Optional[str] = None
        self._voice_prompt = voice_prompt or default_voice_prompt()

        self._model = None
        self._load_error: str = ""
        self._sample_rate: int = 24000

    @classmethod
    def instance(cls) -> "Qwen3TtsEngine":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── 状态 ────────────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return self.load()

    @property
    def load_error(self) -> str:
        return self._load_error

    @property
    def engine(self) -> TtsEngine:
        return TtsEngine.QWEN3 if self.available else TtsEngine.UNAVAILABLE

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_speakers(self) -> int:
        return 1

    @property
    def model_dir(self) -> Optional[str]:
        return self._model_path

    @property
    def num_threads(self) -> int:
        return 0

    @property
    def voice_prompt(self) -> str:
        """当前使用的 voice design 描述文本"""
        return self._voice_prompt

    # ── 加载 ────────────────────────────────────────────────────────────────
    def load(self) -> bool:
        if self._model is not None:
            return True

        with self._instance_lock:
            if self._model is not None:
                return True

            try:
                from qwen_tts import Qwen3TTSModel  # noqa: PLC0415
            except ImportError as exc:
                self._load_error = (
                    f"未安装 qwen-tts：{exc}。"
                    f"请运行：uv pip install qwen-tts torch soundfile accelerate"
                )
                logger.warning("[TTS:qwen3] %s", self._load_error)
                return False

            model_path = _resolve_model_path(self._model_path_input)
            self._model_path = model_path

            try:
                import torch  # noqa: PLC0415
                from shared.model_lock import MODEL_LOAD_LOCK

                # sdpa: PyTorch 内置的 scaled dot product attention，不需要额外装 flash-attn
                # （flash_attention_2 在 Windows 上很难装，sdpa 性能接近且更稳定）
                with MODEL_LOAD_LOCK:
                    self._model = Qwen3TTSModel.from_pretrained(
                        model_path,
                        dtype=torch.bfloat16,
                        attn_implementation="sdpa",
                    )
                    # device_map 在 Windows + accelerate 上会 segfault，
                    # 改为先加载到 CPU 再手动移到 CUDA，并 patch tokenizer 的设备放置。
                    self._model.model = self._model.model.to("cuda:0")
                    # speech_tokenizer（含 decoder）也要在 CUDA 上，否则流式解码时
                    # codes(来自 talker，CUDA) 与 decoder 权重(CPU) 设备不匹配
                    self._model.model.speech_tokenizer.model = self._model.model.speech_tokenizer.model.to("cuda:0")
                    # ⚠️ 光把权重搬到 CUDA 还不够：**解码入口收到的 codes 是 CPU 张量**
                    # （talker 的生成结果留在 CPU，见 qwen3_tts_model.py 的
                    #  `speech_tokenizer.decode([{...}])` 调用），于是
                    # `EuclideanCodebook.decode → F.embedding(codes, embedding)`
                    # 抛 `Expected all tensors to be on the same device`
                    # —— 表现为**每一次合成都失败**，而模型"加载成功"、状态端点也报 available。
                    # 2026-09-27 在 RTX 5060 上实测复现（codes 形状 (1, 16, 37)，设备 cpu）。
                    #
                    # 修法：在 decoder 的 forward 前把**整型**（离散 code）张量搬到 CUDA。
                    # 用 hook 而不是重写 decode：解码路径有 `decode` / `chunked_decode`
                    # 两个入口，hook 挂在模块上两者都覆盖，也不会随上游重构失效。
                    try:
                        _decoder = self._model.model.speech_tokenizer.model.decoder

                        def _codes_to_cuda(module, args, kwargs):
                            args = tuple(
                                a.to("cuda:0") if isinstance(a, torch.Tensor) and not a.is_floating_point()
                                else a
                                for a in args
                            )
                            kwargs = {
                                k: (v.to("cuda:0") if isinstance(v, torch.Tensor) and not v.is_floating_point() else v)
                                for k, v in kwargs.items()
                            }
                            return args, kwargs

                        _decoder.register_forward_pre_hook(_codes_to_cuda, with_kwargs=True)
                        logger.info("[TTS:qwen3] 已挂载 codes→CUDA 前置钩子（修复解码设备不匹配）")
                    except Exception as exc:  # noqa: BLE001
                        # 挂不上就照旧 —— 但要把话说清楚，别让它变成"合成就失败"的谜题
                        logger.warning("[TTS:qwen3] 解码设备钩子挂载失败（合成可能报设备不匹配）：%s", exc)
                    _orig_tokenize = self._model._tokenize_texts
                    def _tokenize_on_cuda(texts, _fn=_orig_tokenize):
                        result = _fn(texts)
                        return [t.to("cuda:0") if isinstance(t, torch.Tensor) else t for t in result]
                    self._model._tokenize_texts = _tokenize_on_cuda
            except Exception as exc:  # noqa: BLE001
                self._model = None
                self._load_error = f"Qwen3-TTS 加载失败（{model_path}）：{exc}"
                logger.exception("[TTS:qwen3] %s", self._load_error)
                return False

            self._load_error = ""
            logger.info(
                "[TTS:qwen3] 已加载 %s（采样率 %d，voice_prompt=%r）",
                model_path, self._sample_rate,
                self._voice_prompt[:40] + ("…" if len(self._voice_prompt) > 40 else ""),
            )
            return True

    # ── 合成 ────────────────────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        seq: int = 0,
        on_chunk: Optional[Callable[[np.ndarray], int]] = None,
        sid: int = 0,
        speed: float = 1.0,
        voice: Optional[str] = None,
    ) -> TtsSynthesisResult:
        """合成一句话。

        参数：
            text:     待合成文本（调用方应已过 ``prepare_for_speech``）
            seq:      本次合成的序号，原样带回结果
            on_chunk: 每产出一块 float32 波形时回调；返回非 0 中止生成
            sid:      对本引擎**无效**（Qwen3-TTS 不是多说话人查表），保留仅为接口对齐
            speed:    语速（Qwen3-TTS 当前版本未暴露此参数，传入会被忽略）
            voice:    可选的临时 voice prompt，覆盖本实例的默认描述

        返回：
            ``TtsSynthesisResult``。音频已通过 ``on_chunk`` 推走，不留在内存里。
        """
        import time  # noqa: PLC0415

        if not self.load():
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE,
                spoken_text=text, error=self._load_error or "TTS 引擎不可用",
            )

        stripped = (text or "").strip()
        if not stripped:
            return TtsSynthesisResult(seq=seq, spoken_text="", audio_ms=0.0)
        if len(stripped) > MAX_TEXT_CHARS:
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error=f"文本过长（{len(stripped)} > {MAX_TEXT_CHARS} 字），调用方应先切句",
            )

        if sid:
            logger.info("[TTS:qwen3] sid=%d 被忽略（Qwen3-TTS 用 prompt 驱动音色）", sid)
        if speed != 1.0:
            logger.info("[TTS:qwen3] speed=%.2f 被忽略（Qwen3-TTS 当前版本不支持语速控制）", speed)

        prompt = (voice or self._voice_prompt or "").strip()
        if not prompt:
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error="Qwen3-TTS 需要 voice prompt（未提供）",
            )

        delivered_samples = 0
        first_chunk_ms: list[float] = []
        cancelled = False
        t0 = time.perf_counter()

        try:
            # generate_voice_design 返回 (wavs, sr)：wavs 是 list[np.ndarray]，sr 是采样率
            # max_new_tokens=512：200 字中文约 60 秒音频 ≈ 720 个 codec token（12Hz），
            # 512 对多数对话句（<15 秒）够用；超长的应由上层先切句。
            # do_sample=True 保留采样（官方推荐，音质更好），但降低 temperature 减少随机性。
            wavs, sr = self._model.generate_voice_design(
                text=stripped,
                language="Chinese",
                instruct=prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.8,
            )
            first_chunk_ms.append((time.perf_counter() - t0) * 1000.0)
            self._sample_rate = sr

            # wavs[0] 是 float32 numpy array
            samples = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            if on_chunk is not None:
                if on_chunk(samples) != 0:
                    cancelled = True
                else:
                    delivered_samples = len(samples)
            else:
                delivered_samples = len(samples)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[TTS:qwen3] 合成失败")
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error=f"合成失败：{exc}",
                synth_ms=(time.perf_counter() - t0) * 1000.0,
            )

        synth_ms = (time.perf_counter() - t0) * 1000.0
        audio_ms = delivered_samples / self._sample_rate * 1000.0 if self._sample_rate else 0.0

        result = TtsSynthesisResult(
            seq=seq,
            engine=TtsEngine.QWEN3,
            spoken_text=stripped,
            sample_rate=self._sample_rate,
            audio_ms=audio_ms,
            first_chunk_ms=first_chunk_ms[0] if first_chunk_ms else synth_ms,
            synth_ms=synth_ms,
            cancelled=cancelled,
        )
        if result.is_silent and not cancelled:
            logger.warning(
                "[TTS:qwen3] 合成结果为空音频（seq=%d，文本 %d 字）：%s",
                seq, len(stripped), stripped[:40],
            )
        return result

    def synthesize_stream(
        self,
        text: str,
        seq: int = 0,
        on_chunk: Optional[Callable[[np.ndarray], int]] = None,
        sid: int = 0,
        speed: float = 1.0,
        voice: Optional[str] = None,
        chunk_size: int = 50,
    ) -> TtsSynthesisResult:
        """流式合成：分块解码，逐步推送音频。

        与 synthesize() 的区别：
        - synthesize()：generate() 完成后一次性解码整段音频，再推送
        - synthesize_stream()：generate() 完成后，手动分块调用 decoder，
          每解一块就通过 on_chunk 推送，减少解码期内存峰值

        参数：
            chunk_size: 每次解码的 codec token 数（12.5 Hz，50 token = 4 秒音频）
        """
        import time  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if not self.load():
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE,
                spoken_text=text, error=self._load_error or "TTS 引擎不可用",
            )

        stripped = (text or "").strip()
        if not stripped:
            return TtsSynthesisResult(seq=seq, spoken_text="", audio_ms=0.0)
        if len(stripped) > MAX_TEXT_CHARS:
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error=f"文本过长（{len(stripped)} > {MAX_TEXT_CHARS} 字），调用方应先切句",
            )

        prompt = (voice or self._voice_prompt or "").strip()
        if not prompt:
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error="Qwen3-TTS 需要 voice prompt（未提供）",
            )

        delivered_samples = 0
        first_chunk_ms: list[float] = []
        cancelled = False
        t0 = time.perf_counter()

        try:
            # 1. 生成 codec codes（慢的部分，RTF≈2.11）
            input_ids = self._model._tokenize_texts(
                [self._model._build_assistant_text(stripped)]
            )
            instruct_ids = self._model._tokenize_texts(
                [self._model._build_instruct_text(prompt)]
            ) if prompt else None

            talker_codes_list, _ = self._model.model.generate(
                input_ids=input_ids,
                instruct_ids=[instruct_ids[0]] if instruct_ids else None,
                languages=["Chinese"],
                non_streaming_mode=True,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.8,
            )

            # 2. 手动分块解码并逐步推送
            # decoder 路径: speech_tokenizer.model.decoder
            # decoder.forward 期望 codes 形状 [batch, num_quantizers, seq_len]
            # chunked_decode 用 left_context 平滑边界，但返回拼接后的整段音频；
            # 我们要逐块推送，所以自己实现分块循环。
            decoder = self._model.model.speech_tokenizer.model.decoder
            total_upsample = decoder.total_upsample
            left_context_size = 25  # 2 秒上下文，与 chunked_decode 默认值一致

            for codes in talker_codes_list:
                # codes 形状 [seq_len, num_quantizers]，BF16 on CUDA（来自 talker 的 hidden states）
                # decoder 权重也是 BF16，在 CUDA 上 conv 正常；
                # 之前错误地 .cpu() 导致 CPU 上 BF16 conv 不支持。
                # 码本索引需要 int64，BF16 不能直转 long，经 float32 中转：
                codes = codes.float().long()  # BF16 CUDA → float32 CUDA → int64 CUDA
                seq_len = codes.shape[0]

                # 转置为 [1, num_quantizers, seq_len] 供 decoder.forward
                codes_t = codes.unsqueeze(0).transpose(1, 2)  # [1, 16, seq_len]

                # 分块循环
                start_idx = 0
                while start_idx < seq_len:
                    end_idx = min(start_idx + chunk_size, seq_len)
                    ctx = left_context_size if start_idx >= left_context_size else start_idx

                    # 取 codes 切片（含左上下文）
                    codes_chunk = codes_t[..., start_idx - ctx : end_idx]

                    # 解码
                    with torch.no_grad():
                        wav_chunk = decoder(codes_chunk)  # [1, 1, samples]

                    # 裁剪左上下文对应的音频样本
                    trim_samples = ctx * total_upsample
                    wav_out = wav_chunk[..., trim_samples:]

                    # 转 numpy 并推送（decoder 输出 BF16，numpy 不支持，需先转 float32）
                    samples = wav_out.cpu().float().numpy().astype(np.float32).reshape(-1)
                    if on_chunk is not None:
                        if not first_chunk_ms:
                            first_chunk_ms.append((time.perf_counter() - t0) * 1000.0)
                        if on_chunk(samples) != 0:
                            cancelled = True
                            break
                        delivered_samples += len(samples)
                    else:
                        delivered_samples += len(samples)

                    start_idx = end_idx
                    if cancelled:
                        break

            if not first_chunk_ms:
                first_chunk_ms.append((time.perf_counter() - t0) * 1000.0)

        except Exception as exc:  # noqa: BLE001
            logger.exception("[TTS:qwen3] 流式合成失败")
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error=f"合成失败：{exc}",
                synth_ms=(time.perf_counter() - t0) * 1000.0,
            )

        synth_ms = (time.perf_counter() - t0) * 1000.0
        audio_ms = delivered_samples / self._sample_rate * 1000.0 if self._sample_rate else 0.0

        result = TtsSynthesisResult(
            seq=seq,
            engine=TtsEngine.QWEN3,
            spoken_text=stripped,
            sample_rate=self._sample_rate,
            audio_ms=audio_ms,
            first_chunk_ms=first_chunk_ms[0] if first_chunk_ms else synth_ms,
            synth_ms=synth_ms,
            cancelled=cancelled,
        )
        if result.is_silent and not cancelled:
            logger.warning(
                "[TTS:qwen3] 流式合成结果为空音频（seq=%d，文本 %d 字）：%s",
                seq, len(stripped), stripped[:40],
            )
        return result

    # ── 自检 ────────────────────────────────────────────────────────────────
    def status(self) -> dict[str, object]:
        ok = self.load()
        return {
            "available": ok,
            "engine": self.engine.value,
            "model_dir": self.model_dir,
            "model_name": "qwen3-tts",
            "sample_rate": self.sample_rate,
            "num_speakers": self.num_speakers,
            "num_threads": self.num_threads,
            "max_num_sentences": -1,
            "voice_prompt": self._voice_prompt if ok else None,
            "error": self._load_error,
        }


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 [-1,1] → int16 小端字节。

    先 clip 再乘，避免超范围样本在 astype 时回绕成刺耳噪声。
    与前端 ``ttsPlayer.ts`` 的 ``floatToInt16`` 互为逆运算。
    """
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def create_engine(backend: Optional[str] = None):
    """根据 ``TTS_BACKEND`` 环境变量（或显式参数）创建 TTS 引擎单例。

    可选值：
        ``vits``      —— 本地离线 sherpa-onnx VITS（默认，零 GPU 占用）
        ``qwen3``     —— Qwen3-TTS VoiceDesign（对话级韵律，需 GPU 显存）
        ``cosyvoice`` —— CosyVoice3 零样本克隆（独立微服务，HTTP 流式）

    未知值记 warning 后回落 ``vits``。本函数**不**缓存结果（引擎自己带单例）。

    ⚠️ 已知的静默降级风险：本项目此前踩过"``TTS_BACKEND`` 拼错 → 只打一条
    warning → 实际跑的是 vits"的坑（现场表现为"换引擎没生效"，日志里毫不知情）。
    所以这里对未知值**同时**记 warning 与 info，且 ``/tts/status`` 会如实报出
    实际生效的引擎名，便于一眼核对。
    """
    choice = (backend or os.environ.get("TTS_BACKEND") or "vits").strip().lower()
    if choice == "qwen3":
        return Qwen3TtsEngine.instance()
    if choice == "cosyvoice":
        from tts.cosyvoice_engine import CosyVoiceTtsEngine  # noqa: PLC0415 —— 懒导入
        return CosyVoiceTtsEngine.instance()
    if choice not in ("", "vits"):
        logger.warning(
            "[TTS] 未知的 TTS_BACKEND=%r（可选 vits/qwen3/cosyvoice），回落到 vits",
            choice,
        )
    from tts.engine import VitsTtsEngine  # noqa: PLC0415 —— 懒导入避免循环
    return VitsTtsEngine.instance()
