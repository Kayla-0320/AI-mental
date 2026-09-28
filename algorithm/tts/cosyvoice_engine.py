# -*- coding: utf-8 -*-
"""CosyVoice3 合成引擎 —— 独立微服务的**同步客户端**

## 为什么是"客户端"而不是"引擎"

CosyVoice3 的依赖（``torch==2.7.0+cu128`` / ``numpy<2`` / ``transformers==4.51.3``）
与算法服务的 ``torch==2.11`` / ``numpy==2.5`` **直接冲突**，无法放进同一解释器。
所以模型跑在独立微服务里（``tts/cosyvoice_service/server.py``，默认 8002 端口），
本模块只负责把文本发过去、把 PCM 边收边交出去。

这与 ``tts/engine.py``（VITS）和 ``tts/qwen3_engine.py``（Qwen3）的差别只在
"合成发生在哪个进程"，**对外接口完全同构**：``available`` / ``load_error`` /
``engine`` / ``sample_rate`` / ``status()`` / ``synthesize_stream()``。
``api/tts.py`` 因此不需要区分它是本地还是远程。

## 音色与语速

音色由服务侧的参考音决定（``cosyvoice_service/prompt_qwen3_voice.wav``，
零样本克隆），本模块不参与，也**不能**用 ``voice`` 字段临时覆盖 ——
与 Qwen3 的 voice design 不同，这里换音色要换参考音文件。

语速由服务侧完成（默认 1.2×，逐块保音高加速）。本模块把
``speed`` 透传给服务，``TTS_SPEED`` 环境变量仍是统一入口。

## 为什么用 HTTP 分块流而不是 WebSocket

本模块是**同步**调用的（跑在 ``api/tts.py`` 的单工作线程池里）。
HTTP 分块流用标准库 ``urllib`` 就能边收边处理，不需要再维护一条 asyncio
事件循环往同步线程里桥接。而本项目既有的 WS 契约方向本就是单向的
（客户端发文本、服务端回 PCM），不需要双工。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

import numpy as np

from shared.dataclasses import TtsEngine, TtsSynthesisResult

logger = logging.getLogger(__name__)

#: 微服务地址。默认与本机 8001（算法）相邻的 8002。
DEFAULT_BASE_URL = "http://127.0.0.1:8002"

#: 单次合成的文本上限（与 ``tts.engine.MAX_TEXT_CHARS`` 对齐，
#: 服务侧也会再校验一次 —— 两道都要有，因为它们是两个进程）
MAX_TEXT_CHARS = 200

#: 建立连接的超时（秒）。**不含**合成时间 —— 那是读超时，见 ``_READ_TIMEOUT_S``。
_CONNECT_TIMEOUT_S = 5.0

#: 读取下一块音频的超时（秒）。取大一些：首块要等 LLM 生成约 53 个 token
#: （实测本机 2–4 s），慢机器留足余量，避免把正常的首块等待误判为超时。
_READ_TIMEOUT_S = 60.0

#: 每次 ``read()`` 的字节数。16 kHz/24 kHz int16 下 8192 字节≈170–256 ms 音频，
#: 与服务的分块粒度匹配，不会引入额外等待。
_READ_CHUNK_BYTES = 8192


def default_base_url() -> str:
    """微服务基址：``COSYVOICE_API_URL`` 环境变量 > 默认 ``http://127.0.0.1:8002``"""
    return (os.environ.get("COSYVOICE_API_URL") or DEFAULT_BASE_URL).rstrip("/")


def default_speed() -> float:
    """默认语速：``COSYVOICE_SPEED`` > ``TTS_SPEED`` > 1.2

    优先级刻意让 ``COSYVOICE_SPEED`` 在前：``TTS_SPEED`` 是给 VITS 的（0.5–2.0
    的简单插值），而 CosyVoice3 的 1.2× 是"逐块保音高加速"，两者语义不同，
    分开配置才不会互相干扰。
    """
    for key in ("COSYVOICE_SPEED", "TTS_SPEED"):
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("[TTS:cosyvoice] %s=%r 不是数字，忽略", key, raw)
            continue
        return max(0.5, min(2.0, value))
    return 1.2


def _pcm16_to_float32(raw: bytes) -> np.ndarray:
    """int16 小端字节 → float32 [-1, 1]

    与 ``tts.qwen3_engine.float_to_pcm16`` 互为逆运算。这里必须要转：
    ``api/tts.py`` 的 ``on_chunk`` 回调契约收的是 ``np.ndarray``，
    而 HTTP 上流过来的是裸字节。
    """
    if len(raw) % 2:  # 理论上不会发生（服务端整样本发送），截断比崩溃好
        logger.warning("[TTS:cosyvoice] PCM 字节数为奇数（%d），丢弃最后一个不完整样本", len(raw))
        raw = raw[:-1]
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class CosyVoiceTtsEngine:
    """CosyVoice3 微服务客户端（单例）

    与 :class:`tts.engine.VitsTtsEngine` / :class:`tts.qwen3_engine.Qwen3TtsEngine`
    同构：懒探测、失败原因记在 ``load_error`` 而不抛异常、
    并发安全由调用方串行化（``api/tts.py`` 的全局单 worker 池）。

    ⚠️ 与那两个引擎的一个关键差别：**模型不在本进程**。所以 ``load()``
    探的是微服务的 ``/health``，而不是本地的模型文件。这意味着"引擎不可用"
    的常见原因是**微服务没起**，错误信息必须把这一点说清楚（本项目的既有教训：
    "服务能起来但某能力静默失效"是最难查的失效类型）。
    """

    _instance: Optional["CosyVoiceTtsEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(self, base_url: Optional[str] = None) -> None:
        self._base_url = (base_url or default_base_url()).rstrip("/")
        self._load_error: str = ""
        self._sample_rate: int = 24000
        self._speed: float = default_speed()
        self._prompt: Optional[str] = None
        self._checked: bool = False

    @classmethod
    def instance(cls) -> "CosyVoiceTtsEngine":
        """取全局单例（双检锁；与 ``VitsTtsEngine.instance()`` 同构）"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── 状态 ────────────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        """微服务是否可用（会触发一次健康探测）"""
        return self.load()

    @property
    def load_error(self) -> str:
        """最近一次探测失败的原因；空串表示无失败"""
        return self._load_error

    @property
    def engine(self) -> TtsEngine:
        """当前实际使用的引擎"""
        return TtsEngine.COSYVOICE if self.available else TtsEngine.UNAVAILABLE

    @property
    def sample_rate(self) -> int:
        """输出采样率（Hz）。取微服务自述值，缺省 24000"""
        return self._sample_rate

    @property
    def num_speakers(self) -> int:
        """说话人数。零样本克隆下恒为 1（音色由参考音决定）"""
        return 1

    @property
    def num_threads(self) -> int:
        """合成线程数。**恒为 0** —— 计算不在本进程，本进程没有合成线程"""
        return 0

    @property
    def model_dir(self) -> Optional[str]:
        """服务侧的模型目录（由微服务自述，本地不推断）"""
        return self._base_url

    @property
    def voice_prompt(self) -> Optional[str]:
        """参考音文件名。只用于展示 —— 换音色要换服务侧的文件，不能逐句覆盖"""
        return self._prompt

    @property
    def speed(self) -> float:
        """当前默认语速倍率"""
        return self._speed

    # ── 探测 ────────────────────────────────────────────────────────────────
    def load(self) -> bool:
        """探测微服务 ``/health``；结果缓存在 :attr:`load_error`。

        为什么不在失败后重试：VITS/Qwen3 那两个引擎是"模型缺失就一直缺失"，
        但这里**微服务是可以后起的**。所以一旦探测成功就不再重复探测；
        探测失败则**每次都重试**（代价是一次 5 s 超时的 HTTP GET，
        而失败路径本来就没有合成可做）。
        """
        if self._checked and not self._load_error:
            return True

        try:
            payload = self._get_json("/health", timeout=_CONNECT_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            self._load_error = (
                f"CosyVoice3 微服务返回 HTTP {exc.code}：{exc.reason}。"
                f"该端点不可用时通常意味着模型加载失败，请查看微服务日志"
            )
            logger.warning("[TTS:cosyvoice] %s", self._load_error)
            return False
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self._load_error = (
                f"连不上 CosyVoice3 微服务（{self._base_url}）：{exc}。"
                f"请先启动它：cd algorithm/tts/cosyvoice_service && "
                f"%COSYVOICE_ROOT%\\.venv\\Scripts\\python.exe -m uvicorn server:app "
                f"--host 127.0.0.1 --port 8002"
            )
            logger.warning("[TTS:cosyvoice] %s", self._load_error)
            return False

        if not payload.get("available"):
            self._load_error = (
                f"CosyVoice3 微服务在线但模型不可用：{payload.get('error') or '未说明原因'}"
            )
            logger.warning("[TTS:cosyvoice] %s", self._load_error)
            return False

        # 服务自述的采样率才是事实，不要猜
        try:
            self._sample_rate = int(payload.get("sample_rate") or self._sample_rate)
        except (TypeError, ValueError):
            logger.warning("[TTS:cosyvoice] 服务上报的 sample_rate 非法：%r", payload.get("sample_rate"))
        if payload.get("speed"):
            try:
                self._speed = float(payload["speed"])
            except (TypeError, ValueError):
                pass
        self._prompt = payload.get("prompt") or self._prompt

        self._load_error = ""
        self._checked = True
        logger.info(
            "[TTS:cosyvoice] 微服务就绪 %s（采样率 %d，参考音 %s，默认语速 %.2f）",
            self._base_url, self._sample_rate, self._prompt, self._speed,
        )
        return True

    # ── 合成 ────────────────────────────────────────────────────────────────
    def synthesize_stream(
        self,
        text: str,
        seq: int = 0,
        on_chunk: Optional[Callable[[np.ndarray], int]] = None,
        sid: int = 0,
        speed: float = 1.0,
        voice: Optional[str] = None,
        **_ignored: Any,
    ) -> TtsSynthesisResult:
        """流式合成：从微服务边收边把音频交给 ``on_chunk``。

        Args:
            text: 待合成文本。调用方保证已过安全闸门；术语替换在
                :func:`tts.pronunciation.prepare_for_speech` 里做。
            seq: 本次合成的序号，原样带回结果。
            on_chunk: 每收到一块波形调用一次，**返回非 0 会让本函数提前停止**
                并把结果标为 ``cancelled``。与 VITS/Qwen3 的契约一致。
            sid: **忽略** —— 零样本克隆下音色由服务侧参考音决定（留参数是为了
                接口同构，换音色请换服务侧的 wav）。
            speed: 语速倍率。传 1.0（默认值）表示"用服务侧默认值"，
                而不是"原速" —— 因为算法侧 ``TTS_SPEED`` 的默认值 1.0 是给 VITS
                设的原速，不该覆盖 CosyVoice 已经调好的 1.2×。
            voice: **忽略** —— 见 ``sid`` 的说明。

        Returns:
            TtsSynthesisResult: 计时与状态；失败时 ``error`` 非空且
                ``engine`` 为 ``UNAVAILABLE`` —— **不抛异常**，与
                ``/asr/transcribe`` 用 200 + error 字段如实上报一致。
        """
        if not self.load():
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE,
                spoken_text=text, error=self._load_error or "CosyVoice3 不可用",
            )

        stripped = (text or "").strip()
        if not stripped:
            # 空文本是**正常**情况（例如闸门只放行了标点），不是错误
            return TtsSynthesisResult(seq=seq, engine=TtsEngine.COSYVOICE,
                                      spoken_text="", audio_ms=0.0)
        if len(stripped) > MAX_TEXT_CHARS:
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error=f"文本过长（{len(stripped)} > {MAX_TEXT_CHARS} 字），调用方应先切句",
            )

        # speed == 1.0 视为"未指定"：算法侧 TTS_SPEED 的默认 1.0 是 VITS 的原速，
        # 不该把服务侧调好的 1.2× 覆盖回原速（那是本项目最怕的"改了没生效"）。
        effective_speed = self._speed if abs(speed - 1.0) < 1e-6 else speed

        delivered_samples = 0
        first_chunk_ms: list[float] = []
        cancelled = False
        t0 = time.perf_counter()
        body = json.dumps({"text": stripped, "speed": effective_speed}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/tts/stream",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=_READ_TIMEOUT_S) as resp:
                sample_rate = self._sample_rate
                raw_sr = resp.headers.get("X-Sample-Rate")
                if raw_sr:
                    try:
                        sample_rate = int(raw_sr)
                    except ValueError:
                        logger.warning("[TTS:cosyvoice] 响应头 X-Sample-Rate 非法：%r", raw_sr)

                while True:
                    block = resp.read(_READ_CHUNK_BYTES)
                    if not block:
                        break
                    if not first_chunk_ms:
                        first_chunk_ms.append((time.perf_counter() - t0) * 1000.0)
                    samples = _pcm16_to_float32(block)
                    if on_chunk is not None and on_chunk(samples) != 0:
                        cancelled = True
                        break
                    delivered_samples += len(samples)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001 —— 读错误体失败不能盖住原始错误
                pass
            logger.warning("[TTS:cosyvoice] 合成请求失败 HTTP %s：%s", exc.code, detail)
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error=f"CosyVoice3 微服务返回 HTTP {exc.code}：{detail or exc.reason}",
                synth_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 —— 单次合成失败不应让整条 WS 断开
            logger.exception("[TTS:cosyvoice] 合成失败")
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error=f"合成失败：{type(exc).__name__}: {exc}",
                synth_ms=(time.perf_counter() - t0) * 1000.0,
            )

        synth_ms = (time.perf_counter() - t0) * 1000.0
        audio_ms = delivered_samples / sample_rate * 1000.0 if sample_rate else 0.0
        result = TtsSynthesisResult(
            seq=seq,
            engine=TtsEngine.COSYVOICE,
            spoken_text=stripped,
            sample_rate=sample_rate,
            audio_ms=audio_ms,
            first_chunk_ms=first_chunk_ms[0] if first_chunk_ms else synth_ms,
            synth_ms=synth_ms,
            cancelled=cancelled,
        )
        # 静默失效必须留痕：模型"成功返回但没产出音频"是本项目最怕的失效类型
        if result.is_silent and not cancelled:
            logger.warning("[TTS:cosyvoice] 合成结果为空音频（seq=%d，%d 字）：%s",
                           seq, len(stripped), stripped[:40])
        return result

    # ── 自检 ────────────────────────────────────────────────────────────────
    def status(self) -> dict[str, object]:
        """引擎状态字典，供 ``/tts/status`` 如实上报。

        刻意包含 ``base_url`` / ``speed`` / ``prompt``：远程引擎出问题时，
        "连的是哪个地址""实际用了什么语速""用的是哪段参考音"是最常见的三个根因。
        """
        ok = self.load()
        return {
            "available": ok,
            "engine": self.engine.value,
            "model_dir": self._base_url,          # 远程引擎：这里报服务地址
            "model_name": f"cosyvoice3 @ {self._base_url}",
            "sample_rate": self._sample_rate,
            "num_speakers": self.num_speakers,
            "num_threads": self.num_threads,      # 恒 0：计算不在本进程
            "max_num_sentences": -1,
            "voice_prompt": self._prompt,
            "speed": self._speed,
            "error": self._load_error,
        }

    # ── HTTP 小工具 ─────────────────────────────────────────────────────────
    def _get_json(self, path: str, timeout: float) -> dict[str, Any]:
        """GET 一个 JSON 端点。失败由调用方处理（这里只保证不吞异常类型）"""
        with urllib.request.urlopen(f"{self._base_url}{path}", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
