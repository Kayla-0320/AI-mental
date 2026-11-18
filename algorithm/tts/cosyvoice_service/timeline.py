# -*- coding: utf-8 -*-
"""TTS 供音时间线记录器 —— 8002 微服务侧的**自带副本**。

## 为什么这里有一份副本

`cosyvoice_service/server.py` 跑在**独立虚拟环境**里（`torch==2.7.0+cu128` /
`numpy<2` 与本仓库的算法服务直接冲突，见 `tts/cosyvoice_engine.py` 的模块说明），
所以**不能** `from tts.timeline import ...` —— 那个包会连带拉起算法侧的依赖。
两边口径必须一致，所以这里是同一份实现的精简副本（**只记时间线，不做任何控制**）。

## 记录什么

每个请求一段：逐块 `(相对首块的毫秒, 字节数)`，并算出
`有效RTF = (末块时刻 − 首块时刻) ÷ 音频时长`。

用户口径（2026-10-11）："小卡顿优先排查 TTS 供音速度是否跟得上播放速度"。
这个模块给出的就是**推理端自己的供音速度**；把它与 8001 的转发时间线、
浏览器的缓冲区读数放在一起，就能判出卡顿发生在哪一层。
"""
from __future__ import annotations

import time
from typing import Optional

#: 块间隔超过它就单独记录（毫秒）。见 `algorithm/tts/timeline.py` 同一常量的说明。
GAP_ALERT_MS = 250.0

#: 逐块明细最多保留多少块
MAX_KEPT_CHUNKS = 200


class TtsTimeline:
    """一段（一个 HTTP 流请求）的供音时间线

    ⚠️ 构造函数必须与 8001 侧那份（``algorithm/tts/timeline.py``）**同签名** ——
    两边是同一套口径的两份副本，签名漂了就会在运行期才炸（本地已踩过一次）。
    """

    def __init__(self, stage: str, seq: int, sample_rate: int) -> None:
        self.stage = stage
        self.seq = seq
        self.label = ""
        self.sample_rate = sample_rate
        self.started_at = time.perf_counter()
        self.chunks: list[tuple[float, int]] = []
        self.total_bytes = 0
        self.first_chunk_ms: Optional[float] = None
        self.last_chunk_ms: Optional[float] = None
        self.gaps_ms: list[tuple[float, float]] = []

    def bytes_to_ms(self, n_bytes: int) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return n_bytes / (self.sample_rate * 2.0) * 1000.0

    def add_chunk(self, n_bytes: int, t_ms: Optional[float] = None) -> None:
        """记一块 PCM。``t_ms`` 不传就用当前时刻算（相对本段起点）。

        ``t_ms`` 显式传入只是为了**可测**（本地用构造好的时间线做自检），
        生产路径一律走"取当前时刻"。
        """
        now_ms = (time.perf_counter() - self.started_at) * 1000.0 if t_ms is None else t_ms
        if self.first_chunk_ms is None:
            self.first_chunk_ms = now_ms
        if self.last_chunk_ms is not None:
            gap = now_ms - self.last_chunk_ms
            if gap >= GAP_ALERT_MS:
                self.gaps_ms.append((self.last_chunk_ms, gap))
        self.last_chunk_ms = now_ms
        self.total_bytes += n_bytes
        if len(self.chunks) < MAX_KEPT_CHUNKS:
            self.chunks.append((round(now_ms, 1), n_bytes))

    @property
    def audio_ms(self) -> float:
        return self.bytes_to_ms(self.total_bytes)

    @property
    def span_ms(self) -> float:
        if self.first_chunk_ms is None or self.last_chunk_ms is None:
            return 0.0
        return self.last_chunk_ms - self.first_chunk_ms

    @property
    def effective_rtf(self) -> float:
        """时间线自己算的 RTF = 流出耗时 ÷ 音频时长（与内部计时自报的 rtf 是两个口径）"""
        if self.audio_ms <= 0:
            return 0.0
        return self.span_ms / self.audio_ms

    def summary(self) -> str:
        head = f"cvsvc|timeline|{self.stage}"
        if self.label:
            head += f"|{self.label}"
        if self.total_bytes == 0:
            return f"{head}|无音频"
        return (
            f"{head}|首块={self.first_chunk_ms or 0:.0f}ms|末块={self.last_chunk_ms or 0:.0f}ms"
            f"|流出={self.span_ms:.0f}ms|音频={self.audio_ms:.0f}ms"
            f"|有效RTF={self.effective_rtf:.2f}|块数={len(self.chunks)}|字节={self.total_bytes}"
            + (f"|⚠️停顿{len(self.gaps_ms)}处" if self.gaps_ms else "")
        )

    def dump(self, limit: int = 40) -> str:
        lines = [self.summary()]
        for idx, (t_ms, n_bytes) in enumerate(self.chunks[:limit]):
            step = "" if idx == 0 else f" (+{t_ms - self.chunks[idx - 1][0]:.0f}ms)"
            lines.append(
                f"    #{idx} t={t_ms:.0f}ms {n_bytes}B ≈{self.bytes_to_ms(n_bytes):.0f}ms{step}"
            )
        for at_ms, gap_ms in self.gaps_ms:
            lines.append(f"    ⚠️ t={at_ms:.0f}ms 起停顿 {gap_ms:.0f}ms")
        return "\n".join(lines)
