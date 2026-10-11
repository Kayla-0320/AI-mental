# -*- coding: utf-8 -*-
"""TTS 供音时间线记录器 —— 回答"卡顿是供音不足还是片段接缝"。

## 为什么要单独一个模块（2026-10-11 用户："现在可以正常出声了，但是中途还有些小卡顿"）

用户给的排查顺序是对的，而两条候选原因**必须用时间线分开**：

    A. 供音速度跟不上播放（合成 RTF 长期 > 1，缓冲区被逐渐吃空）
    B. 每段内部供给正常，但**片段之间**出现短暂断档（下一段启动太晚）

判据只能是"**同一时刻**的音量/时间对齐"：把
  ① 微服务（8002）**生成**每一块 PCM 的时刻、
  ② 算法服务（8001）**收到 / 转发**每一块的时刻、
  ③ 浏览器**播放缓冲区剩余毫秒**
放在同一条时间线上，看卡顿那一刻缓冲区是不是见底、以及见底时上游在干什么。

## 记录什么（每个 seq 一段）

* `chunks`：按顺序记 `(t_ms, bytes)` —— `t_ms` 相对**本段开始**的毫秒数。
  块与块之间**间隔大于阈值**（默认 250 ms）的会在摘要里单独列出来：
  那一段空隙如果又长于该块自身的音频时长，就是"听得到的断档"。
* `audio_ms`：本段音频总时长（由字节数与采样率算出，与设备无关）。
* `effective_rtf`：**由时间线自己算出来的** RTF = `(最后一块时刻 − 第一块时刻) ÷ 音频时长`。
  ⚠️ 这是对上游自报 `rtf` 的**独立复核**：自报值来自合成内部计时，
  而这里是"音频到底以多快的速度流出来"——供音不足看的是后者。
  `> 1` 表示平均供音速度慢于播放速度（缓冲区会持续被吃）。
* `first_chunk_ms` / `span_ms`：首块延迟与整段流出耗时。

## 输出

`summary()` 返回一行式摘要（进日志）；`dump()` 返回逐块明细（排查时展开看）。
两者都**不参与任何控制逻辑** —— 这个模块只观察，不改变行为。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

#: 块间隔超过它就单独记录（毫秒）。取 250 ms：8192 字节 @24k/16bit ≈ 170 ms 音频，
#: 正常连着推时块间隔远小于它；超过它就是一次"供给停顿"。
GAP_ALERT_MS = 250.0

#: 逐块明细最多保留多少块（防长通话里无限增长）
MAX_KEPT_CHUNKS = 200


@dataclass
class TtsTimeline:
    """一段（一个 ``seq``）的供音时间线。所有时刻都是相对本段起点（毫秒）。"""

    stage: str
    """记录方：``microservice``（8002 生成）/ ``algorithm``（8001 收转发）/ ``browser``（播放）"""

    seq: int
    sample_rate: int
    started_at: float = field(default_factory=time.perf_counter)
    """本段起点（``time.perf_counter()`` 秒）。浏览器侧用 `performance.now()/1000` 填。"""

    chunks: list[tuple[float, int]] = field(default_factory=list)
    """按序 `(t_ms, bytes)`"""

    audio_ms: float = 0.0
    """累计音频时长（毫秒），由字节数算出：``bytes / (sample_rate * 2) * 1000``"""

    total_bytes: int = 0
    first_chunk_ms: Optional[float] = None
    last_chunk_ms: Optional[float] = None
    gaps_ms: list[tuple[float, float]] = field(default_factory=list)
    """超过 {@link GAP_ALERT_MS} 的块间隔：`(间隔起点_ms, 间隔长度_ms)`"""

    def bytes_to_ms(self, n_bytes: int) -> float:
        """裸 PCM16 单声道：每毫秒 ``sample_rate * 2 / 1000`` 字节"""
        if self.sample_rate <= 0:
            return 0.0
        return n_bytes / (self.sample_rate * 2.0) * 1000.0

    def add_chunk(self, n_bytes: int, t_ms: Optional[float] = None) -> None:
        """记一块 PCM。``t_ms`` 不传就用当前时刻算（相对本段起点）。"""
        now_ms = (time.perf_counter() - self.started_at) * 1000.0 if t_ms is None else t_ms
        if self.first_chunk_ms is None:
            self.first_chunk_ms = now_ms
        if self.last_chunk_ms is not None:
            gap = now_ms - self.last_chunk_ms
            if gap >= GAP_ALERT_MS:
                self.gaps_ms.append((self.last_chunk_ms, gap))
        self.last_chunk_ms = now_ms
        self.audio_ms += self.bytes_to_ms(n_bytes)
        self.total_bytes += n_bytes
        if len(self.chunks) < MAX_KEPT_CHUNKS:
            self.chunks.append((round(now_ms, 1), n_bytes))

    @property
    def span_ms(self) -> float:
        """首块到末块的流出耗时。它才是"供音速度"的分子。"""
        if self.first_chunk_ms is None or self.last_chunk_ms is None:
            return 0.0
        return self.last_chunk_ms - self.first_chunk_ms

    @property
    def effective_rtf(self) -> float:
        """时间线自己算的 RTF = 流出耗时 ÷ 音频时长。

        ⚠️ 与上游自报 ``rtf``（合成内部计时）是**两个口径**：这里量的是
        "音频以多快的速度交给下游"。供音是否跟得上播放，看的是这一个。
        """
        if self.audio_ms <= 0:
            return 0.0
        return self.span_ms / self.audio_ms

    def summary(self, label: str = "") -> str:
        """一行式摘要（进日志）"""
        head = f"[TTS时间线:{self.stage}]{f' {label}' if label else ''} seq={self.seq}"
        if not self.chunks and self.total_bytes == 0:
            return f"{head} 无音频"
        return (
            f"{head} 首块={_fmt(self.first_chunk_ms)}ms 末块={_fmt(self.last_chunk_ms)}ms "
            f"流出={self.span_ms:.0f}ms 音频={self.audio_ms:.0f}ms "
            f"有效RTF={self.effective_rtf:.2f} 块数={len(self.chunks)} 字节={self.total_bytes}"
            + (f" ⚠️停顿{len(self.gaps_ms)}处" if self.gaps_ms else "")
        )

    def dump(self, limit: int = 60) -> str:
        """逐块明细（排查时展开）"""
        lines = [self.summary()]
        for idx, (t_ms, n_bytes) in enumerate(self.chunks[:limit]):
            step = "" if idx == 0 else f" (+{t_ms - self.chunks[idx - 1][0]:.0f}ms)"
            lines.append(
                f"    #{idx} t={t_ms:.0f}ms {n_bytes}B ≈{self.bytes_to_ms(n_bytes):.0f}ms{step}"
            )
        if len(self.chunks) > limit:
            lines.append(f"    …（另有 {len(self.chunks) - limit} 块）")
        for at_ms, gap_ms in self.gaps_ms:
            lines.append(f"    ⚠️ t={at_ms:.0f}ms 起停顿 {gap_ms:.0f}ms（阈值 {GAP_ALERT_MS:.0f}ms）")
        return "\n".join(lines)


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.0f}"
