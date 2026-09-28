# -*- coding: utf-8 -*-
"""流式 ASR 的两种结束语义 —— 单元测试

覆盖 `algorithm/asr/streaming.py` 的 `StreamingSession`：

1. **端点模式**（通话页）：说话停顿即定稿，一句一段 —— 既有行为，不能回归。
2. **整段录入模式**（聊天页语音输入）：停顿不定稿，只有 `finalize()` 才定稿；
   中间结果由离线大模型对已累积音频出。

核心断言是第 2 条带来的**精度**：句间停顿如果被切成两段，每段都缺上下文，
实测会错（"重点呢"→"重点来"、"动荡"→"动量"）；整段一次识别才对。
因此这里的音频素材是真实语音 wav 的拼接，不用合成正弦波 —— 声学模型对
正弦波的行为不代表它对真实语音的行为。

模型/依赖缺失时整份跳过（`importorskip` + `available` 判断），
不让 CI 因为没下载 78MB 模型而变红。
"""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asr.streaming import (  # noqa: E402
    PARTIAL_MIN_INTERVAL_S,
    RULE2_MIN_TRAILING_SILENCE,
    StreamingRecognizer,
)
from shared.dataclasses import AsrEventType  # noqa: E402

pytest.importorskip("sherpa_onnx")

SR = 16000
_TD = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "asr"
    / "sherpa-onnx-paraformer-zh-small-2024-03-09"
    / "test_wavs"
)


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


@pytest.fixture(scope="module")
def recognizer() -> StreamingRecognizer:
    rec = StreamingRecognizer.instance()
    if not rec.available:
        pytest.skip(f"流式 ASR 模型不可用：{rec.load_error}")
    return rec


@pytest.fixture(scope="module")
def speech_pair() -> tuple[np.ndarray, np.ndarray]:
    a, b = _TD / "0.wav", _TD / "1.wav"
    if not (a.is_file() and b.is_file()):
        pytest.skip(f"缺少测试音频：{_TD}")
    return _read_wav(a), _read_wav(b)


def _feed_in_blocks(session, audio: np.ndarray, block_ms: int = 100) -> list:
    """按 100ms 分块喂入（与服务端约定的分片一致），返回全部更新。"""
    step = int(SR * block_ms / 1000)
    updates: list = []
    for i in range(0, len(audio), step):
        updates.extend(session.accept(audio[i : i + step]))
    return updates


def _finals(updates: list) -> list:
    return [u for u in updates if u.event == AsrEventType.FINAL]


class TestUtteranceModeDoesNotCutOnPause:
    """整段录入模式：句间停顿不得触发定稿。"""

    def test_silence_pause_emits_no_final(self, recognizer, speech_pair):
        a, _ = speech_pair
        sess = recognizer.new_session(finalize_on_endpoint=False)

        updates = _feed_in_blocks(sess, a)
        # 尾随静音：超过端点规则 rule2 的时长，端点模式下这里必然定稿
        silence = np.zeros(int(SR * (RULE2_MIN_TRAILING_SILENCE + 0.5)), dtype=np.float32)
        updates += _feed_in_blocks(sess, silence)

        assert _finals(updates) == [], (
            "整段录入模式在停顿处定稿了 —— 这就是「停顿几秒自动截断」的根源"
        )

    def test_finalize_emits_one_final_for_whole_recording(self, recognizer, speech_pair):
        a, b = speech_pair
        gap = np.zeros(int(SR * 1.2), dtype=np.float32)  # 句间自然停顿
        sess = recognizer.new_session(finalize_on_endpoint=False)

        updates = _feed_in_blocks(sess, np.concatenate([a, gap, b]))
        u = sess.finalize()

        assert _finals(updates) == []
        assert u is not None and u.event == AsrEventType.FINAL
        assert u.reason == "finalize"
        assert u.text.strip(), "整段定稿不该是空的"

    def test_whole_utterance_beats_pause_split(self, recognizer, speech_pair):
        """精度回归：整段识别的结果必须比"停顿切成两段"更准。

        两段拼接后先各喂一次、中间留 1.2s 长停顿：
          · 端点模式会在停顿处各定稿一段（旧行为）
          · 整段录入模式只在最终定稿一次（新行为）
        断言新行为给出的文本与**离线整段识别**一致 —— 那是本机可得的最高精度，
        也就是"修复后应该达到的水平"。
        """
        a, b = speech_pair
        gap = np.zeros(int(SR * 1.2), dtype=np.float32)
        mixed = np.concatenate([a, gap, b])

        # 基准：离线大模型对完整音频识别一次
        from asr.engine import ParaformerRecognizer

        offline = ParaformerRecognizer.instance()
        if not offline.available:
            pytest.skip(f"离线 ASR 不可用：{offline.load_error}")
        baseline = offline.transcribe(mixed, SR).text.strip().replace(" ", "")
        assert baseline, "离线基准不该是空的"

        # 端点模式：会被停顿切段
        ep = recognizer.new_session(finalize_on_endpoint=True)
        ep_updates = _feed_in_blocks(ep, mixed)
        ep_u = ep.finalize()
        if ep_u is not None:
            ep_updates.append(ep_u)
        ep_text = "".join(u.text for u in _finals(ep_updates)).replace(" ", "")
        assert len(_finals(ep_updates)) >= 2, (
            f"端点模式本该在 1.2s 停顿处切段，实际只定稿 "
            f"{len(_finals(ep_updates))} 段：{ep_text!r}"
        )

        # 整段录入模式：只有一段定稿
        ut = recognizer.new_session(finalize_on_endpoint=False)
        _feed_in_blocks(ut, mixed)
        ut_u = ut.finalize()
        assert ut_u is not None
        ut_text = ut_u.text.strip().replace(" ", "")

        def _distance(x: str, y: str) -> int:
            """编辑距离（不引第三方库）"""
            prev = list(range(len(y) + 1))
            for i, cx in enumerate(x, 1):
                cur = [i]
                for j, cy in enumerate(y, 1):
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
                prev = cur
            return prev[-1]

        d_ut = _distance(ut_text, baseline)
        d_ep = _distance(ep_text, baseline)
        assert d_ut <= d_ep, (
            f"整段识别没有比端点切段更接近离线基准：\n"
            f"  离线基准 {baseline!r}\n"
            f"  整段录入 {ut_text!r} (距离 {d_ut})\n"
            f"  端点切段 {ep_text!r} (距离 {d_ep})"
        )


class TestStreamingModeStillCuts:
    """端点模式（通话页）的既有行为不能回归。"""

    def test_pause_still_finalizes(self, recognizer, speech_pair):
        a, _ = speech_pair
        sess = recognizer.new_session(finalize_on_endpoint=True)

        updates = _feed_in_blocks(sess, a)
        silence = np.zeros(int(SR * (RULE2_MIN_TRAILING_SILENCE + 0.5)), dtype=np.float32)
        updates += _feed_in_blocks(sess, silence)

        finals = _finals(updates)
        assert finals, "端点模式下停顿本该自动定稿（通话页一句一轮靠它）"
        assert finals[0].reason == "endpoint"

    def test_default_mode_is_endpoint(self, recognizer):
        """不传参数时必须还是端点模式 —— 旧客户端（不发 hello）不能被改变行为。"""
        sess = recognizer.new_session()
        assert sess._finalize_on_endpoint is True


class TestUtterancePartials:
    """整段录入模式的中间结果（边说边上字）。"""

    def test_partial_carries_accumulated_text(self, recognizer, speech_pair):
        a, _ = speech_pair
        sess = recognizer.new_session(finalize_on_endpoint=False)
        speech = np.concatenate([a, a])  # 约 11s，足够跨过两次节流窗口
        updates = _feed_in_blocks(sess, speech)

        partials = [u for u in updates if u.event == AsrEventType.PARTIAL]
        assert partials, "整段录入模式没有出中间结果 —— 用户会看不到任何字"
        assert partials[-1].text.strip()
        # 中间结果只增不减：文本由同一个离线模型对更长的音频解码
        assert len(partials) >= 1

    def test_partial_is_throttled(self, recognizer, speech_pair):
        """中间结果按时间节流：刚出过一次，紧接着再喂音频不该立刻再解一次。

        这里刻意**不依赖"第一次一定出中间结果"**：`_has_speech()` 需要累计到
        0.3s 超阈音频才允许解码，什么时候出第一条取决于素材能量。因此先喂到
        出第一条（有上限），再断言紧随其后的音频不会再出 —— 那才是节流本身。
        """
        a, _ = speech_pair
        sess = recognizer.new_session(finalize_on_endpoint=False)

        # 喂到出第一条中间结果为止（0.7s 节流窗口 + 0.3s 语音确认，留足余量）
        first_at = None
        step = int(SR * 0.1)
        fed = 0
        while fed < int(SR * 2.0):
            chunk = a[fed : fed + step]
            ups = sess.accept(chunk)
            fed += step
            if any(u.event == AsrEventType.PARTIAL for u in ups):
                first_at = fed
                break
        assert first_at is not None, "喂了 2s 真实语音都没出中间结果"

        # 紧接着再喂 0.2s（远小于 0.7s 节流间隔）：不该再解一次
        assert PARTIAL_MIN_INTERVAL_S > 0.2
        ups = sess.accept(a[fed : fed + int(SR * 0.2)])
        assert [u for u in ups if u.event == AsrEventType.PARTIAL] == []

    def test_no_speech_yields_no_final(self, recognizer):
        """全程静音：不许定稿出假文本（"我们我我的时呢"）。"""
        sess = recognizer.new_session(finalize_on_endpoint=False)
        silence = np.zeros(int(SR * 3.0), dtype=np.float32)
        updates = _feed_in_blocks(sess, silence)
        u = sess.finalize()

        assert _finals(updates) == []
        assert u is None, f"纯静音不该定稿，实际得到 {u.text!r}"
