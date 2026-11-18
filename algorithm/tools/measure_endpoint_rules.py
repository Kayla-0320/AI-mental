# -*- coding: utf-8 -*-
"""量端点规则的实际手感：**说了话之后，尾随静音多久才定稿**。

为什么需要这个工具（而不是只看配置值）：sherpa-onnx 的端点判定基准是
"**已解出的 token**"而不是"音频能量"，还存在 100 ms 分块的量化误差。
所以配置写 1.5 s，实测可能是 1.7 s；更要紧的一条是实测发现的：

    ⚠️ 有效阈值 ≈ min(rule1, rule2)

只调 rule2、把 rule1 留在 1.6 s，实测依然是 1.7 s 左右 —— 两条规则在
"说过话之后"都会生效。改这个参数**必须两条一起看**。

用法（在仓库根目录）：

    # 用当前代码里的默认值测一组
    & '.venv\\Scripts\\python.exe' algorithm/tools/measure_endpoint_rules.py

    # 扫几组候选值，挑一个手感（不用改代码、不用重启服务）
    & '.venv\\Scripts\\python.exe' algorithm/tools/measure_endpoint_rules.py \\
        --rules 1.6/0.8 1.8/1.5 2.0/1.8 2.4/2.4

    # 也可以指定环境变量里那套（真机调参用的就是它）
    $env:ASR_RULE2_SILENCE_S = '1.7'

输出两件事：
    A. 尾随静音到**第一次端点**的真实间隔（"说完之后 AI 要等多久才接话"）
    B. 句间停顿会不会被切开（同一句话中间停 0.6~1.6 s 时是否提前定稿）

配套的自动验收在 `algorithm/tests/test_asr_streaming_modes.py`（不需要人工听）。
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.streaming import (  # noqa: E402
    RULE1_MIN_TRAILING_SILENCE,
    RULE2_MIN_TRAILING_SILENCE,
    RULE3_MIN_UTTERANCE_LENGTH,
    SAMPLE_RATE,
    resolve_streaming_model_dir,
)

#: 与 `streaming.py` 的 SPEECH_BLOCK_RMS 同口径：判断"这个 100ms 块里有没有人说话"
SPEECH_BLOCK_RMS = 0.02
BLOCK_SAMPLES = SAMPLE_RATE // 10  # 100 ms

_DEFAULT_WAV = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "asr"
    / "sherpa-onnx-paraformer-zh-small-2024-03-09"
    / "test_wavs"
    / "0.wav"
)


def read_wav(path: Path) -> np.ndarray:
    """读 16kHz 单声道 wav → float32 [-1,1]"""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE:
            raise SystemExit(f"{path} 不是 {SAMPLE_RATE}Hz（实际 {w.getframerate()}Hz）")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def last_speech_s(audio: np.ndarray) -> float:
    """最后一个"有声" 100ms 块的结束时刻（秒）。全静音时返回 0。"""
    n = len(audio) // BLOCK_SAMPLES
    if n == 0:
        return 0.0
    blocks = audio[: n * BLOCK_SAMPLES].reshape(n, BLOCK_SAMPLES)
    rms = np.sqrt((blocks.astype(np.float32) ** 2).mean(axis=1))
    loud = np.nonzero(rms >= SPEECH_BLOCK_RMS)[0]
    return float((int(loud[-1]) + 1) * BLOCK_SAMPLES / SAMPLE_RATE) if loud.size else 0.0


def build_recognizer(rule1: float, rule2: float):
    import sherpa_onnx

    md = resolve_streaming_model_dir()
    if md is None:
        raise SystemExit("找不到流式模型目录（见 asr.streaming.resolve_streaming_model_dir）")
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(md / "tokens.txt"),
        encoder=str(md / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(md / "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=str(md / "joiner-epoch-99-avg-1.int8.onnx"),
        num_threads=2,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=rule1,
        rule2_min_trailing_silence=rule2,
        rule3_min_utterance_length=RULE3_MIN_UTTERANCE_LENGTH,
        decoding_method="greedy_search",
        provider="cpu",
    )


def first_endpoint_at(rec, audio: np.ndarray) -> float | None:
    """逐 100ms 喂入，返回第一次端点的时刻（秒）；没有端点返回 None。"""
    stream = rec.create_stream()
    for i in range(0, len(audio), BLOCK_SAMPLES):
        stream.accept_waveform(SAMPLE_RATE, audio[i : i + BLOCK_SAMPLES])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        if rec.is_endpoint(stream):
            return (i + BLOCK_SAMPLES) / SAMPLE_RATE
    return None


def measure_tail(rec, speech: np.ndarray, silence_s: float = 3.0) -> float | None:
    """A：说完之后，尾随静音多久定稿（从最后一个有声块算起）"""
    audio = np.concatenate([speech, np.zeros(int(SAMPLE_RATE * silence_s), dtype=np.float32)])
    at = first_endpoint_at(rec, audio)
    if at is None:
        return None
    return at - last_speech_s(speech)


def measure_gap(rec, speech: np.ndarray, gap_s: float) -> tuple[bool, float | None]:
    """B：句中停 `gap_s` 秒会不会被切开。

    Returns:
        (是否在第二段说完之前就定稿, 第一次端点时刻)
    """
    gap = np.zeros(int(SAMPLE_RATE * gap_s), dtype=np.float32)
    audio = np.concatenate([speech, gap, speech])
    at = first_endpoint_at(rec, audio)
    second_end = (len(speech) * 2 + len(gap)) / SAMPLE_RATE
    cut = at is not None and at < second_end - 0.3
    return cut, at


def _parse_rules(raw: list[str]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in raw:
        try:
            r1, r2 = item.split("/")
            out.append((float(r1), float(r2)))
        except ValueError:
            raise SystemExit(f"--rules 的格式应为 rule1/rule2，收到 {item!r}") from None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="量 sherpa 端点规则的实际手感")
    ap.add_argument(
        "--rules",
        nargs="+",
        default=[f"{RULE1_MIN_TRAILING_SILENCE}/{RULE2_MIN_TRAILING_SILENCE}"],
        help="要测的 rule1/rule2 组合，空格分隔（默认当前代码里的值）",
    )
    ap.add_argument("--wav", type=Path, default=_DEFAULT_WAV, help="测试语音（16kHz 单声道）")
    ap.add_argument("--gap", type=float, default=0.8, help="B 组用的句间停顿时长（秒）")
    args = ap.parse_args()

    if not args.wav.is_file():
        raise SystemExit(f"缺少测试语音：{args.wav}")
    speech = read_wav(args.wav)
    print(
        f"素材 {args.wav.name}：{len(speech)/SAMPLE_RATE:.2f}s，"
        f"最后一个有声块在 {last_speech_s(speech):.2f}s\n"
    )

    for rule1, rule2 in _parse_rules(args.rules):
        rec = build_recognizer(rule1, rule2)
        tail = measure_tail(rec, speech)
        cut, at = measure_gap(rec, speech, args.gap)
        print(
            f"rule1={rule1:.2f}s rule2={rule2:.2f}s  →  "
            f"尾随静音 {('%.2fs 定稿' % tail) if tail is not None else '3s 内未定稿'}"
            f"；句中停 {args.gap:.1f}s：{'被切开' if cut else '保住了'}"
            f"（第一次端点 {'%.2fs' % at if at is not None else '无'}）"
        )
    print(
        "\n判读：第二列就是「说完之后 AI 要等多久才接话」；值越大越不容易截断，"
        "但也越像在对讲机。有效阈值 ≈ min(rule1, rule2)。"
    )


if __name__ == "__main__":
    main()
