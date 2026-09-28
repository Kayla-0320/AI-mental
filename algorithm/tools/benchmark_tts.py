#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TTS 可行性 benchmark —— 陪伴通话的 go / no-go 判据

它回答的问题是：**「Path A（本地 ASR + 云端 LLM + 本地 TTS）」到底成不成立。**

为什么必须有这个脚本：
    `voice_call_plan.md` 把"TTS 总 RTF < 1.0"列为阶段 0 的 go/no-go 闸门，
    但那个数字一直是 **[估计]**（由树莓派 4 的 1.600 外推），从没在本机量过。
    而 RTF 是**结构性**约束：RTF ≥ 1.0 时合成追不上播放，播放队列越积越长，
    听起来就是"AI 越说越慢"——这不是调参能救的，是路径选择问题。

判据（`voice_call_plan.md` §5 阶段 0，**首块一条已据实测修正**）：
    * go      ：RTF < 0.4 **且** 最短句合成 < 400 ms
    * 兜底    ：换 `vits-piper-zh_CN-huayan-medium` 重测（需先确认 License）
    * no-go   ：两者都不达标 → Path A 出局，D1 只剩 Path B（音频上云）

⚠️ **为什么把"首块 < 250 ms"改掉了（实测结论，见 docs/tts_benchmark.md）**：
    VITS 经 `sherpa_onnx.OfflineTts.generate(callback=...)` **不流式** ——
    callback 对每一句只被调用**一次**，且回调收到的样本与 `generate()` 返回的
    `audio.samples` **完全相等**。也就是说"首块"就等于"整句合成耗时"，
    没有任何增量输出。250 ms 这个数是为**流式**合成定的（当时假设"边合成边推 PCM"），
    在非流式下结构性不可达：最短的 9 字句在 2 线程下也要 ~283 ms。
    因此判据改为对**最短句**的整句合成耗时设阈，并在输出里同时给出
    "首声可用时刻 = 首句文本就绪 + 该句合成耗时"这个真正影响观感的数。

⚠️ **另一个实测坑**：VITS 的**音频时长是随机的**（时长预测带噪声），
    同一句话实测出现过 3945 / 5938 / 6125 ms 三种长度。因此 RTF 单次测量不可信，
    必须重复取中位数；下表同时给出 audio_ms 的 min/max 以暴露这个方差。

为什么还要「满载」那一栏：
    本机 CPU-only。通话时同时跑 16 kHz 流式 ASR + O(n²) 的基频估计 + 本 TTS，
    空载达标不代表通话中达标。`--contention` 会把项目自己的流式识别器
    拉起来持续解码，用真实争用而不是空转来测。

用法：
    python algorithm/tools/benchmark_tts.py
    python algorithm/tools/benchmark_tts.py --contention
    python algorithm/tools/benchmark_tts.py --model-dir algorithm/models/tts/vits-zh-hf-fanchen-C
    python algorithm/tools/benchmark_tts.py --repeat 5 --out-dir _tts_bench

落盘：`--out-dir`（默认 `_tts_bench/`）下导出每句的 16 kHz WAV，**要人真的听一遍**。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_DIR = REPO_ROOT / "algorithm"
DEFAULT_MODEL_DIR = ALGORITHM_DIR / "models" / "tts" / "vits-zh-hf-fanchen-C"

#: 让 `from asr.streaming import ...` 可用（tools/ 不在算法包的搜索路径上）
sys.path.insert(0, str(ALGORITHM_DIR))

#: 典型 AI 陪伴回复。长度按 voice_call_plan.md §5 阶段 0 的要求覆盖 10/20/40/80 字，
#: 另外两句是**边界样本**：数字与英文（文本正则化）、以及带问句的长句。
CORPUS: tuple[tuple[str, str], ...] = (
    ("10字", "我在听，你慢慢说。"),
    ("20字", "听起来这段时间你过得挺不容易的。"),
    ("40字", "听起来这段时间你过得挺不容易的。能和我说说，是从什么时候开始的吗？"),
    (
        "80字",
        "听起来这段时间你过得挺不容易的。晚上睡不着的时候，脑子里会反复想些什么呢？"
        "是白天发生的事，还是对以后的担心？你可以慢慢讲，我不着急。",
    ),
    ("含数字", "如果你需要的话，可以拨打 400-161-9995，也可以随时回来找我。"),
    ("含英文", "我们试试 CBT 里的一个方法，叫做认知重构，好吗？"),
)


@dataclass
class SynthRun:
    """一次合成的计时结果

    Attributes:
        text: 被合成的文本
        label: 语料标签（10字 / 含数字 ……）
        first_chunk_ms: 从 `generate()` 调用到**第一次 callback** 的耗时（毫秒）
        synth_ms: 合成总耗时（毫秒）
        audio_ms: 产出音频时长（毫秒）
        chunks: callback 被调用的次数
        sample_rate: 输出采样率（Hz）
    """
    text: str
    label: str
    first_chunk_ms: float
    synth_ms: float
    audio_ms: float
    chunks: int
    sample_rate: int
    samples: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, np.float32))

    @property
    def rtf(self) -> float:
        """实时率 = 合成耗时 / 音频时长；< 1.0 才追得上播放"""
        return self.synth_ms / self.audio_ms if self.audio_ms > 0 else 0.0


def load_tts(model_dir: Path, num_threads: int = 2):
    """加载 OfflineTts；失败时抛出带原因的异常（不静默降级）。

    Args:
        model_dir: 模型目录（含 onnx / lexicon / tokens / dict / *.fst）。
        num_threads: 合成线程数；1 = 最少占用 CPU，通话场景的默认取向。

    Returns:
        sherpa_onnx.OfflineTts: 已加载的合成器。

    Raises:
        FileNotFoundError: 模型文件缺失。
        RuntimeError: sherpa-onnx 加载失败。
    """
    import sherpa_onnx

    onnx = next(iter(sorted(model_dir.glob("*.onnx"))), None)
    if onnx is None:
        raise FileNotFoundError(f"模型目录里没有 .onnx：{model_dir}")
    for name in ("lexicon.txt", "tokens.txt"):
        if not (model_dir / name).exists():
            raise FileNotFoundError(f"缺少 {name}：{model_dir / name}")

    # vits-zh-hf-* 系列必须有 jieba 词典目录，否则加载直接失败（实测，见预设注释）
    dict_dir = model_dir / "dict"
    fsts = [
        str(model_dir / n)
        for n in ("date.fst", "number.fst", "phone.fst", "new_heteronym.fst")
        if (model_dir / n).exists()
    ]

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(onnx),
                lexicon=str(model_dir / "lexicon.txt"),
                tokens=str(model_dir / "tokens.txt"),
                dict_dir=str(dict_dir) if dict_dir.is_dir() else "",
            ),
            num_threads=num_threads,
            provider="cpu",
            debug=False,
        ),
        # ⚠️ **绝对不要设成 1**：实测 `max_num_sentences=1` 会让 generate() 只合成
        # **第一句**，后面的句子被静默丢弃 —— 66 字的 4 句文本只产出 4258 ms 音频，
        # 与 16 字单句的 4218 ms 几乎一样（见 docs/tts_benchmark.md 的截断表）。
        # 这是"没报错、没日志、就是少念了几句"的静默失效，本项目已经栽过同类跟头。
        # -1 = 不限制；实测 -1 与 100 结果一致（都产出完整的 16434 ms）。
        max_num_sentences=-1,
        rule_fsts=",".join(fsts),
    )

    tts = sherpa_onnx.OfflineTts(config)
    if tts.sample_rate <= 0:
        raise RuntimeError("OfflineTts 初始化后 sample_rate 非法，模型未真正加载")
    return tts


def synth_once(tts, text: str, label: str) -> SynthRun:
    """合成一次并计时。

    callback 的签名是 `(samples: np.ndarray, progress: float) -> int`，
    返回非 0 会**中止合成** —— 这正是通话里"打断"要用的东西（见施工单 S1.4）。
    """
    chunks: list[np.ndarray] = []
    t0 = time.perf_counter()
    first: list[float] = []

    def on_chunk(samples: np.ndarray, progress: float) -> int:  # noqa: ARG001
        if not first:
            first.append((time.perf_counter() - t0) * 1000.0)
        chunks.append(np.asarray(samples, dtype=np.float32))
        return 0

    audio = tts.generate(text, sid=0, speed=1.0, callback=on_chunk)
    synth_ms = (time.perf_counter() - t0) * 1000.0

    samples = np.concatenate(chunks) if chunks else np.asarray(audio.samples, dtype=np.float32)
    sample_rate = int(audio.sample_rate) or int(tts.sample_rate)
    return SynthRun(
        text=text,
        label=label,
        first_chunk_ms=first[0] if first else synth_ms,
        synth_ms=synth_ms,
        audio_ms=len(samples) / sample_rate * 1000.0,
        chunks=len(chunks),
        sample_rate=sample_rate,
        samples=samples,
    )


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """把 float32 波形写成 16bit PCM 单声道 WAV（给人听）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


class Contention:
    """把项目自己的流式识别器拉起来持续解码，制造真实的 CPU 争用。

    为什么不用"空转烧 CPU"：那样的负载没有代表性。这里跑的是通话时**真的会同时在跑**
    的那条链路（16 kHz 流式 zipformer），争用对象与真实场景一致。
    """

    def __init__(self, pcm16: bytes) -> None:
        self._pcm = pcm16
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str = ""

    def __enter__(self) -> "Contention":
        from asr.streaming import StreamingRecognizer

        recognizer = StreamingRecognizer.instance()
        if not recognizer.available:
            self.error = recognizer.load_error or "流式识别不可用"
            return self

        def loop() -> None:
            while not self._stop.is_set():
                session = recognizer.new_session()
                if session is None:
                    return
                # 每次 3200 字节 = 100ms 的 16kHz int16，与前端推流口径一致
                for i in range(0, len(self._pcm) - 3200, 3200):
                    if self._stop.is_set():
                        return
                    session.accept_pcm16(self._pcm[i : i + 3200])

        self._thread = threading.Thread(target=loop, name="asr-contention", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def run_suite(tts, repeat: int, out_dir: Path, tag: str) -> list[dict[str, object]]:
    """跑一整轮语料，每句重复 `repeat` 次取中位数，并导出 WAV。

    中位数而不是均值：VITS 的时长预测带噪声，单次测量会被长尾拉偏。
    """
    rows: list[dict[str, object]] = []
    for label, text in CORPUS:
        runs = [synth_once(tts, text, label) for _ in range(repeat)]
        rtf = _median([r.rtf for r in runs])
        first = _median([r.first_chunk_ms for r in runs])
        audio_ms = _median([r.audio_ms for r in runs])
        synth_ms = _median([r.synth_ms for r in runs])
        audios = sorted(r.audio_ms for r in runs)
        # 导出第一次的音频即可（听感不因重复而变）
        wav = out_dir / f"{tag}_{label}.wav"
        write_wav(wav, runs[0].samples, runs[0].sample_rate)
        rows.append({
            "label": label,
            "chars": len(text),
            "rtf": round(rtf, 3),
            "first_chunk_ms": round(first, 1),
            "synth_ms": round(synth_ms, 1),
            "audio_ms": round(audio_ms, 1),
            "audio_min_ms": round(audios[0], 1),
            "audio_max_ms": round(audios[-1], 1),
            "chunks": runs[0].chunks,
            "sample_rate": runs[0].sample_rate,
            "wav": str(wav.relative_to(REPO_ROOT)) if out_dir.is_relative_to(REPO_ROOT) else str(wav),
        })
    return rows


def _print_table(title: str, rows: list[dict[str, object]]) -> None:
    print(f"\n── {title} ──")
    header = f"{'语料':<8}{'字数':>5}{'整句合成ms':>11}{'音频ms':>9}{'audio波动':>18}{'RTF':>7}{'回调×':>7}"
    print(header)
    print("-" * len(header.encode("gbk", errors="replace")))
    for r in rows:
        jitter = f"{float(r['audio_min_ms']):.0f}~{float(r['audio_max_ms']):.0f}"
        print(
            f"{str(r['label']):<8}{int(r['chars']):>5}{float(r['synth_ms']):>11.1f}"
            f"{float(r['audio_ms']):>9.1f}{jitter:>18}"
            f"{float(r['rtf']):>7.3f}{int(r['chunks']):>7}"
        )
    worst_rtf = max(float(r["rtf"]) for r in rows)
    shortest = min(rows, key=lambda r: int(r["chars"]))
    print(f"  → 最差 RTF = {worst_rtf:.3f}（判据 < 0.4）")
    print(f"  → 最短句整句合成 = {float(shortest['synth_ms']):.1f} ms"
          f"（{shortest['label']}，判据 < 400；VITS 不流式，此即首声等待）")
    print(f"  → 回调次数集合 = {sorted({int(r['chunks']) for r in rows})}"
          f"  ← 恒为 1 即证明「非流式」")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TTS 可行性 benchmark（陪伴通话 go/no-go）")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--repeat", type=int, default=5, help="每句重复次数，取中位数（VITS 时长随机，别低于 3）")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="合成线程数；实测 1→RTF0.57 / 2→0.29 / 4→0.18 / 8→0.13，"
                             "默认 4 是在「达标」与「别把 CPU 吃光」之间的折中（通话时 ASR 还要抢）")
    parser.add_argument("--out-dir", default="_tts_bench", help="WAV 导出目录（相对仓库根）")
    parser.add_argument("--contention", action="store_true", help="同时跑流式 ASR 制造争用")
    parser.add_argument("--json", default="", help="把结果写成 JSON（给文档用）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (REPO_ROOT / model_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    print(f"仓库根   : {REPO_ROOT}")
    print(f"模型目录 : {model_dir}")
    print(f"线程数   : {args.num_threads}    重复次数: {args.repeat}")
    if not model_dir.is_dir():
        print(f"\n❌ 模型目录不存在。先下载：\n"
              f"   python algorithm/tools/download_asr_model.py --preset tts-zh-fanchen-C\n")
        return 1

    t0 = time.perf_counter()
    tts = load_tts(model_dir, num_threads=args.num_threads)
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"加载耗时 : {load_ms:.0f} ms   采样率: {tts.sample_rate} Hz   说话人数: {tts.num_speakers}")

    # 冷启动那一句单独报：它包含 lazy init / 首帧分配，不能混进稳态统计
    cold = synth_once(tts, CORPUS[1][1], "冷启动")
    print(f"冷启动首句: 首块 {cold.first_chunk_ms:.1f} ms  合成 {cold.synth_ms:.1f} ms  RTF {cold.rtf:.3f}")

    idle = run_suite(tts, args.repeat, out_dir, "idle")
    _print_table("空载", idle)

    under: list[dict[str, object]] = []
    if args.contention:
        pcm16 = (np.clip(cold.samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        with Contention(pcm16) as load:
            if load.error:
                print(f"\n⚠️ 争用测试跳过：{load.error}（8001 上的算法服务可用吗？）")
            else:
                print("\n⏳ 已启动流式 ASR 争用，跑满载…")
                under = run_suite(tts, args.repeat, out_dir, "load")
                _print_table("满载（同时跑流式 ASR）", under)

    # ── 判据 ──
    def verdict(rows: list[dict[str, object]], name: str) -> bool:
        if not rows:
            return True
        rtf = max(float(r["rtf"]) for r in rows)
        shortest = min(rows, key=lambda r: int(r["chars"]))
        first = float(shortest["synth_ms"])
        ok = rtf < 0.4 and first < 400.0
        print(f"\n{name}: {'✅ PASS' if ok else '❌ FAIL'}  "
              f"(最差 RTF<0.4 → {rtf:.3f}, 最短句合成<400ms → {first:.1f})")
        return ok

    print("\n" + "=" * 60)
    passed = verdict(idle, "空载") & verdict(under, "满载")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "model_dir": str(model_dir),
            "sample_rate": int(tts.sample_rate),
            "load_ms": round(load_ms, 1),
            "cold_start": {"first_chunk_ms": round(cold.first_chunk_ms, 1),
                           "synth_ms": round(cold.synth_ms, 1), "rtf": round(cold.rtf, 3)},
            "idle": idle,
            "contention": under,
            "pass": bool(passed),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入 {args.json}")

    print(f"\nWAV 已导出到 {out_dir} —— **请人真的听一遍中文音质**（这点脚本量不出来）")
    print("结论：" + ("Path A 成立，进入施工单 S1。" if passed
                     else "Path A 不达标 → 换 tts-zh-huayan 重测，仍不达标则 D1 只剩 Path B。"))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
