"""真实公开原片上的受控信号故障回归；报告小样本区间，不推断现实检出率。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import av
import cv2
import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))
from data_quality.raw_signals import assess_audio_signal, assess_av_alignment, assess_visual_signal


def read_media(path: Path) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    """解码真实音频与均匀抽帧；缩放仅为有界质量代理，不作情绪真值。"""
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            chunks.extend(result.to_ndarray().reshape(-1) for result in resampler.resample(frame))
        chunks.extend(result.to_ndarray().reshape(-1) for result in resampler.resample(None))
    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.array([], dtype=np.float32)
    frames, timestamps = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else container.duration / 1e6
        count = stream.frames or max(1, int(duration * float(stream.average_rate)))
        indices = set(np.linspace(0, count - 1, 8, dtype=int).tolist())
        for index, frame in enumerate(container.decode(video=0)):
            if index in indices:
                frames.append(cv2.resize(frame.to_ndarray(format="rgb24"), (256, 256), interpolation=cv2.INTER_AREA))
                timestamps.append(float(frame.time))
            if len(frames) == len(indices):
                break
    return audio, np.stack(frames), timestamps, duration


def interval(successes: int, count: int) -> list[float]:
    """Clopper–Pearson双侧95%区间，避免把几个成功案例当低误差率证明。"""
    if count == 0:
        return [0.0, 1.0]
    return [0.0 if successes == 0 else float(beta.ppf(0.025, successes, count - successes + 1)),
            1.0 if successes == count else float(beta.ppf(0.975, successes + 1, count - successes))]


def run(receipt: Path, output_dir: Path) -> dict[str, Any]:
    """在四个原片注入固定强度的故障，检查预先定义的代理理由是否出现。"""
    samples = json.loads(receipt.read_text(encoding="utf-8"))["samples"]
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    for item in samples:
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("原片摘要不匹配")
        audio, frames, timestamps, video_duration = read_media(path)
        audio_base = assess_audio_signal(audio, 16000)
        visual_base = assess_visual_signal(frames, timestamps=timestamps)
        alignment_base = assess_av_alignment(len(audio) / 16000, video_duration)
        clipped = np.clip(audio * (4 / max(float(np.sqrt((audio ** 2).mean())), 1e-6)), -1, 1)
        noise = rng.normal(0, 0.15, len(audio)).clip(-1, 1)
        blurred = np.stack([cv2.GaussianBlur(frame, (51, 51), 15) for frame in frames])
        repeated = np.repeat(frames[:1], 8, axis=0)
        interventions = {
            "silence": (assess_audio_signal(np.zeros_like(audio), 16000), "near_silent_audio"),
            "severe_clipping": (assess_audio_signal(clipped, 16000), "clipping_proxy"),
            "white_noise_replacement": (assess_audio_signal(noise, 16000), "noise_like_spectrum_proxy"),
            "black_video": (assess_visual_signal(np.zeros_like(frames)), "underexposure_proxy"),
            "white_video": (assess_visual_signal(np.full_like(frames, 255)), "overexposure_proxy"),
            "gaussian_blur": (assess_visual_signal(blurred), "blur_proxy"),
            "repeated_frame": (assess_visual_signal(repeated), "repeated_frames_proxy"),
            "one_second_av_start_offset": (assess_av_alignment(len(audio) / 16000, video_duration, audio_start=1.0, video_start=0.0), "av_start_offset_exceeded"),
        }
        rows.append({"sample_id": item["id"], "sha256": item["sha256"],
                     "baseline": {"audio": asdict(audio_base), "visual": asdict(visual_base), "alignment": asdict(alignment_base)},
                     "injections": {name: {"detected_expected_proxy": expected in report.reason_codes, "expected_proxy": expected, "quality": asdict(report)} for name, (report, expected) in interventions.items()}})
    counts = {}
    if rows:
        for scenario in rows[0]["injections"]:
            success = sum(row["injections"][scenario]["detected_expected_proxy"] for row in rows)
            counts[scenario] = {"detected": int(success), "n": len(rows), "rate": success / len(rows), "clopper_pearson_95": interval(success, len(rows))}
    result = {"seed": 42, "dataset": "CH-SIMS-v2 four public raw-clip smoke samples", "n": len(rows),
              "scope": "controlled severe faults on real raw signals; not real-world quality detection sensitivity/specificity",
              "threshold_policy": "predefined engineering proxies; no threshold tuned on these clips",
              "face_presence": "not independently labeled; explicitly unverified",
              "alignment": "duration/start metadata only; no real lip-sync labels",
              "baseline_fault_false_positive_rate": None,
              "limitation": "baseline raw clips have no independent quality gold; no false-positive or real sensitivity claim",
              "counts": counts, "samples": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_signal_benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(rows), "counts": counts}, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.samples, args.output_dir)


if __name__ == "__main__":
    main()
