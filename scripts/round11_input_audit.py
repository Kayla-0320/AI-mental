"""第十一轮开发来源的原始媒体时间戳审计，不读取历史测试标签。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))
from data_quality.raw_signals import assess_av_alignment
from experiments.round11_common import ROOT, OUTPUT, digest, save

MEDIA = ROOT / "data/public/ch_sims_v2/supervised"
CACHE = ROOT / "_国赛优化/round3/model/ch_sims_v2/sample_features"


def presentation_span(times: list[float], durations: list[float]) -> dict[str, Any]:
    """以解码后的呈现时间估计覆盖区间，缺失末帧时长才使用中位间隔。"""
    if not times:
        return {"verified": False, "reason": "no_decoded_timestamps"}
    if not np.isfinite(times).all() or not np.isfinite(durations).all():
        return {"verified": False, "reason": "nonfinite_timestamps"}
    intervals = np.diff(times)
    if len(intervals) and np.any(intervals <= 0):
        return {"verified": False, "reason": "nonmonotonic_timestamps"}
    fallback = float(np.median(intervals[intervals > 0])) if np.any(intervals > 0) else 0.
    lengths = [duration if duration > 0 else fallback for duration in durations]
    end = max(stamp + duration for stamp, duration in zip(times, lengths))
    return {"verified": end > times[0], "start": times[0], "end": end,
            "duration": end-times[0], "decoded_frames": len(times),
            "duration_fallback_count": sum(duration <= 0 for duration in durations),
            "median_positive_interval": fallback,
            "largest_interval": float(max(intervals)) if len(intervals) else 0.}


def audit_one(item: dict[str, Any]) -> dict[str, Any]:
    """保留原音频/画面质量判定，仅替换可核验的容器时长测量方法。"""
    identifier = item["id"]
    with np.load(CACHE / f"{identifier}.npz", allow_pickle=False) as cached:
        old = json.loads(str(cached["raw_quality_json"]))
        media_sha = str(cached["media_sha256"])
    result: dict[str, Any] = {**item, "media_sha256": media_sha,
        "old_alignment": old["synchrony"], "audio_quality": old["audio"],
        "visual_quality": old["visual"]}
    try:
        path = MEDIA / f"{identifier}.mp4"
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            stream_duration = float(stream.duration*stream.time_base) if stream.duration else None
            times, durations = [], []
            for frame in container.decode(video=0):
                if frame.time is None:
                    raise ValueError("视频帧缺失呈现时间戳")
                times.append(float(frame.time))
                durations.append(float(frame.duration*frame.time_base) if frame.duration else 0.)
            video = presentation_span(times, durations)
            video["stream_duration"] = stream_duration
            video["advertised_frame_count"] = stream.frames
        audio_times, audio_lengths, decoded = [], [], 0
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            stream.codec_context.thread_count = 1
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                if frame.time is None:
                    raise ValueError("音频帧缺失呈现时间戳")
                audio_times.append(float(frame.time))
                audio_lengths.append(frame.samples/frame.sample_rate)
                decoded += sum(out.samples for out in resampler.resample(frame))
            decoded += sum(out.samples for out in resampler.resample(None))
        audio = presentation_span(audio_times, audio_lengths)
        audio["decoded_waveform_duration"] = decoded/16000
        result.update(video_pts=video, audio_pts=audio)
        reliable = video["verified"] and audio["verified"]
        # 容器尾部与解码时轴明显不符时不能凭更短的解码结果放行。
        if stream_duration is not None:
            reliable = reliable and abs(video["duration"]-stream_duration) <= max(.25, .1*stream_duration)
        if not reliable:
            result.update(new_mask=item["old_mask"], new_quality=item["old_quality"],
                          measurement_verified=False, measurement_issue="unverified_or_incomplete_pts")
            return result
        origin = min(video["start"], audio["start"], 0.)
        alignment = assess_av_alignment(audio["decoded_waveform_duration"], video["duration"],
            audio_start=audio["start"]-origin, video_start=video["start"]-origin)
        observed = decoded >= 512
        mask = [True, bool(observed and old["audio"]["usable"] and alignment.usable),
                bool(old["visual"]["usable"] and alignment.usable)]
        quality = [1., old["audio"]["quality_score"]*float(mask[1]),
                   old["visual"]["quality_score"]*float(mask[2])]
        result.update(new_alignment=asdict(alignment), new_mask=mask, new_quality=quality,
                      measurement_verified=True)
    except (OSError, ValueError, IndexError, av.error.FFmpegError) as exc:
        result.update(new_mask=item["old_mask"], new_quality=item["old_quality"],
                      measurement_verified=False, measurement_issue=type(exc).__name__)
    return result


def main() -> None:
    """全量审计开发集；断点记录只允许同一输入和代码继续。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = OUTPUT / "input_audit"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        raise ValueError("拒绝覆盖已完成审计")
    items = []
    for split in ("train", "selection"):
        with np.load(ROOT / f"_国赛优化/round5/text_features/{split}.npz", allow_pickle=False) as data:
            for key, mask, quality in zip(data["sample_ids"], data["modality_mask"], data["quality"]):
                items.append({"id": str(key), "split": split, "old_mask": mask.tolist(),
                              "old_quality": quality.tolist()})
    plan = {"scope": "train+selection only; no labels read", "n": len(items),
        "code_sha256": digest(Path(__file__)), "thresholds": {"absolute_seconds": .25, "relative_duration": .1},
        "change": "decoded presentation timestamps replace OpenCV frame_count/fps; original audio/visual quality retained"}
    lock = output / "plan.json"
    if lock.exists():
        if json.loads(lock.read_text("utf-8")) != plan:
            raise ValueError("审计代码或输入计划发生变化")
    else:
        save(lock, plan)
    partial = output / "records.jsonl"
    records = [json.loads(line) for line in partial.read_text("utf-8").splitlines()] if partial.exists() else []
    done = {row["id"] for row in records}
    if len(done) != len(records) or not done <= {row["id"] for row in items}:
        raise ValueError("断点ID不符合计划")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, partial.open("a", encoding="utf-8") as handle:
        for row in pool.map(audit_one, [row for row in items if row["id"] not in done]):
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            handle.flush()
            records.append(row)
            if len(records) % 100 == 0 or len(records) == len(items):
                print(json.dumps({"audited": len(records), "total": len(items),
                                  "seconds": round(time.monotonic()-started, 1)}), flush=True)
    by_id = {row["id"]: row for row in records}
    ordered = [by_id[row["id"]] for row in items]
    summary = {**plan, "measurement_verified": sum(row["measurement_verified"] for row in ordered),
        "restored_audio": sum(not row["old_mask"][1] and row["new_mask"][1] for row in ordered),
        "restored_visual": sum(not row["old_mask"][2] and row["new_mask"][2] for row in ordered),
        "newly_rejected_audio": sum(row["old_mask"][1] and not row["new_mask"][1] for row in ordered),
        "newly_rejected_visual": sum(row["old_mask"][2] and not row["new_mask"][2] for row in ordered),
        "old_audio_rejected": sum(not row["old_mask"][1] for row in ordered),
        "remaining_audio_rejected": sum(not row["new_mask"][1] for row in ordered),
        "measurement_issues": dict(Counter(row.get("measurement_issue", "") for row in ordered if not row["measurement_verified"])),
        "records_sha256": digest(partial), "limitation": "timestamp consistency is not lip-sync or semantic alignment gold"}
    np.savez_compressed(output / "development_masks.npz", sample_ids=np.array([row["id"] for row in ordered]),
        modality_mask=np.array([row["new_mask"] for row in ordered], bool),
        quality=np.array([row["new_quality"] for row in ordered], np.float32))
    save(output / "complete.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
