"""本地日记的个人跨日摘要，只描述本人自评，不把模型推断冒充自评。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.round14_common import OUT
from experiments.round14_human import analyze_study, read_csv, timestamp


def tracking_summary(directory: Path, participant: str, now: datetime | None = None) -> dict[str, Any]:
    """先复用同意和来源校验；缺测保留、长间隔重建参考，当前日不进入其个人基线。"""
    now = now or datetime.now(timezone.utc)
    consent = {row["participant_id"]: row for row in read_csv(directory / "consent.csv")}
    if participant not in consent: raise ValueError("没有该匿名参与者的同意记录")
    if consent[participant]["withdrawn_at"]: return {"status": "withdrawn", "retained_in_analysis": False}
    analyze_study(directory, directory / "analysis.json", now)
    rows = [row for row in read_csv(directory / "diary.csv") if row["participant_id"] == participant]
    if not rows: return {"status": "no_observations", "participant_id": participant, "clinical_alert": False}
    rows.sort(key=lambda row: timestamp(row["observed_at"]))
    local = timezone(timedelta(hours=8)); daily: dict[str, list[dict]] = {}
    for row in rows:
        day = str(timestamp(row["observed_at"]).astimezone(local).date())
        daily.setdefault(day, []).append(row)
    observations = [{"date": day, "mood": float(np.mean([float(row["mood_0_100"]) for row in values])),
        "stress": float(np.mean([float(row["stress_0_100"]) for row in values])), "responses": len(values)} for day, values in sorted(daily.items())]
    segments: list[list[dict]] = [[]]
    for observation in observations:
        if segments[-1] and (datetime.fromisoformat(observation["date"]) - datetime.fromisoformat(segments[-1][-1]["date"])).days > 3:
            segments.append([])
        segments[-1].append(observation)
    recent_segment = segments[-1]
    latest = recent_segment[-1]; reference = recent_segment[:-1][:7]
    sufficient = len(reference) >= 7 and (datetime.fromisoformat(reference[-1]["date"]) - datetime.fromisoformat(reference[0]["date"])).days <= 14
    baseline = {metric: float(np.mean([row[metric] for row in reference])) for metric in ("mood", "stress")} if sufficient else None
    return {"status": "stale" if now - timestamp(rows[-1]["observed_at"]) > timedelta(hours=48) else ("personal_reference_ready" if sufficient else "warming_up"),
        "participant_id": participant, "observed_days": len(observations), "record_count": len(rows), "daily_observations": observations,
        "current_day": latest, "personal_reference": baseline, "reference_dates": [row["date"] for row in reference] if sufficient else [],
        "current_minus_reference": {metric: latest[metric] - baseline[metric] for metric in baseline} if baseline else None,
        "new_segment_after_gap": len(segments) > 1, "unobserved_days_imputed": 0,
        "prediction": None, "prediction_reason": "StudentLife PAM模型尚未在项目0–100心情/压力量表和目标用户验证，不自动迁移",
        "rqi_trust_inferred": False, "clinical_alert": False, "automatic_intervention": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--participant", required=True)
    parser.add_argument("--directory", type=Path, default=OUT / "human_study")
    args = parser.parse_args(); print(json.dumps(tracking_summary(args.directory, args.participant), ensure_ascii=False))
