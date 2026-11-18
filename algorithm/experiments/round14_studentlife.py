"""核对 StudentLife 的 PAM 效价映射和美国东部日期，不使用损坏的压力转换表。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import requests
import rdata

from experiments.round14_common import DATA, OUT, save, sha256

# 逐项抄录维护者 PAM_categorise，先映射1..16再按4个上界分箱。
VALENCE_ORDER = np.array([1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16])


def pam_valence(pictures: np.ndarray) -> np.ndarray:
    """严格采用维护者1–4效价编码，拒绝无效图片序号。"""
    values = np.asarray(pictures, dtype=float)
    if not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any((values < 1) | (values > 16)):
        raise ValueError("无效 PAM 图片序号")
    return (VALENCE_ORDER[values.astype(int) - 1] - 1) // 4 + 1


def studentlife_daily() -> tuple[pd.DataFrame, dict[str, Any]]:
    """只保留匿名ID、时间和PAM效价，日内多次取均值，保留无响应日缺测。"""
    path = DATA / "ema_rds" / "PAM.Rds"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        frame = rdata.read_rds(path)
        stress = rdata.read_rds(DATA / "ema_rds" / "Stress.Rds")
    required = ["picture_idx", "timestamp", "uid"]
    if set(frame.columns) != set(required): raise ValueError("PAM转换表字段变化，需重新审核")
    if frame[required].isna().any().any(): raise ValueError("PAM关键字段缺失")
    released_rows = len(frame)
    duplicate_rows = int(frame.duplicated(["uid", "timestamp"], keep=False).sum())
    conflict = frame.groupby(["uid", "timestamp"], observed=True).picture_idx.transform("nunique") > 1
    quarantined_rows = int(conflict.sum())
    # 同人同秒若图片冲突则全组隔离；同值完全重复只保留一条，不任意选结局。
    frame = frame.loc[~conflict].drop_duplicates(["uid", "timestamp", "picture_idx"]).copy()
    raw_sha = sha256(path)
    code_path = DATA / "SL_tbl_trans.R"
    code = code_path.read_text(encoding="utf-8")
    expected = "v1 <- c(1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16)"
    if expected not in code: raise ValueError("原始PAM映射未核实")
    frame["valence"] = pam_valence(frame.picture_idx.to_numpy())
    # 转换版timestamp仍是Unix秒；不能按UTC自然日合并美国本地晚间响应。
    dates = pd.to_datetime(frame.timestamp, unit="s", utc=True).dt.tz_convert("America/New_York").dt.date
    origin = min(dates)
    frame["day"] = np.array([(date - origin).days for date in dates])
    frame["person"] = frame.uid.astype(str)
    frame["value"] = (frame.valence - 1) / 3
    daily = frame.groupby(["person", "day"], observed=True).value.mean().reset_index()
    numeric_null = pd.to_numeric(stress["null"], errors="coerce") if "null" in stress else pd.Series(dtype=float)
    audit = {"stress_rows": len(stress), "stress_columns": list(stress.columns),
        "stress_nonmissing_null_values": int(stress["null"].notna().sum()) if "null" in stress else None,
        "stress_numeric_1_5_values": int(numeric_null.isin([1, 2, 3, 4, 5]).sum()),
        "stress_excluded": True, "reason": "no level column; mixed null contains location strings or old responses; cannot recover original stress without raw schema",
        "pam_released_rows": released_rows, "pam_duplicate_key_rows": duplicate_rows,
        "pam_conflicting_timestamp_rows_quarantined": quarantined_rows,
        "pam_exact_duplicate_extras_removed": released_rows - quarantined_rows - len(frame),
        "duplicate_policy": "quarantine all conflicting person/timestamp groups; collapse exact duplicate records",
        "pam_rows": len(frame), "pam_people": int(frame.person.nunique()), "daily_rows": len(daily),
        "calendar_start": str(origin), "calendar_end": str(max(dates)),
        "daily_response_count": {str(k): int(v) for k, v in frame.groupby(["person", "day"]).size().value_counts().sort_index().items()},
        "pam_picture_counts": {str(k): int(v) for k, v in frame.picture_idx.value_counts().sort_index().items()},
        "pam_rds_sha256": raw_sha, "mapping_code_sha256": sha256(code_path),
        "mapping_code_url": "https://raw.githubusercontent.com/frycast/studentlife/master/R/SL_tbl_trans.R",
        "mapping": {str(index): int(value) for index, value in enumerate(pam_valence(np.arange(1, 17)), 1)},
        "excluded_columns": "all location/passive sensor data, full-course aggregates, survey outcomes, academic records",
        "release_metadata": json.loads((OUT / "data" / "studentlife_provenance.json").read_text(encoding="utf-8"))}
    save(OUT / "data" / "studentlife_field_audit.json", audit)
    info = {"source": "https://zenodo.org/records/3529253", "license": "CC BY 4.0", "pam_sha256": raw_sha,
        "source_status": "new to project experiments; previously only catalogued; original foundation-model training overlap unknown",
        "target": "next calendar-day average observed PAM valence coded1–4; ordinal spacing is an explicit modeling convention",
        "scale": "1–4; errors in category-spacing units, not 0–100 clinical scores", "timezone": "America/New_York",
        "origin_calendar_day": str(origin), "mapping_code_sha256": sha256(code_path),
        "population": "original university-student cohort; no adolescent or clinical transfer claim",
        "split_status": "new source with own train/validation/test; test label predictions produced only after validation selection lock",
        "stress_conversion_excluded": True}
    return daily, info


if __name__ == "__main__":
    from experiments.round14_longitudinal import run_dataset
    data, info = studentlife_daily()
    run_dataset("studentlife_calendar_days", data, info, 3., 1 / 3)
