"""第九轮增补三：长序列预测的可行性探针。

对应 PPT 批注 00:11:40「提出多模态情绪长序列分析及预测方法」。

本脚本只做一件事：诚实地数清楚**现有数据到底支持多长的序列**。
- 公开纵向数据：`green exposure data.xlsx`（中国青少年，两波）
- 站内时间序列：`mood_records` / `assessments`（需要 PostgreSQL 在线）

判定标准（事先写死，避免事后解释）：
- 可做"长序列"：同一参与者 ≥5 个时间点，且时间戳可排序；
- 可做"一步随访"：2 个时间点；
- 两者都不满足：不做，如实写"数据不足"。
"""
from __future__ import annotations

import json
import math
import socket
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl

ROOT = Path(r"D:\Develop\AIC")
XLSX = ROOT / "data" / "public" / "chinese_adolescent_longitudinal_d8h6gbs4jk" / "green exposure data.xlsx"
OUTPUT = ROOT / "_国赛优化" / "round9_additions"
POSTGRES = ("127.0.0.1", 5432)
MIN_POINTS_FOR_SEQUENCE = 5
MIN_POINTS_FOR_STEP = 2


def probe_workbook() -> dict[str, Any]:
    workbook = openpyxl.load_workbook(XLSX, read_only=True)
    sheet = workbook["Sheet1"]
    rows = sheet.iter_rows(values_only=True)
    header = [c if isinstance(c, str) else "" for c in next(rows)]
    prefix = Counter()
    for name in header:
        for candidate in ("wave2",):
            if name.startswith(candidate):
                prefix[candidate] += 1
                break
    wave1_items = [n for n in header if n[:1] in "abcdef" and n[1:2].isdigit()]
    wave2_items = [n for n in header if n[:1] in "gh" and n[1:2].isdigit()]
    codes, followup_rows, rows_with_wave2 = set(), 0, 0
    code_column = header.index("code") if "code" in header else 0
    follow_column = header.index("wave2_followup") if "wave2_followup" in header else None
    wave2_columns = [header.index(name) for name in wave2_items]
    total = 0
    for row in rows:
        total += 1
        code = row[code_column]
        if code not in (None, ""):
            codes.add(code)
        if follow_column is not None and row[follow_column] in (1, "1", True):
            followup_rows += 1
        if any(row[i] not in (None, "") for i in wave2_columns if i < len(row)):
            rows_with_wave2 += 1
    return {
        "file": str(XLSX),
        "rows": total,
        "columns": len(header),
        "unique_participants": len(codes),
        "wave_columns": dict(prefix),
        "wave1_item_columns": len(wave1_items),
        "wave2_item_columns": len(wave2_items),
        "followup_flag_rows": followup_rows,
        "rows_with_wave2_items": rows_with_wave2,
        "time_points_per_participant": 2 if wave2_items else 1,
    }


def probe_postgres() -> dict[str, Any]:
    """只探测端口与连接；真正计数需要服务在线（server 侧 pg 客户端）。"""
    try:
        with socket.create_connection(POSTGRES, timeout=1.5):
            reachable = True
    except OSError:
        reachable = False
    return {
        "endpoint": f"{POSTGRES[0]}:{POSTGRES[1]}",
        "reachable": reachable,
        "query_when_online": (
            "select count(*) as users, "
            "sum(case when n>=3 then 1 else 0 end) as users_ge3, "
            "sum(case when n>=5 then 1 else 0 end) as users_ge5 "
            "from (select \"userId\", count(*) n from mood_records group by 1) t"
        ),
        "criteria": f"users_ge{MIN_POINTS_FOR_SEQUENCE} > 0 才可做长序列；否则只能做两波一步预测",
    }


def main() -> None:
    report: dict[str, Any] = {"schema": "round9_sequence_probe/1"}
    report["public_longitudinal"] = probe_workbook()
    report["internal_timeseries"] = probe_postgres()
    points = report["public_longitudinal"]["time_points_per_participant"]
    internal_ok = report["internal_timeseries"]["reachable"]
    report["verdict"] = {
        "max_horizon_public": f"{points} 个时间点（最多一步随访）",
        "sequence_feasible_public": points >= MIN_POINTS_FOR_SEQUENCE,
        "sequence_feasible_internal": "未验证（PostgreSQL 未在线）" if not internal_ok else "需执行查询后判定",
        "decision": ("可做长序列" if points >= MIN_POINTS_FOR_SEQUENCE
                     else ("只能做一步随访预测" if points >= MIN_POINTS_FOR_STEP else "数据不足")),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "sequence_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
