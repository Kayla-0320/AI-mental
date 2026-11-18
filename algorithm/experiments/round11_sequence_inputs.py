"""真实媒体观测序列的因果窗口，不读取任何未来文本、状态或目标标签。"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def index_windows(records: list[dict[str,Any]],context: int,horizon: int,
                  minimum_lead_seconds: float = 0.) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    """以完整历史的最后结束时间为截止，目标必须在截止后才开始。"""
    if context<5 or horizon<1 or minimum_lead_seconds<0:
        raise ValueError("窗口需要至少5次历史、正跨度和非负提前量")
    if len({row["id"] for row in records})!=len(records):
        raise ValueError("序列样本ID重复")
    sequences: dict[tuple[str,str],list[dict[str,Any]]]=defaultdict(list)
    for row in records:
        if not np.isfinite([row["start"],row["end"]]).all() or row["end"]<=row["start"]:
            raise ValueError("非法时间边界")
        sequences[row["conversation"],row["speaker"]].append(row)
    windows=[]
    excluded=[]
    candidate_count=0
    for rows in sequences.values():
        rows.sort(key=lambda row:(row["start"],row["end"],row["id"]))
        for t in range(5,len(rows)-horizon+1):
            candidate_count+=1
            history=rows[max(0,t-context):t]
            target=rows[t+horizon-1]
            prefix=rows[:t]
            cutoff=max(row["end"] for row in prefix)
            lead=target["start"]-cutoff
            if lead<minimum_lead_seconds-1e-9:
                excluded.append({"target_id":target["id"],"lead_seconds":lead,
                    "reason":"overlap" if lead<0 else "insufficient_inference_lead"})
                continue
            if any(row["group"]!=target["group"] or row["split"]!=target["split"] for row in history):
                raise ValueError("同一个会话跨来源或划分")
            windows.append({"target_id":target["id"],"history_ids":[row["id"] for row in history],
                "history_end_times":[row["end"] for row in history],
                "history_durations":[row["end"]-row["start"] for row in history],
                "prefix_ids":[row["id"] for row in prefix],
                "prefix_end_times":[row["end"] for row in prefix],
                "cutoff":cutoff,"target_start":target["start"],"lead_seconds":lead,
                "conversation":target["conversation"],"group":target["group"],
                "split":target["split"],"speaker":target["speaker"],"horizon":horizon})
    return windows,{"candidates_after5past":candidate_count,"eligible":len(windows),
                    "excluded":excluded,"minimum_lead_seconds":minimum_lead_seconds}


def materialize(windows: list[dict[str,Any]],observations: dict[str,np.ndarray],
                context: int) -> dict[str,np.ndarray]:
    """未来观测不被索引；输入仅含历史概率/表示、已知时长和距截止的间隔。"""
    if not windows:
        raise ValueError("没有合格的序列窗口")
    widths={len(np.asarray(value)) for value in observations.values()}
    if len(widths)!=1:
        raise ValueError("历史观测宽度不一致")
    width=next(iter(widths))
    values=np.zeros((len(windows),context,width+3),np.float32)
    last=np.zeros((len(windows),width),np.float32)
    for i,window in enumerate(windows):
        identifiers=window["history_ids"][-context:]
        ends=window["history_end_times"][-context:]
        durations=window["history_durations"][-context:]
        if any(end>window["cutoff"]+1e-9 for end in window.get("prefix_end_times",ends)) or window["cutoff"]>window["target_start"]+1e-9:
            raise ValueError("窗口包含尚未完成的未来观测")
        for j,(key,end,duration) in enumerate(zip(identifiers,ends,durations),context-len(identifiers)):
            observed=np.asarray(observations[key],np.float32)
            if observed.shape!=(width,) or not np.isfinite(observed).all():
                raise ValueError("真实识别观测无效")
            values[i,j,:width]=observed
            values[i,j,width:]=[1.,np.log1p(duration),np.log1p(max(0.,window["cutoff"]-end))]
        last[i]=np.asarray(observations[identifiers[-1]],np.float32)
    return {"sequence":values,"last_observation":last,
        "target_ids":np.array([row["target_id"] for row in windows]),
        "conversations":np.array([row["conversation"] for row in windows]),
        "groups":np.array([row["group"] for row in windows]),
        "splits":np.array([row["split"] for row in windows]),
        "cutoff":np.array([row["cutoff"] for row in windows]),
        "target_start":np.array([row["target_start"] for row in windows]),
        "lead_seconds":np.array([row["lead_seconds"] for row in windows])}
