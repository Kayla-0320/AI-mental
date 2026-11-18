"""读取官方Audio标注的时间边界，剔除情绪字段，不下载14GB音频媒体。"""
from __future__ import annotations

import json
import math
from pathlib import Path,PurePosixPath
import sys
import tarfile
from typing import Any

from huggingface_hub import HfFileSystem

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import OUTPUT,digest,save


def main() -> None:
    """只保存时间与说话人，未来标签不参与次序和窗口边界的确定。"""
    output=OUTPUT/"timestamps"
    output.mkdir(parents=True,exist_ok=True)
    if (output/"complete.json").exists():
        raise ValueError("拒绝覆盖时间标注工件")
    fs=HfFileSystem()
    remote="datasets/BAAI/Emotiontalk@adbc17fc944e8cf2873643906160c6ca0259ab61/Audio.tar"
    records={}
    with fs.open(remote,"rb",block_size=1024*1024) as source,tarfile.open(fileobj=source,mode="r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            key=PurePosixPath(member.name).stem
            value=json.loads(archive.extractfile(member).read())
            paragraph=value["paragraphs"]
            row={"id":key,"speaker":value["speaker_id"],"start":float(paragraph["startTime"]),
                 "end":float(paragraph["endTime"]),"duration":float(paragraph["duration"]),
                 "source_file_path":value["file_path"]}
            if key in records or not all(math.isfinite(row[field]) for field in ("start","end","duration")):
                raise ValueError("时间标记重复或非有限")
            if row["start"]<0 or row["end"]<=row["start"] or abs(row["end"]-row["start"]-row["duration"])>1e-5:
                raise ValueError("时间边界不一致")
            records[key]=row
            if len(records)%5000==0:
                print(json.dumps({"timestamp_records":len(records)}),flush=True)
            if len(records)==19250:
                break
    if len(records)!=19250:
        raise ValueError("官方时间标记没有覆盖全部样本")
    path=output/"records.jsonl"
    with path.open("x",encoding="utf-8") as handle:
        for key in sorted(records):
            handle.write(json.dumps(records[key],ensure_ascii=False)+"\n")
    sequences={}
    for key,row in records.items():
        pieces=key.split("_")
        sequences.setdefault("_".join(pieces[:3]),[]).append(row)
    order_conflicts=[]
    overlaps=[]
    for key,items in sequences.items():
        ordered=sorted(items,key=lambda row:int(row["id"].split("_")[-1]))
        for previous,current in zip(ordered,ordered[1:]):
            if current["start"]<previous["start"]:
                order_conflicts.append([previous["id"],current["id"]])
            if current["start"]<previous["end"]:
                overlaps.append({"previous":previous["id"],"current":current["id"],
                                 "seconds":previous["end"]-current["start"]})
    conversations={}
    for row in records.values():
        conversations.setdefault("_".join(row["id"].split("_")[:2]),[]).append(row)
    cross_conflicts=[]
    repeated_turns=[]
    for key,items in conversations.items():
        ordered=sorted(items,key=lambda row:int(row["id"].split("_")[-1]))
        turns=[int(row["id"].split("_")[-1]) for row in ordered]
        if len(set(turns))!=len(turns):
            repeated_turns.append(key)
        cross_conflicts.extend([[a["id"],b["id"]] for a,b in zip(ordered,ordered[1:]) if b["start"]<a["start"]])
    save(output/"cross_speaker_order_audit.json",{"conversations":len(conversations),
        "duplicate_utterance_numbers":repeated_turns,"number_start_time_conflicts":cross_conflicts,
        "source":"official annotation timeline;not instrument-verified hardware synchronization"})
    summary={"repo":"BAAI/Emotiontalk","revision":"adbc17fc944e8cf2873643906160c6ca0259ab61",
        "samples":len(records),"same_speaker_sequences":len(sequences),"records_sha256":digest(path),
        "filename_order_conflicts":order_conflicts,"same_speaker_interval_overlaps":overlaps,
        "emotion_values_stored":False,"media_duration_match_verified":False,
        "interpretation":"official annotation timeline;clip-duration verification and cross-speaker common-clock verification still required",
        "source":"Audio.tar/json paragraphs startTime/endTime/duration","code_sha256":digest(Path(__file__))}
    save(output/"complete.json",summary)
    print(json.dumps({"samples":len(records),"order_conflicts":len(order_conflicts),"overlaps":len(overlaps)}),flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"timestamps_failed":type(error).__name__,"details_redacted":True}),flush=True)
        raise SystemExit(1)
