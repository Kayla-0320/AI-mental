"""构造来源明确的开发监督与无标签外部测试输入，标签极性不从情绪类别推断。"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import ROOT,OUTPUT,digest,save

VALID={"G00001","G00012"}
TEST={"G00003","G00015"}
EMOTIONS=("angry","disgusted","fearful","happy","neutral","sad","surprised")
ALIASES={"anger":"angry","disgust":"disgusted","fear":"fearful","surprise":"surprised",
         "happiness":"happy","sadness":"sad"}


def normalized(text: str) -> str:
    """只规范字符和标点；长文本精确重合用作复核标记。"""
    return "".join(c.lower() for c in text if c.isalnum())


def emotion(value: str) -> int:
    """七类情绪只做拼写别名统一，独立于三类极性。"""
    return EMOTIONS.index(ALIASES.get(value.lower(),value.lower()))


def main() -> None:
    """只发布train/valid标签分布，外部test仅输出模型可见输入。"""
    output=OUTPUT/"corpus"
    if (output/"complete.json").exists():
        raise ValueError("拒绝覆盖已冻结的外部来源清单")
    output.mkdir(parents=True,exist_ok=True)
    source=ROOT/"_国赛优化/round6/data"
    text={row["id"]:row["annotation"] for row in map(json.loads,(source/"text_annotations.jsonl").read_text("utf-8").splitlines())}
    multimodal={}
    # 测试ID在提取目标值之前被过滤；本脚本不生成外部test标签文件。
    for line in (source/"multimodal_annotations.jsonl").read_text("utf-8").splitlines():
        item=json.loads(line)
        if item["id"].split("_")[0] not in TEST:
            multimodal[item["id"]]=item["annotation"]
    rows={"train":[],"valid":[],"test_inputs":[]}
    for key in sorted(text):
        parts=key.split("_")
        split="test_inputs" if parts[0] in TEST else "valid" if parts[0] in VALID else "train"
        item={"id":key,"group":parts[0],"conversation":"_".join(parts[:2]),
              "speaker":text[key]["speaker_id"],"turn":int(parts[-1]),"text":text[key]["content"],
              "media_relative_path":f"clips/{key}.mp4"}
        if item["speaker"]!=parts[-2]:
            raise ValueError("说话人标记与文件名不一致")
        if split!="test_inputs":
            label=multimodal[key]
            continuous=float(label["Continuous label_result"])
            if not -2<=continuous<=2:
                raise ValueError("官方连续值越界")
            item.update(M3=int(np.sign(continuous))+1,M_continuous=continuous/2,
                M_emotion7=emotion(label["emotion_result"]),T_emotion7=emotion(text[key]["emotion_result"]))
        rows[split].append(item)
    expected={"train":15413,"valid":1908,"test_inputs":1929}
    if {key:len(value) for key,value in rows.items()}!=expected:
        raise ValueError("官方划分数量不符")
    for split,items in rows.items():
        with (output/f"{split}.jsonl").open("x",encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item,ensure_ascii=False)+"\n")
    ch_text={}
    for split in ("train","selection"):
        with np.load(ROOT/f"_国赛优化/round5/text_features/{split}.npz",allow_pickle=False) as cached:
            ch_text.update({str(key):str(value) for key,value in zip(cached["sample_ids"],cached["raw_text"])})
    reverse={}
    for key,value in ch_text.items():
        normalized_text=normalized(value)
        if len(normalized_text)>=12:
            reverse.setdefault(normalized_text,[]).append(key)
    overlaps=[]
    for split,items in rows.items():
        for item in items:
            value=normalized(item["text"])
            if value in reverse:
                overlaps.append({"emotiontalk_id":item["id"],"split":split,"ch_development_ids":reverse[value],
                                 "normalized_characters":len(value)})
    summary={"repo":"BAAI/Emotiontalk","revision":"adbc17fc944e8cf2873643906160c6ca0259ab61",
        "official_split_counts":expected,"test_label_values_used":False,"test_label_distribution_reported":False,
        "label_rule":"human Continuous label_result sign => M3; /2 only for intensity; no emotion7-to-valence mapping",
        "development_M3_counts":{split:dict(Counter(item["M3"] for item in rows[split])) for split in ("train","valid")},
        "development_emotion7_counts":{split:dict(Counter(item["M_emotion7"] for item in rows[split])) for split in ("train","valid")},
        "groups":{split:sorted({item["group"] for item in items}) for split,items in rows.items()},
        "conversations":{split:len({item["conversation"] for item in items}) for split,items in rows.items()},
        "long_exact_text_overlap_with_CH_development":overlaps,
        "overlap_rule":"normalized alphanumeric exact match at least12characters;flags are not proof of copied media;no silent sample deletion",
        "upstream_files_sha256":{name:digest(source/name) for name in ("text_annotations.jsonl","multimodal_annotations.jsonl","acquisition.json")},
        "files_sha256":{split:digest(output/f"{split}.jsonl") for split in rows},
        "limitations":["project previously used EmotionTalk emotion7 and oracle-state sequences",
                       "source groups are recording groups;actor identities are not verified disjoint",
                       "media hash and near-duplicate audit pending full media acquisition",
                       "test only two recording groups;dialogue bootstrap is conditional on these groups"],
        "code_sha256":digest(Path(__file__))}
    save(output/"complete.json",summary)
    print(json.dumps({key:value for key,value in summary.items() if key in
        ("official_split_counts","development_M3_counts","conversations","long_exact_text_overlap_with_CH_development")},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
