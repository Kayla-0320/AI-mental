"""统一外部表征与封存的目标边界，供跨数据集与序列实验复用。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from experiments.round11_common import ROOT,OUTPUT,digest
from experiments.round11_features import alignment_feature

SPLITS=("train","valid","test_inputs")


def read_rows(split: str) -> list[dict[str,Any]]:
    """测试输入文件从构建时就没有情绪目标。"""
    if split not in SPLITS:
        raise ValueError("未知外部划分")
    return [json.loads(line) for line in (OUTPUT/"corpus"/f"{split}.jsonl").read_text("utf-8").splitlines()]


def feature_file(component: str,split: str) -> dict[str,np.ndarray]:
    """所有分量都必须按同一官方ID顺序完整覆盖，不能按可用性删样本。"""
    with np.load(OUTPUT/"emotiontalk_features"/component/f"{split}.npz",allow_pickle=False) as source:
        return {name:source[name] for name in source.files}


def load_data(splits: tuple[str,...],quality: str = "pts",text_model: str = "7b") -> dict[str,np.ndarray]:
    """返回冻结模型需要的纯输入，标签只能经单独入口取得。"""
    if quality not in ("legacy","pts") or text_model not in ("7b","3b","light"):
        raise ValueError("未知外部输入配方")
    result=[]
    for split in splits:
        rows=read_rows(split)
        ids=np.array([row["id"] for row in rows])
        data={"sample_ids":ids,"raw_text":np.array([row["text"] for row in rows]),
              "source_groups":np.array([row["group"] for row in rows]),
              "conversation_ids":np.array([row["conversation"] for row in rows]),
              "speaker_ids":np.array([row["speaker"] for row in rows])}
        names=["signal_features","face_features","clip_features","speech_features",
               "sensevoice_deterministic_features","bge_features","auxiliary_features"]
        if text_model!="light":
            names.append("llm_large_features" if text_model=="7b" else "llm_features")
        for name in names:
            part=feature_file(name,split)
            if not np.array_equal(part.pop("sample_ids"),ids):
                raise ValueError(f"{name}外部ID未对齐")
            if name=="signal_features":
                data["audio"]=part["audio"]
                data["modality_mask"]=part["legacy_mask"] if quality=="legacy" else part["modality_mask"]
                data["quality"]=part["legacy_quality"] if quality=="legacy" else part["quality"]
            elif name=="face_features":
                data.update({field:part[field] for field in ("face_summary","deep_face","coverage")})
            elif name=="clip_features":
                data["visual"],data["text"]=part["visual"],part["text"]
            elif name=="speech_features":
                data.update(speech=part["speech"],speech_emotion=part["speech_emotion"],speech_available=part["available"])
            elif name=="sensevoice_deterministic_features":
                data.update(sensevoice=part["sensevoice"],sensevoice_query=part["query"][:,1,:],
                    sensevoice_emotion=part["emotion_logprobs"][:,1,:],sensevoice_available=part["available"],
                    asr_alignment=np.stack([alignment_feature(str(a),str(b)) for a,b in zip(data["raw_text"],part["transcript"])]))
            elif name=="bge_features":
                data["bge_cls"]=part["bge_cls"]
            elif name=="auxiliary_features":
                data["aux_mean"]=part["aux_mean"]
            else:
                data.update({field:part[field] for field in ("llm_last","llm_mean","llm_scores")})
        result.append(data)
    return {name:np.concatenate([part[name] for part in result]) for name in result[0]}


def targets(splits: tuple[str,...],task: str,evaluation_lock: Path | None = None) -> dict[str,int]:
    """开发目标可读；测试目标须先核对锁定清单内每个实际模型与配方摘要。"""
    if task not in ("M3","M_emotion7"):
        raise ValueError("未知情绪任务")
    if "test_inputs" in splits:
        if evaluation_lock is None:
            raise ValueError("外部test目标封存：必须先锁定评测模型与配方")
        lock=json.loads(evaluation_lock.read_text("utf-8"))
        if task not in lock.get("tasks",[]):
            raise ValueError("评测锁未授权当前任务的测试目标")
        if not lock.get("all_selection_complete") or not lock.get("model_files"):
            raise ValueError("评测锁缺少实际完成的选型与模型")
        for row in lock["model_files"]:
            path=Path(row["path"])
            if not path.is_file() or digest(path)!=row["sha256"]:
                raise ValueError("锁定模型文件缺失或摘要不符")
        for row in lock.get("selection_files",[]):
            if digest(Path(row["path"]))!=row["sha256"]:
                raise ValueError("锁定后选型配方发生变化")
    result={}
    ids=set()
    for split in splits:
        rows=read_rows(split)
        ids.update(row["id"] for row in rows)
        if split!="test_inputs":
            result.update({row["id"]:int(row[task]) for row in rows})
    if "test_inputs" in splits:
        test_ids={row["id"] for row in read_rows("test_inputs")}
        classes=("angry","disgusted","fearful","happy","neutral","sad","surprised")
        aliases={"anger":"angry","disgust":"disgusted","fear":"fearful","surprise":"surprised",
                 "happiness":"happy","sadness":"sad"}
        path=ROOT/"_国赛优化/round6/data/multimodal_annotations.jsonl"
        for line in path.read_text("utf-8").splitlines():
            row=json.loads(line)
            if row["id"] not in test_ids:
                continue
            annotation=row["annotation"]
            if task=="M3":
                label=int(np.sign(float(annotation["Continuous label_result"])))+1
            else:
                value=annotation["emotion_result"].lower()
                label=classes.index(aliases.get(value,value))
            result[row["id"]]=label
    if set(result)!=ids:
        raise ValueError("外部目标没有完整覆盖所选ID")
    return result
