"""
推理接口 —— 提供统一的文本情感预测方法

使用方式：
    from perception.text_emotion.predict import predict

    result = predict("我今天心情很低落")
    # result = {"label": "depression", "confidence": 0.85, "probs": [...]}
"""
from __future__ import annotations

import functools
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from perception.text_emotion.config import (
    ID2LABEL,
    MAX_LENGTH,
    MODEL_SAVE_PATH,
    NUM_LABELS,
)
from perception.text_emotion.model import TextEmotionClassifier


# ============================================================
# 模型缓存（避免每次调用重新加载）
# ============================================================

@functools.lru_cache(maxsize=1)
def _load_model_and_tokenizer() -> tuple[TextEmotionClassifier, AutoTokenizer, str]:
    """加载训练好的模型和分词器（仅加载一次）

    Returns:
        (model, tokenizer, device)
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = TextEmotionClassifier(num_labels=NUM_LABELS)

    # 加载 state_dict
    state_dict_path = os.path.join(MODEL_SAVE_PATH, "pytorch_model.bin")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


# ============================================================
# 预测接口
# ============================================================

@torch.no_grad()
def predict(text: str) -> dict:
    """文本情感预测

    Args:
        text: 待预测的中文文本

    Returns:
        {
            "label": "anxiety",           # 预测类别
            "confidence": 0.85,           # 预测置信度
            "probs": [0.35, 0.10, ...]    # 5 类概率分布
        }
    """
    model, tokenizer, device = _load_model_and_tokenizer()

    # 编码
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 推理
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    # 取概率最大的类别
    pred_id = max(range(NUM_LABELS), key=lambda i: probs[i])
    pred_label = ID2LABEL[pred_id]
    confidence = probs[pred_id]

    return {
        "label": pred_label,
        "confidence": round(confidence, 4),
        "probs": [round(p, 4) for p in probs],
    }


def predict_batch(texts: list[str]) -> list[dict]:
    """批量文本情感预测

    Args:
        texts: 待预测的文本列表

    Returns:
        预测结果列表
    """
    return [predict(text) for text in texts]
