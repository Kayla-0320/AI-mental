"""
推理接口 —— 提供统一的文本情感预测方法

使用方式：
    from perception.text_emotion.predict import predict

    result = predict("我今天心情很低落")
    # result = {"label": "depression", "confidence": 0.85, "probs": [...],
    #           "engine": "roberta_valence", "mapping": "valence_only"}

引擎优先级（2026 更新）：
    1. **效价三分类 RoBERTa**（outputs/valence_model/）—— 在 EATD 484 条**人工标注**上
       微调，按被试分组 5 折 CV macro-F1 0.8588、acc 0.8677。见 valence_model.py。
    2. 本地训练的 5 类 RoBERTa（outputs/model/）—— 保留的历史路径。⚠️ 其训练数据由
       dataset.generate_fake_data() 合成，不应产出上线权重，详见该函数注释。
    3. jieba + 情感词典 —— 轻量降级方案。

⚠️ 标签空间不一致：路径 1 训练的是 3 类**效价**，而平台下游要 5 类。
   anxiety / anger 不是模型学出来的，是由词典线索推断的；无线索时负向质量全部
   归入 depression(悲伤)。因此返回结果里带 `mapping` 字段标明本次用了哪种映射，
   消费方不应把 probs[0]/probs[2] 当作模型预测值。详见 valence_model.map_valence_to_platform()。
"""
from __future__ import annotations

import functools
import json
import logging
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from perception.text_emotion.config import (
    ID2LABEL,
    LABEL2ID,
    MAX_LENGTH,
    MODEL_SAVE_PATH,
    NUM_LABELS,
)
from perception.text_emotion.model import TextEmotionClassifier
from perception.text_emotion.valence_model import (
    get_valence_model,
    map_valence_to_platform,
    dominant_label,
    MAP_TYPED,
    MAP_VALENCE_ONLY,
)

logger = logging.getLogger(__name__)


# ============================================================
# 模型缓存（避免每次调用重新加载）
# ============================================================

_trained_model_available = False


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


def _check_trained_model() -> bool:
    """检查本地训练的 RoBERTa 模型是否可用"""
    global _trained_model_available
    model_bin = os.path.join(MODEL_SAVE_PATH, "pytorch_model.bin")
    if os.path.isfile(model_bin):
        try:
            _load_model_and_tokenizer()
            _trained_model_available = True
            logger.info("文本情感模型（RoBERTa）加载成功")
        except Exception as e:
            logger.warning(f"RoBERTa 模型加载失败: {e}")
            _trained_model_available = False
    else:
        logger.info(f"本地训练模型不存在 ({MODEL_SAVE_PATH})，使用词典方案")
        _trained_model_available = False
    return _trained_model_available


# ============================================================
# jieba + 情感词典方案（替代规则引擎降级）
# ============================================================

# 5 类情感词典：[anxiety, depression, anger, neutral, positive]
# 每个词附带权重 (1.0~2.0)，权重越高信号越强
_SENTIMENT_LEXICON: dict[str, tuple[int, float]] = {
    # ── anxiety (index 0) ──
    "焦虑": (0, 1.5), "紧张": (0, 1.3), "担心": (0, 1.2),
    "害怕": (0, 1.4), "恐惧": (0, 1.5), "不安": (0, 1.2),
    "忐忑": (0, 1.3), "心慌": (0, 1.3), "惶恐": (0, 1.4),
    "惊慌": (0, 1.4), "忐忑不安": (0, 1.6), "心神不宁": (0, 1.5),
    "胸闷": (0, 1.2), "心悸": (0, 1.3), "失眠": (0, 1.3),
    "睡不着": (0, 1.3), "做噩梦": (0, 1.2), "喘不过气": (0, 1.4),
    "压力": (0, 1.1), "压力大": (0, 1.4), "压力山大": (0, 1.5),
    "崩溃": (0, 1.3), "要崩溃": (0, 1.5), "受不了": (0, 1.3),
    "担心": (0, 1.2), "担忧": (0, 1.3), "忧心": (0, 1.3),
    # ── depression (index 1) ──
    "难过": (1, 1.4), "悲伤": (1, 1.5), "伤心": (1, 1.5),
    "低落": (1, 1.3), "哭泣": (1, 1.4), "哭了": (1, 1.4),
    "想哭": (1, 1.3), "绝望": (1, 1.6), "痛苦": (1, 1.4),
    "孤独": (1, 1.3), "孤单": (1, 1.2), "无助": (1, 1.3),
    "疲惫": (1, 1.2), "心累": (1, 1.4), "好累": (1, 1.3),
    "累": (1, 1.0), "空虚": (1, 1.2), "迷茫": (1, 1.1),
    "无聊": (1, 0.8), "没意思": (1, 1.3), "没用": (1, 1.4),
    "不开心": (1, 1.3), "郁闷": (1, 1.2), "丧": (1, 1.2),
    "emo": (1, 1.3), "抑郁": (1, 1.5), "麻木": (1, 1.2),
    "心酸": (1, 1.3), "泪目": (1, 1.2), "忧愁": (1, 1.3),
    "苦闷": (1, 1.3), "哀伤": (1, 1.4), "悲痛": (1, 1.5),
    "有点累": (1, 1.0), "好疲惫": (1, 1.4), "很累": (1, 1.4),
    "太累了": (1, 1.5), "身心疲惫": (1, 1.5), "心力交瘁": (1, 1.5),
    # ── anger (index 2) ──
    "生气": (2, 1.4), "愤怒": (2, 1.6), "气死": (2, 1.5),
    "恼火": (2, 1.3), "恼怒": (2, 1.3), "火大": (2, 1.3),
    "讨厌": (2, 1.2), "烦": (2, 1.1), "烦死": (2, 1.4),
    "好烦": (2, 1.3), "好气": (2, 1.3), "暴躁": (2, 1.3),
    "恨": (2, 1.4), "厌恶": (2, 1.3), "恶心": (2, 1.2),
    "过分": (2, 1.1), "不公平": (2, 1.2), "凭什么": (2, 1.3),
    "憋屈": (2, 1.3), "窝火": (2, 1.3), "恼火": (2, 1.3),
    "暴怒": (2, 1.6), "怒火": (2, 1.5), "发脾气": (2, 1.3),
    # ── neutral (index 3) ── 中性词通常不加分，作为基线
    # ── positive (index 4) ──
    "开心": (4, 1.5), "快乐": (4, 1.5), "高兴": (4, 1.5),
    "幸福": (4, 1.6), "满意": (4, 1.3), "棒": (4, 1.3),
    "好棒": (4, 1.5), "喜欢": (4, 1.3), "爱": (4, 1.2),
    "谢谢": (4, 1.2), "感谢": (4, 1.3), "哈哈": (4, 1.3),
    "哈哈哈": (4, 1.5), "太好了": (4, 1.6), "真好": (4, 1.4),
    "不错": (4, 1.2), "优秀": (4, 1.4), "厉害": (4, 1.3),
    "加油": (4, 1.2), "期待": (4, 1.2), "希望": (4, 1.1),
    "兴奋": (4, 1.4), "激动": (4, 1.3), "欣慰": (4, 1.4),
    "安心": (4, 1.3), "放松": (4, 1.2), "舒服": (4, 1.2),
    "愉快": (4, 1.4), "欢乐": (4, 1.4), "喜悦": (4, 1.5),
    "自豪": (4, 1.3), "骄傲": (4, 1.2), "感恩": (4, 1.4),
    "美好": (4, 1.3), "甜": (4, 1.1), "温暖": (4, 1.2),
    "治愈": (4, 1.3), "感动": (4, 1.3), "惊喜": (4, 1.4),
    "乐观": (4, 1.4), "自信": (4, 1.3), "勇敢": (4, 1.2),
    "嘻嘻": (4, 1.3), "好开心": (4, 1.6), "好快乐": (4, 1.6),
    "超开心": (4, 1.7), "太棒了": (4, 1.7),
}

# 否定词列表（翻转后面情绪的方向）
_NEGATION_WORDS = {"不", "没", "没有", "别", "莫", "未", "非", "无法", "不能", "不会", "不太"}

# 程度副词（增强后面情绪强度）
_INTENSIFIERS = {
    "很": 1.3, "非常": 1.5, "特别": 1.5, "超级": 1.6, "极其": 1.6,
    "太": 1.4, "真": 1.3, "好": 1.3, "十分": 1.4, "格外": 1.4,
    "超": 1.5, "蛮": 1.2, "挺": 1.2, "比较": 1.1, "有点": 0.8,
    "有些": 0.8, "稍微": 0.7, "略微": 0.7,
}


def _lexicon_predict(text: str) -> dict:
    """基于 jieba 分词 + 情感词典的文本情感预测

    相比规则引擎的优势：
    1. 包含正面情绪词汇（规则引擎完全缺失）
    2. 使用 jieba 分词而非子串匹配，减少误匹配
    3. 支持否定词翻转（如 "不开心" → 负面）
    4. 支持程度副词调节（如 "非常开心" → 更强正面）

    Args:
        text: 待预测的中文文本

    Returns:
        {"label": str, "confidence": float, "probs": [5类概率], "engine": str}
    """
    import jieba

    words = list(jieba.cut(text))
    text_lower = text.lower().strip()

    # 初始化 5 类得分
    scores = [0.0] * NUM_LABELS  # [anxiety, depression, anger, neutral, positive]
    matched_words: list[str] = []

    # 逐词匹配词典
    has_negation = False  # 前一个词是否是否定词
    prev_intensifier = 1.0  # 程度副词倍数

    for i, word in enumerate(words):
        word_stripped = word.strip()

        # 检查否定词
        if word_stripped in _NEGATION_WORDS:
            has_negation = True
            continue

        # 检查程度副词
        if word_stripped in _INTENSIFIERS:
            prev_intensifier = _INTENSIFIERS[word_stripped]
            continue

        # 词典匹配
        if word_stripped in _SENTIMENT_LEXICON:
            class_idx, base_weight = _SENTIMENT_LEXICON[word_stripped]
            weight = base_weight * prev_intensifier

            if has_negation:
                # 否定翻转：正面→负面，负面→正面，中性不变
                if class_idx == 4:  # positive → depression
                    scores[1] += weight
                    matched_words.append(f"否定+{word_stripped}")
                elif class_idx in (1, 2):  # depression/anger → positive (减弱)
                    scores[4] += weight * 0.3
                    matched_words.append(f"否定+{word_stripped}")
                elif class_idx == 0:  # anxiety → 减弱
                    scores[3] += weight * 0.5
                    matched_words.append(f"否定+{word_stripped}")
                else:
                    scores[class_idx] += weight
                    matched_words.append(word_stripped)
                has_negation = False
            else:
                scores[class_idx] += weight
                matched_words.append(word_stripped)

            prev_intensifier = 1.0
        else:
            has_negation = False
            prev_intensifier = 1.0

    # 也检查子串匹配（捕获词典中的多字词组）
    for phrase, (class_idx, base_weight) in _SENTIMENT_LEXICON.items():
        if len(phrase) >= 2 and phrase in text_lower and phrase not in matched_words:
            scores[class_idx] += base_weight * 0.5
            matched_words.append(f"[{phrase}]")

    # 计算中性基线：如果没有匹配到任何情感词，中性概率高
    emotion_total = sum(scores)
    if emotion_total < 0.1:
        # 几乎没有情感信号 → 高概率中性
        probs = [0.05, 0.05, 0.05, 0.80, 0.05]
    else:
        # 添加中性基线（防止极端分布）
        scores[3] = max(0.05, 0.3 - emotion_total * 0.1)
        # softmax 归一化（温度参数控制锐度）
        temperature = 1.5
        exp_scores = [math.exp(s / temperature) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

    # 取概率最大的类别
    pred_id = max(range(NUM_LABELS), key=lambda i: probs[i])
    pred_label = ID2LABEL[pred_id]
    confidence = probs[pred_id]

    return {
        "label": pred_label,
        "confidence": round(confidence, 4),
        "probs": [round(p, 4) for p in probs],
        "engine": "lexicon",
        "matched": matched_words[:10],  # 最多保留 10 个匹配词
    }


# ============================================================
# 预测接口
# ============================================================

# 词典给出的「负向类型」线索低于此值就认为没有类型证据，
# 不再用它拆分负向质量（避免把中性基线 [0.05,0.05,0.05] 当成真实线索）
_MIN_TYPED_WEIGHT = 0.15


def _lexicon_typed_weights(text: str) -> Optional[list[float]]:
    """用词典取负向情绪类型的相对线索强度 (anxiety, depression, anger)。

    返回 None 表示「没有可用的类型证据」，调用方应走 valence_only 映射：
      * 词典不可用（如缺 jieba）→ None
      * 词典一个情感词都没匹配到 → None（此时它的输出恒为中性基线，不是线索）
      * 词典在负向三通道上的最大概率低于 _MIN_TYPED_WEIGHT → None
    """
    try:
        lex = _lexicon_predict(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("词典类型线索不可用: %s", exc)
        return None

    if not lex.get("matched"):
        return None
    probs = lex.get("probs") or []
    if len(probs) < NUM_LABELS:
        return None
    typed = [float(probs[0]), float(probs[1]), float(probs[2])]
    if max(typed) < _MIN_TYPED_WEIGHT:
        return None
    return typed


@torch.no_grad()
def predict(text: str) -> dict:
    """文本情感预测

    引擎优先级：效价 RoBERTa → 5 类 RoBERTa → jieba 词典。

    Args:
        text: 待预测的中文文本

    Returns:
        {
            "label": "depression",        # 平台 5 类标签
            "confidence": 0.85,
            "probs": [a, d, g, n, p],     # [焦虑, 抑郁, 愤怒, 中性, 积极]
            "engine": "roberta_valence",  # 实际使用的引擎
            "mapping": "valence_only",    # 3类→5类 的映射方式（仅效价引擎有）
            "valence": [neg, neu, pos],   # 模型原始效价分布（仅效价引擎有）
            "matched": [...],             # 词典匹配词
        }
    """
    # ── 优先级 1：效价三分类 RoBERTa（真实标注训练，CV macro-F1 0.8588）──
    vm = get_valence_model()
    if vm.available:
        try:
            p3 = vm.predict_valence(text)
            typed = _lexicon_typed_weights(text)
            probs5, mapping = map_valence_to_platform(
                float(p3[0]), float(p3[1]), float(p3[2]), typed_weights=typed,
            )
            label, conf = dominant_label(probs5)
            return {
                "label": label,
                "confidence": round(conf, 4),
                "probs": [round(p, 4) for p in probs5],
                "engine": "roberta_valence",
                "mapping": mapping,
                "valence": [round(float(x), 4) for x in p3],
                "matched": ([] if typed is None else ["词典类型线索"]),
                "n_seeds": vm.n_seeds,
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"效价 RoBERTa 推理失败，降级: {e}")

    # ── 优先级 2：本地 5 类 RoBERTa（历史路径；训练数据为合成数据，见 dataset.py）──
    if _trained_model_available:
        try:
            model, tokenizer, device = _load_model_and_tokenizer()
            encoding = tokenizer(
                text,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

            pred_id = max(range(NUM_LABELS), key=lambda i: probs[i])
            pred_label = ID2LABEL[pred_id]
            confidence = probs[pred_id]

            return {
                "label": pred_label,
                "confidence": round(confidence, 4),
                "probs": [round(p, 4) for p in probs],
                "engine": "roberta5",
            }
        except Exception as e:
            logger.warning(f"RoBERTa 推理失败，切换到词典方案: {e}")

    # ── 优先级 3：jieba + 情感词典 ──
    return _lexicon_predict(text)


def predict_batch(texts: list[str]) -> list[dict]:
    """批量文本情感预测

    Args:
        texts: 待预测的文本列表

    Returns:
        预测结果列表
    """
    return [predict(text) for text in texts]
