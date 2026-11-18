"""ESConv 支持策略描述模型；只学习数据中采用的策略，不推断有效性。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from shared import SupportStrategyPrediction


def context_text(messages: list[dict[str, str]], use_context: bool = True) -> str:
    """仅把已发生的角色与文本编码为特征，不读取待预测回复或结束后问卷。"""
    if not messages or len(messages) > 64:
        raise ValueError("需要1至64条过去消息")
    if any(set(m) != {"role", "content"} or m["role"] not in {"user", "assistant"}
           or not isinstance(m["content"], str) or len(m["content"]) > 4000 for m in messages):
        raise ValueError("过去消息的角色、文本或大小无效")
    if use_context:
        return "\n".join(f'{m["role"]}: {m["content"]}' for m in messages[-4:])[-8000:]
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


class SupportStrategyModel:
    """加载可追溯的轻量策略分类器；中文或分布外输入明确不支持。"""

    def __init__(self, directory: Path) -> None:
        m = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(m, dict) or m.get("schema") != "esconv_strategy.v1" or m.get("language") != "en":
            raise ValueError("策略任务或语言不匹配")
        if m.get("selection_split") != "selection" or m.get("calibration_split") != "calibration":
            raise ValueError("选择与校准来源不合法")
        if type(m.get("use_context")) is not bool:
            raise ValueError("策略上下文模式必须显式为布尔值")
        labels = m.get("labels")
        if not isinstance(labels, list) or not 2 <= len(labels) <= 64 or not all(isinstance(x, str) and x.strip() for x in labels) or len(set(labels)) != len(labels):
            raise ValueError("策略类别无效")
        weights = directory / "weights.npz"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != m["weights_sha256"]:
            raise ValueError("策略权重摘要不匹配")
        with np.load(weights, allow_pickle=False) as data:
            self.coef, self.intercept, idf = data["coef"], data["intercept"], data["idf"]
        vocab = m["vocabulary"]
        if (not isinstance(vocab, dict) or not 1 <= len(vocab) <= 100000
                or not all(isinstance(k, str) and type(v) is int for k, v in vocab.items())
                or set(vocab.values()) != set(range(len(vocab)))):
            raise ValueError("策略词表编号无效")
        if self.coef.shape != (len(labels), len(vocab)) or self.intercept.shape != (len(labels),) or idf.shape != (len(vocab),):
            raise ValueError("策略权重形状不匹配")
        if not all(x.dtype.kind in "fiu" and np.isfinite(x).all() for x in (self.coef, self.intercept, idf)) or (idf <= 0).any():
            raise ValueError("策略权重非有限或非法")
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True, vocabulary=vocab)
        self.vectorizer.idf_ = idf
        t = m.get("temperature")
        if type(t) not in (int, float) or not np.isfinite(t) or not .1 <= t <= 10:
            raise ValueError("策略温度无效")
        self.meta = m

    def predict(self, messages: list[dict[str, str]], language: str = "en") -> SupportStrategyPrediction:
        """返回行为描述标签；不自动把标签映射成支持动作或工具调用。"""
        text = context_text(messages, self.meta["use_context"])
        # 角色前缀 user:/assistant: 不是用户提供的语言证据。
        actual_content = "\n".join(m["content"] for m in messages[-4:]) if self.meta["use_context"] else context_text(messages, False)
        limits = ["describes_dataset_strategy_not_optimal_policy", "no_clinical_effect_evidence",
                  "english_research_dataset_not_validated_for_chinese_adolescents"]
        if language != "en" or re.search(r"[\u3400-\u9fff]", actual_content) or not re.search(r"[A-Za-z]{3}", actual_content):
            return SupportStrategyPrediction(False, "", [], self.meta["labels"], self.meta["weights_sha256"], language,
                                             limits, "unsupported_language_or_empty_evidence")
        x = self.vectorizer.transform([text])
        if x.nnz == 0:
            return SupportStrategyPrediction(False, "", [], self.meta["labels"], self.meta["weights_sha256"], language,
                                             limits, "out_of_vocabulary")
        z = (np.asarray(x @ self.coef.T)[0] + self.intercept) / self.meta["temperature"]
        z -= z.max()
        probabilities = np.exp(z) / np.exp(z).sum()
        return SupportStrategyPrediction(True, self.meta["labels"][int(probabilities.argmax())],
                                         probabilities.tolist(), self.meta["labels"], self.meta["weights_sha256"], language, limits)
