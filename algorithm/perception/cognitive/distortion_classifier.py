"""C2D2 认知扭曲 8 分类 —— 推理模块。

定位
----
``perception/cognitive/cognitive_distortion.py`` 目前是**纯关键词子串匹配**
（`if kw in text`），没有语义能力，误报漏报都不可控。本模块用微调后的
RoBERTa（``chinese-roberta-wwm-ext`` + 8 类分类头）作为「扭曲」轴的判定源。

与场景检索的分工（两根**正交**的轴）::

    场景（情境）= 发生了什么   → intervention.scene_retrieval
                                 决定提问「措辞贴合什么处境」
    扭曲（标签）= 怎么解读的   → 本模块
                                 决定提问「往哪个方向走」

诚实边界（必须与方案一起声明，不得只说"准确率"）
----------------------------------------------
    - 8 类官方 test **acc ≈ 0.68–0.69 / macro-F1 ≈ 0.65–0.66**，不是 85%。
    - 塌缩为「有扭曲 / 无扭曲」二分类时 **acc ≈ 0.94**，这才是可靠的判定粒度。
    - 逐类 F1 差距极大：非扭曲 0.87、乱贴标签 0.80、读心术 0.77 … 算命仅 0.37。
    - 因此预测结果只作 **prompt 的软提示，不是诊断结论**；注入文本中已要求
      模型不得说出扭曲类型名称（AGENTS.md 禁止诊断性语言）。

设计约束：
    - **延迟加载**：权重约 400MB，import 时不加载，首次 ``predict`` 才载入。
    - **永不抛异常**：``predict_distortion_safe`` 失败即返回 None，
      保证 ``/smart-chat`` 主流程不因模型不可用而中断。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 检查点目录解析顺序：**仓库内 artifacts 优先**（随代码分发），
# 再退到开发机的训练产物目录。parents[2] 即 algorithm/。
_DEFAULT_CKPT_DIRS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[2] / "artifacts" / "cognitive_distortion_ckpt",
    Path(r"D:\Develop\AIC\data\training_runs\cognitive_distortion\ckpt"),
)

# 「非扭曲」是唯一代表"没有认知扭曲"的类别名
NON_DISTORTED_ZH = "非扭曲"


@dataclass(frozen=True)
class DistortionPrediction:
    """单条文本的扭曲预测结果。

    Attributes:
        label: 预测类别 id。
        label_zh: 预测类别中文名。
        confidence: 该类别概率。
        probs: 全部类别的概率（与 ``id2label`` 同序）。
        is_distorted: 是否判定为「存在认知扭曲」。
        margin: 最高概率与次高概率之差；越小说明模型越不确定。
    """

    label: int
    label_zh: str
    confidence: float
    probs: tuple[float, ...]
    is_distorted: bool
    margin: float


def resolve_ckpt_dir() -> Optional[Path]:
    """定位可用的检查点目录。

    Returns:
        Optional[Path]: 含 ``config.json`` 与 ``label_map.json`` 的目录；未找到则 None。
    """
    env_dir = os.environ.get("DISTORTION_CKPT_DIR", "").strip()
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(_DEFAULT_CKPT_DIRS)

    for path in candidates:
        if (path / "config.json").exists() and (path / "label_map.json").exists():
            return path
    return None


class DistortionClassifier:
    """认知扭曲 8 分类器（惰性加载）。"""

    def __init__(self, ckpt_dir: str | Path) -> None:
        """
        Args:
            ckpt_dir: 含 ``config.json`` / ``label_map.json`` / 权重的目录。
        """
        self._ckpt_dir = Path(ckpt_dir)
        label_map = json.loads(
            (self._ckpt_dir / "label_map.json").read_text(encoding="utf-8")
        )
        self._id2label: dict[int, str] = {
            int(k): v for k, v in label_map["id2label"].items()
        }
        self._max_len = int(label_map.get("max_len", 160))
        self._meta = {
            "best_epoch": label_map.get("best_epoch"),
            "val_macro_f1": label_map.get("val_macro_f1"),
            "encoder": label_map.get("encoder"),
        }

        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    # ----------------------------------------------------------
    # 属性
    # ----------------------------------------------------------

    @property
    def ckpt_dir(self) -> Path:
        """当前检查点目录。"""
        return self._ckpt_dir

    @property
    def num_labels(self) -> int:
        """类别数。"""
        return len(self._id2label)

    @property
    def id2label(self) -> dict[int, str]:
        """id → 中文类别名。"""
        return dict(self._id2label)

    @property
    def meta(self) -> dict:
        """训练元信息（最优 epoch、验证集 macro-F1、编码器）。"""
        return dict(self._meta)

    # ----------------------------------------------------------
    # 加载
    # ----------------------------------------------------------

    @classmethod
    def load(cls, ckpt_dir: Optional[str | Path] = None) -> "DistortionClassifier":
        """加载分类器。

        Args:
            ckpt_dir: 检查点目录；默认自动解析。

        Returns:
            DistortionClassifier: 分类器实例（权重仍为惰性加载）。

        Raises:
            FileNotFoundError: 找不到检查点。
        """
        if ckpt_dir is None:
            resolved = resolve_ckpt_dir()
            if resolved is None:
                raise FileNotFoundError(
                    "未找到认知扭曲分类器检查点。请用 DISTORTION_CKPT_DIR 指定目录，"
                    "或先运行 tools/train_cognitive_distortion.py --mode finetune 生成。"
                )
            ckpt_dir = resolved
        return cls(ckpt_dir)

    def _ensure_loaded(self) -> None:
        """首次推理时加载权重（延迟加载）。"""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from shared.model_lock import MODEL_LOAD_LOCK

        with MODEL_LOAD_LOCK:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._ckpt_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(self._ckpt_dir))
        model.eval()
        model.to(self._device)
        self._model = model

    # ----------------------------------------------------------
    # 推理
    # ----------------------------------------------------------

    def predict_proba(self, text: str) -> np.ndarray:
        """返回 8 类概率向量。

        Args:
            text: 待判定文本。

        Returns:
            np.ndarray: shape ``(num_labels,)`` 的概率（softmax 后）。
        """
        self._ensure_loaded()

        import torch

        enc = self._tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self._max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits
        return torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)

    def predict(self, text: str) -> DistortionPrediction:
        """预测单条文本的认知扭曲类别。

        Args:
            text: 待判定文本。

        Returns:
            DistortionPrediction: 预测结果。
        """
        probs = self.predict_proba(text)
        order = np.argsort(-probs)
        label = int(order[0])
        top = float(probs[label])
        second = float(probs[order[1]]) if len(order) > 1 else 0.0
        label_zh = self._id2label.get(label, f"UNKNOWN_{label}")

        return DistortionPrediction(
            label=label,
            label_zh=label_zh,
            confidence=top,
            probs=tuple(float(p) for p in probs),
            is_distorted=(label_zh != NON_DISTORTED_ZH),
            margin=top - second,
        )


# ============================================================
# 全局单例 + 容错入口
# ============================================================

_classifier: Optional[DistortionClassifier] = None
_load_failed = False


def get_distortion_classifier() -> Optional[DistortionClassifier]:
    """获取全局分类器（惰性单例）。

    检查点缺失或加载失败时返回 ``None`` 而不抛异常，
    保证 ``/smart-chat`` 主流程不中断。

    Returns:
        Optional[DistortionClassifier]: 分类器；不可用时为 None。
    """
    global _classifier, _load_failed

    if _classifier is not None:
        return _classifier
    if _load_failed:
        return None

    resolved = resolve_ckpt_dir()
    if resolved is None:
        logger.info(
            "认知扭曲分类器检查点未找到，跳过模型判定（仍可用场景检索的标签投票）。"
        )
        _load_failed = True
        return None

    try:
        _classifier = DistortionClassifier(resolved)
    except Exception as e:  # noqa: BLE001
        logger.warning("认知扭曲分类器加载失败（%s）: %s", resolved, e)
        _load_failed = True
        return None

    logger.info(
        "认知扭曲分类器已就绪：%d 类，检查点=%s，训练元信息=%s",
        _classifier.num_labels, resolved, _classifier.meta,
    )
    return _classifier


def reset_distortion_classifier() -> None:
    """清空单例缓存（测试用）。"""
    global _classifier, _load_failed
    _classifier = None
    _load_failed = False


def predict_distortion_safe(text: str) -> Optional[DistortionPrediction]:
    """容错版预测：任何异常都退化为 ``None``。

    Args:
        text: 待判定文本。

    Returns:
        Optional[DistortionPrediction]: 预测结果；不可用时为 None。
    """
    if not (text or "").strip():
        return None
    try:
        classifier = get_distortion_classifier()
        if classifier is None:
            return None
        return classifier.predict(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("认知扭曲预测失败，已跳过: %s", e)
        return None


def format_distortion_hint(prediction: DistortionPrediction) -> str:
    """把预测结果格式化为可注入 prompt 的提示串。

    只产出「倾向」表述，不产出诊断结论。

    Args:
        prediction: 模型预测结果。

    Returns:
        str: 形如 ``过度泛化（置信度 0.62）`` 的提示串。
    """
    return f"{prediction.label_zh}（置信度 {prediction.confidence:.2f}）"


__all__ = [
    "NON_DISTORTED_ZH",
    "DistortionClassifier",
    "DistortionPrediction",
    "format_distortion_hint",
    "get_distortion_classifier",
    "predict_distortion_safe",
    "reset_distortion_classifier",
    "resolve_ckpt_dir",
]
