"""效价三分类 RoBERTa —— 接入平台 5 类文本情绪空间的适配层。

════════════════════════════════════════════════════════════════════════
这个文件为什么存在（必须先读）
════════════════════════════════════════════════════════════════════════
平台原有的 `predict.py` 声称有「本地训练的 RoBERTa（5 类）」主路径，但：
  * `config.MODEL_SAVE_PATH` 指向 `text_emotion/outputs/model/`，该目录一直是空的；
  * 该目录**即使填上**也训不出可信模型 —— `dataset.py` 的 `generate_fake_data()`
    用 `_FAKE_SAMPLES` 里手写的几十条文本，靠加前后缀、换程度副词凑到每类 80 条，
    在这样的合成数据上训 5 类分类器，产出的权重没有任何真实效度。

因此本项目改为接入**真实训练出来的效价三分类 RoBERTa**：
  * 数据：EATD-Corpus 484 条话语 / 162 人，逐条标注的字面情感（非文件夹名）；
  * 评估：按被试分组 5 折 CV，macro-F1 **0.8588 ± 0.0271**、acc 0.8677 ± 0.0245
    （对照：原字符级 TextCNN 0.6958、TF-IDF+LR 0.6989、手工词典 0.6203）；
  * 权重：`outputs/valence_model/seed{42,55,68}/`，由
    `tools/roberta_sentiment/train.py --full` 生成。

════════════════════════════════════════════════════════════════════════
⚠️ 标签来源的重大限制（不要把它读成「人工标注」）
════════════════════════════════════════════════════════════════════════
上述 484 条标签是**由 AI 模型（本项目开发用的编码智能体）逐条判读后生成的**，
不是人类标注者的标注。证据：生成脚本 `tools/textcnn_sentiment/merge_labels.py`
的文档字符串自述「本脚本把我（模型）对全部 485 条话语的字面情感人工标注合并进来」。

因此 0.8588 这个数字的确切含义是「模型对**另一模型所给标签**的复现程度」，
**不是**对真实语义极性的准确率。它证明的是「标签是自洽且可学的」，
不能证明「标签是对的」。

已知的旁证（非人工验证）：该标签与 EATD 文件夹名的一致率为 0.6095，
而文件夹名是情绪诱发题目、本身就不是情感标签，故这个数字两边都不干净。

**要让它成为可报告的效价识别能力，必须补人类标注**：
  * 最少：1 名真人标注 `data/labeling/round2/blind.csv` 的 200 条，
    得到「真人 vs 模型标签」一致率（这是 agreement，不是 inter-annotator κ）；
  * 严谨：2 名真人独立标注同一批（可用其中 100~200 条），才算真正的 IAA。
在此之前，本模块的输出只能当作**待人工校验的辅助信号**，不得用于临床或风险判定依据。

════════════════════════════════════════════════════════════════════════
标签空间不一致 —— 这是本适配层的核心问题，不允许含糊
════════════════════════════════════════════════════════════════════════
  * 本模型训练的是 **3 类效价**：[negative, neutral, positive]
  * 平台下游（risk / scale / fusion / 前端）要的是 **5 类**：
    [anxiety, depression, anger, neutral, positive]

两者**不是同一个标签空间**。EATD 只标了效价，没有「焦虑 / 抑郁 / 愤怒」的类型标签，
所以 5 类里的 anxiety 与 anger **不是学出来的**，只能由词典线索推断；
depression 是与负向效价直接对齐的通道，因此作为无类型线索时的兜底。

任何消费本模块输出的代码都必须知道：**probs[0]（焦虑）与 probs[2]（愤怒）是
词典推断值，不是模型预测值。** 详见 `map_valence_to_platform()` 的文档。
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# 模型自身的标签空间
VALENCE_LABELS: tuple[str, ...] = ("negative", "neutral", "positive")

# 平台 5 类顺序，必须与 config.LABEL2ID 保持一致
PLATFORM_LABELS: tuple[str, ...] = (
    "anxiety", "depression", "anger", "neutral", "positive",
)
IDX_ANXIETY, IDX_DEPRESSION, IDX_ANGER, IDX_NEUTRAL, IDX_POSITIVE = range(5)

MODEL_DIR = Path(__file__).resolve().parent / "outputs" / "valence_model"
DEFAULT_MAX_LEN = 160

# 映射方式标记，写进 evidence 供追溯
MAP_TYPED = "valence_x_lexicon_type"
MAP_VALENCE_ONLY = "valence_only"


# ============================================================
# 标签空间映射（纯函数，可独立测试）
# ============================================================
def map_valence_to_platform(
    p_neg: float,
    p_neu: float,
    p_pos: float,
    typed_weights: Optional[Sequence[float]] = None,
) -> tuple[list[float], str]:
    """把 3 类效价概率投影到平台 5 维空间。

    规则（显式写死，不做"看起来像 5 类"的模糊处理）：

    1. ``P(积极) → positive``、``P(中性) → neutral`` 直接搬运，不缩放、不加噪。
    2. ``P(消极)`` 是唯一无法一一对应的部分，按类型线索拆分：
       * ``typed_weights = (w_anxiety, w_depression, w_anger)`` 有正值时，
         按相对强度把 ``P(消极)`` 拆到三个通道，标记 ``MAP_TYPED``；
       * 无有效线索（缺省、全 0、全负）时，**全部给 depression(悲伤)**，
         标记 ``MAP_VALENCE_ONLY``。
    3. 为什么不平均分：把负向质量平摊给焦虑/愤怒，等于**凭空捏造型别**。
       悲伤是负向效价直接对应的情绪，把不确定的部分放在它上面，
       宁可"少说"（只知道消极、不知道是哪一种），也不要"编造"。
    4. 返回值之和恒为 1（对浮点误差做了归一化）。

    Args:
        p_neg, p_neu, p_pos: 模型输出的三类概率
        typed_weights: 词典给出的焦虑/抑郁/愤怒相对线索强度（非负）

    Returns:
        (长度为 5 的概率列表, 映射方式标记)
    """
    probs = [0.0] * len(PLATFORM_LABELS)
    probs[IDX_NEUTRAL] = max(0.0, float(p_neu))
    probs[IDX_POSITIVE] = max(0.0, float(p_pos))

    neg = max(0.0, float(p_neg))
    weights: list[float] = []
    if typed_weights is not None:
        weights = [max(0.0, float(w)) for w in typed_weights]
    total = sum(weights) if weights else 0.0

    if total <= 0.0:
        probs[IDX_DEPRESSION] = neg
        mapping = MAP_VALENCE_ONLY
    else:
        probs[IDX_ANXIETY] = neg * weights[0] / total
        probs[IDX_DEPRESSION] = neg * weights[1] / total
        probs[IDX_ANGER] = neg * weights[2] / total
        mapping = MAP_TYPED

    s = sum(probs)
    if s <= 0:
        # 理论上不会发生（三类概率之和为 1），保底防 NaN
        probs = [0.0, 0.0, 0.0, 1.0, 0.0]
        s = 1.0
    return [p / s for p in probs], mapping


def dominant_label(probs5: Sequence[float]) -> tuple[str, float]:
    """取概率最大的平台标签。"""
    idx = int(np.argmax(np.asarray(probs5, dtype=float)))
    return PLATFORM_LABELS[idx], float(probs5[idx])


# ============================================================
# 模型加载与推理
# ============================================================
class ValenceRoBERTa:
    """多随机种子集成的效价三分类 RoBERTa。"""

    def __init__(self, model_dir: Path | str = MODEL_DIR,
                 max_len: int = DEFAULT_MAX_LEN,
                 device: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir)
        self.max_len = max_len
        self._device = device
        self._models: list = []
        self._tokenizer = None
        self.seed_dirs: list[Path] = []
        self.meta: dict = {}
        self.load_error: Optional[str] = None

    # --------------------------------------------------------
    def _resolve_seed_dirs(self) -> list[Path]:
        """优先读 training_meta.json 的 seed_dirs，否则退化为 glob seed*。"""
        meta_path = self.model_dir / "training_meta.json"
        if meta_path.is_file():
            try:
                self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
                dirs = [Path(p) for p in self.meta.get("seed_dirs", []) if Path(p).is_dir()]
                if dirs:
                    return dirs
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取 training_meta.json 失败: %s", exc)
        return sorted(p for p in self.model_dir.glob("seed*") if p.is_dir())

    def load(self) -> bool:
        """加载全部种子模型。任一必需组件缺失即视为不可用。"""
        if self._models:
            return True
        self.seed_dirs = self._resolve_seed_dirs()
        if not self.seed_dirs:
            self.load_error = f"未找到权重目录（{self.model_dir}）"
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.seed_dirs[0]))
            for d in self.seed_dirs:
                m = AutoModelForSequenceClassification.from_pretrained(str(d))
                m = m.to(self._device).eval()
                self._models.append(m)
            return True
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("效价 RoBERTa 加载失败: %s", self.load_error)
            self._models = []
            return False

    # --------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self._models)

    @property
    def n_seeds(self) -> int:
        return len(self._models)

    @property
    def group_cv_macro_f1(self) -> Optional[float]:
        """训练时按被试分组 5 折 CV 的 macro-F1（写进 evidence 便于追溯）。"""
        v = self.meta.get("group_cv_macro_f1")
        return float(v) if v is not None else None

    # --------------------------------------------------------
    def predict_valence(self, text: str) -> np.ndarray:
        """返回 3 维效价概率 [negative, neutral, positive]（多种子平均）。"""
        if not self.available:
            raise RuntimeError("模型未加载")
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            enc = self._tokenizer(
                text, max_length=self.max_len, truncation=True,
                padding=True, return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            probs = None
            for m in self._models:
                p = F.softmax(m(**enc).logits, dim=-1)[0].cpu().numpy()
                probs = p if probs is None else probs + p
        return probs / len(self._models)


# ============================================================
# 单例入口
# ============================================================
@functools.lru_cache(maxsize=1)
def get_valence_model() -> ValenceRoBERTa:
    """进程内单例；加载失败时返回不可用实例（不抛异常，交由上层降级）。"""
    model = ValenceRoBERTa()
    model.load()
    if model.available:
        logger.info(
            "效价 RoBERTa 就绪：%d 个种子，分组CV macro-F1=%s",
            model.n_seeds, model.group_cv_macro_f1,
        )
    else:
        logger.info("效价 RoBERTa 不可用（%s），文本情绪将走词典降级", model.load_error)
    return model
