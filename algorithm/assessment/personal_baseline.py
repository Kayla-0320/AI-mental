"""
后端个人基线存储与偏离计算

与前端 usePersonalBaseline.ts 对齐：
  - EMA 算法：alpha=0.05，公式 mean_new = alpha * x + (1-alpha) * mean_old
  - 6 个生理/行为模态 + 5 维情绪基线
  - Z-score 偏离计算

核心接口：
  - sync_baseline(user_id, baseline_data) -> PersonalBaseline
  - compute_deviation(user_id, current_features) -> BaselineDeviation
  - get_baseline(user_id) -> Optional[PersonalBaseline]

存储：内存 dict + JSON 文件持久化（assessment/outputs/baselines.json）
"""
from __future__ import annotations

import functools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ============================================================
# 配置加载
# ============================================================

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_CONFIG_PATH = _CONFIG_DIR / "baseline.yaml"
_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def load_baseline_config() -> dict:
    """加载基线配置

    Returns:
        配置字典
    """
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return _get_default_config()


def _get_default_config() -> dict:
    """默认基线配置"""
    return {
        "ema": {"alpha": 0.05},
        "calibration": {"min_samples": 20},
        "deviation": {
            "significant_threshold": 2.0,
            "low_threshold": 0.5,
        },
        "storage": {"baselines_file": "outputs/baselines.json"},
        "modality_mapping": {
            "heartRate": "heartRate",
            "breathingRate": "breathingRate",
            "typingSpeed": "typingSpeed",
            "blinkRate": "blinkRate",
            "voiceF0": "voiceF0",
            "speechRate": "speechRate",
        },
    }


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BaselineMetric:
    """单模态基线指标

    Attributes:
        mean: EMA 均值
        std: EMA 标准差（基于 EMA 方差）
        n: 样本数
    """
    mean: float = 0.0
    std: float = 0.0
    n: int = 0


@dataclass
class PersonalBaseline:
    """个人基线数据

    Attributes:
        user_id: 用户唯一标识
        metrics: 各模态基线指标 {modality_name: BaselineMetric}
        emotion_baseline: 5维情绪 EMA 均值 [快乐, 悲伤, 焦虑, 愤怒, 中性]
        created_at: 创建时间戳
        last_updated: 最后更新时间戳
        sample_count: 总样本数
    """
    user_id: str
    metrics: dict[str, BaselineMetric] = field(default_factory=dict)
    emotion_baseline: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.2])
    created_at: float = 0.0
    last_updated: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "user_id": self.user_id,
            "metrics": {
                k: {"mean": v.mean, "std": v.std, "n": v.n}
                for k, v in self.metrics.items()
            },
            "emotion_baseline": self.emotion_baseline,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PersonalBaseline:
        """从字典反序列化

        支持两种 metrics 格式：
        1. BaselineMetric 字典：{"phq9": {"mean": 8, "std": 1, "n": 5}}
        2. 纯数值：{"phq9": 8} —— 自动转为 mean=value, std=0, n=1
        """
        metrics = {}
        for k, v in data.get("metrics", {}).items():
            if isinstance(v, dict):
                metrics[k] = BaselineMetric(
                    mean=v.get("mean", 0.0),
                    std=v.get("std", 0.0),
                    n=v.get("n", 0),
                )
            elif isinstance(v, (int, float)):
                # 纯数值：视为单次观测值，mean=value, std=0, n=1
                metrics[k] = BaselineMetric(mean=float(v), std=0.0, n=1)
            else:
                # 其他类型跳过
                continue
        return cls(
            user_id=data.get("user_id", ""),
            metrics=metrics,
            emotion_baseline=data.get("emotion_baseline", [0.2, 0.2, 0.2, 0.2, 0.2]),
            created_at=data.get("created_at", 0.0),
            last_updated=data.get("last_updated", 0.0),
            sample_count=data.get("sample_count", 0),
        )


@dataclass
class BaselineDeviation:
    """基线偏离结果

    Attributes:
        user_id: 用户唯一标识
        computed_at: 计算时间戳
        modality_z_scores: 各模态 Z-score {modality_name: z_score}
        emotion_z_scores: 5维情绪 Z-score
        significant_deviations: |z| > threshold 的维度名列表
        confidence: 基于 sample_count 的置信度 [0, 1]
        calibrated: 基线是否已校准
    """
    user_id: str
    computed_at: float = 0.0
    modality_z_scores: dict[str, float] = field(default_factory=dict)
    emotion_z_scores: list[float] = field(default_factory=list)
    significant_deviations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    calibrated: bool = False

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "user_id": self.user_id,
            "computed_at": self.computed_at,
            "modality_z_scores": self.modality_z_scores,
            "emotion_z_scores": self.emotion_z_scores,
            "significant_deviations": self.significant_deviations,
            "confidence": round(self.confidence, 4),
            "calibrated": self.calibrated,
        }


# ============================================================
# EMA 算法（与前端 usePersonalBaseline.ts 一致）
# ============================================================

def _update_metric_ema(metric: BaselineMetric, value: float, alpha: float) -> BaselineMetric:
    """EMA 更新单个模态的基线指标

    与前端 updateMetric() 算法一致：
      - 第一个样本：直接初始化
      - 后续样本：EMA 均值 + EMA 方差更新

    Args:
        metric: 当前基线指标
        value: 新观测值
        alpha: EMA 平滑系数

    Returns:
        更新后的基线指标
    """
    if metric.n == 0:
        return BaselineMetric(mean=value, std=0.0, n=1)

    mean = metric.mean
    variance = metric.std ** 2

    # EMA 均值更新
    new_mean = alpha * value + (1 - alpha) * mean
    # EMA 方差更新：alpha * (x - mean_old)^2 + (1 - alpha) * var_old
    new_variance = alpha * (value - mean) ** 2 + (1 - alpha) * variance
    new_std = math.sqrt(max(0.0, new_variance))

    return BaselineMetric(
        mean=round(new_mean, 4),
        std=round(new_std, 4),
        n=metric.n + 1,
    )


def _update_emotion_ema(
    baseline: list[float], probs: list[float], alpha: float
) -> list[float]:
    """EMA 更新情绪基线

    Args:
        baseline: 当前 5 维情绪基线
        probs: 新观测的 5 维情绪概率
        alpha: EMA 平滑系数

    Returns:
        更新后的 5 维情绪基线
    """
    if len(baseline) != 5 or len(probs) != 5:
        return baseline
    return [
        round(alpha * p + (1 - alpha) * b, 4)
        for p, b in zip(probs, baseline)
    ]


def _calc_z_score(metric: BaselineMetric, value: float) -> float:
    """计算 Z-score

    Args:
        metric: 基线指标
        value: 当前值

    Returns:
        Z-score，样本不足时返回 0
    """
    if metric.n < 2 or metric.std < 0.001:
        return 0.0
    return (value - metric.mean) / metric.std


def _welford_update(metric: BaselineMetric, value: float) -> BaselineMetric:
    """Welford 在线算法更新均值和标准差

    避免存储所有历史值，用 O(1) 空间累积统计量。
    数值稳定性优于朴素算法（避免大数相减精度损失）。

    算法：
        n += 1
        delta = value - mean
        mean += delta / n
        delta2 = value - mean_new
        M2 += delta * delta2
        variance = M2 / n
        std = sqrt(variance)

    Args:
        metric: 当前基线指标（含 mean, std, n）
        value: 新观测值

    Returns:
        更新后的基线指标
    """
    if metric.n == 0:
        return BaselineMetric(mean=float(value), std=0.0, n=1)

    n = metric.n
    mean = metric.mean
    # 从 std 反推 M2（ sum of squared deviations ）
    # variance = std^2 = M2 / n  =>  M2 = std^2 * n
    m2 = (metric.std ** 2) * n

    # Welford 更新
    n_new = n + 1
    delta = value - mean
    mean_new = mean + delta / n_new
    delta2 = value - mean_new
    m2_new = m2 + delta * delta2

    variance_new = m2_new / n_new
    std_new = math.sqrt(max(0.0, variance_new))

    return BaselineMetric(
        mean=round(mean_new, 4),
        std=round(std_new, 4),
        n=n_new,
    )


# ============================================================
# 存储层（内存 + JSON 文件持久化）
# ============================================================

class BaselineStore:
    """基线数据存储

    内存 dict 缓存 + JSON 文件持久化。
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        config = load_baseline_config()
        if storage_path is None:
            rel_path = config.get("storage", {}).get("baselines_file", "outputs/baselines.json")
            storage_path = _OUTPUT_DIR / rel_path
        self._path = storage_path
        self._data: dict[str, PersonalBaseline] = {}
        self._load_from_file()

    def _load_from_file(self) -> None:
        """从 JSON 文件加载"""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for user_id, user_data in raw.items():
                self._data[user_id] = PersonalBaseline.from_dict(user_data)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def _save_to_file(self) -> None:
        """保存到 JSON 文件"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = {uid: bl.to_dict() for uid, bl in self._data.items()}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    def get(self, user_id: str) -> Optional[PersonalBaseline]:
        """获取用户基线"""
        return self._data.get(user_id)

    def put(self, baseline: PersonalBaseline) -> None:
        """存储/更新用户基线并持久化"""
        self._data[baseline.user_id] = baseline
        self._save_to_file()

    def delete(self, user_id: str) -> bool:
        """删除用户基线"""
        if user_id in self._data:
            del self._data[user_id]
            self._save_to_file()
            return True
        return False

    def list_users(self) -> list[str]:
        """列出所有有基线的用户 ID"""
        return list(self._data.keys())


# 全局存储实例（延迟初始化）
_store: Optional[BaselineStore] = None


def _get_store() -> BaselineStore:
    """获取全局存储实例"""
    global _store
    if _store is None:
        _store = BaselineStore()
    return _store


def reset_store() -> None:
    """重置全局存储（测试用）"""
    global _store
    _store = None


# ============================================================
# 核心接口
# ============================================================

def sync_baseline(user_id: str, baseline_data: dict) -> PersonalBaseline:
    """同步前端基线数据到后端

    接收前端 POST 的基线数据，合并到已有基线中。
    支持两种模式：
      1. 完整基线同步：前端发送 PersonalBaseline 结构（含 metrics + emotion_baseline）
      2. 增量更新：前端发送实时观测值，后端用 EMA 更新

    Args:
        user_id: 用户唯一标识
        baseline_data: 基线数据，可以是：
            - 完整结构：{"metrics": {...}, "emotion_baseline": [...], "sample_count": N}
            - 增量观测：{"heartRate": 72, "breathingRate": 16, ...}

    Returns:
        PersonalBaseline: 更新后的基线
    """
    config = load_baseline_config()
    alpha = config.get("ema", {}).get("alpha", 0.05)
    store = _get_store()

    existing = store.get(user_id)
    now = time.time()

    # 判断是完整同步还是增量更新
    if "metrics" in baseline_data:
        # 完整同步模式：Welford 累积更新（而非覆盖）
        new_data = {**baseline_data, "user_id": user_id}
        new_baseline = PersonalBaseline.from_dict(new_data)

        if existing is not None:
            # 有已有基线：用 Welford 算法累积每个指标
            updated_metrics = dict(existing.metrics)
            for k, new_metric in new_baseline.metrics.items():
                old_metric = updated_metrics.get(k, BaselineMetric())
                # 将新值视为一次观测，用 Welford 累积
                updated_metrics[k] = _welford_update(old_metric, new_metric.mean)

            existing.metrics = updated_metrics
            existing.sample_count += 1
            existing.last_updated = now

            # 累积情绪基线
            if new_baseline.emotion_baseline:
                existing.emotion_baseline = _update_emotion_ema(
                    existing.emotion_baseline,
                    new_baseline.emotion_baseline,
                    alpha,
                )

            store.put(existing)
            return existing
        else:
            # 无已有基线：直接初始化
            new_baseline.created_at = now
            new_baseline.last_updated = now
            store.put(new_baseline)
            return new_baseline

    # 增量更新模式：用 EMA 更新各模态
    if existing is None:
        existing = PersonalBaseline(
            user_id=user_id,
            created_at=now,
            last_updated=now,
        )

    modality_mapping = config.get("modality_mapping", {})
    updated_metrics = dict(existing.metrics)

    for frontend_key, value in baseline_data.items():
        if frontend_key == "emotionProbs":
            # 情绪基线更新
            if isinstance(value, list) and len(value) == 5:
                existing.emotion_baseline = _update_emotion_ema(
                    existing.emotion_baseline, value, alpha
                )
            continue

        # 映射模态名
        modality_name = modality_mapping.get(frontend_key, frontend_key)

        if not isinstance(value, (int, float)) or value <= 0:
            continue

        current_metric = updated_metrics.get(modality_name, BaselineMetric())
        updated_metrics[modality_name] = _update_metric_ema(current_metric, value, alpha)

    existing.metrics = updated_metrics
    existing.sample_count += 1
    existing.last_updated = now

    store.put(existing)
    return existing


def compute_deviation(
    user_id: str,
    current_features: dict,
) -> BaselineDeviation:
    """计算当前特征相对个人基线的 Z-score 偏离

    Args:
        user_id: 用户唯一标识
        current_features: 当前各模态观测值
            - 生理模态：{"heartRate": 72, "breathingRate": 16, ...}
            - 情绪：{"emotionProbs": [0.1, 0.3, 0.4, 0.1, 0.1]}

    Returns:
        BaselineDeviation: 偏离计算结果
    """
    config = load_baseline_config()
    sig_threshold = config.get("deviation", {}).get("significant_threshold", 2.0)
    min_samples = config.get("calibration", {}).get("min_samples", 20)
    modality_mapping = config.get("modality_mapping", {})

    store = _get_store()
    baseline = store.get(user_id)

    deviation = BaselineDeviation(
        user_id=user_id,
        computed_at=time.time(),
    )

    if baseline is None:
        return deviation

    deviation.calibrated = baseline.sample_count >= min_samples
    # 置信度基于样本数，min_samples 时达到 1.0
    deviation.confidence = min(1.0, baseline.sample_count / min_samples)

    significant = []

    # 计算各模态 Z-score
    for frontend_key, value in current_features.items():
        if frontend_key == "emotionProbs":
            # 情绪偏离
            if isinstance(value, list) and len(value) == 5:
                emotion_z = []
                for i, (p, b) in enumerate(zip(value, baseline.emotion_baseline)):
                    # 情绪基线的 std 用经验值估计
                    emotion_std = 0.15  # 情绪概率的经验标准差
                    if emotion_std > 0.001:
                        z = (p - b) / emotion_std
                    else:
                        z = 0.0
                    z = round(z, 4)
                    emotion_z.append(z)
                    if abs(z) > sig_threshold:
                        significant.append(f"emotion_dim_{i}")
                deviation.emotion_z_scores = emotion_z
            continue

        # 映射模态名
        modality_name = modality_mapping.get(frontend_key, frontend_key)

        if not isinstance(value, (int, float)):
            continue

        metric = baseline.metrics.get(modality_name)
        if metric is None:
            continue

        z = _calc_z_score(metric, value)
        deviation.modality_z_scores[modality_name] = round(z, 4)

        if abs(z) > sig_threshold:
            significant.append(modality_name)

    deviation.significant_deviations = significant
    return deviation


def get_baseline(user_id: str) -> Optional[PersonalBaseline]:
    """获取用户个人基线

    Args:
        user_id: 用户唯一标识

    Returns:
        PersonalBaseline 或 None（无基线数据时）
    """
    store = _get_store()
    return store.get(user_id)
