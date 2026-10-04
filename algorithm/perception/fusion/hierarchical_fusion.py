"""
HUAF-TC: Hierarchical Uncertainty-Aware Fusion with Temporal Consistency
层级不确定性感知的时序一致性融合算法

================================================================
原创性声明
================================================================
本算法为面向青少年心理健康多模态感知系统的原创设计。
不同于传统的加权平均、Stacking 或动态加权融合方法，
HUAF-TC 提出了四个核心创新机制：

1. 三层递进融合架构 (Three-Layer Progressive Fusion)
   模拟人类情绪处理的层级认知模型：
   - L1 基础感知层（文本/语音/面部）→ 直接情绪信号
   - L2 认知解释层（昼夜节律/认知扭曲/行为模式）→ 认知评估
   - L3 生理背景层（HRV/呼吸/行为激活/眼动/语音语义）→ 生理基线
   各层独立融合后再通过门控机制组合，而非将所有模态扁平混合。

2. 熵驱动不确定性量化 (Entropy-Driven Uncertainty Quantification)
   每个模态的权重不由固定配置决定，而是由其输出的 Shannon 熵
   动态计算。高熵（接近均匀分布）= 低信息量 = 低权重；
   低熵（尖锐分布）= 高信息量 = 高权重。

3. 年龄自适应门控 (Age-Adaptive Gating)
   基于发展心理学研究，不同年龄段的三层信任度不同：
   - 初中生：L1 权重高（情绪表达直接，面部/语音信号可靠）
   - 高中生：L2 权重提升（认知能力发展，认知扭曲信号增强）
   - 大学生：L2 权重最高（掩饰能力强，需要深层认知分析）
   门控还会根据各层的信号偏离度动态调整权重（信号驱动修正）。

4. 时序一致性平滑 (Temporal Consistency Smoothing)
   情绪状态具有内在连续性，不会瞬间跳变。
   使用自适应 EMA（指数移动平均）平滑跨时间窗口的预测，
   平滑因子由当前预测波动性动态决定。

================================================================
理论依据
================================================================
- 层级处理：Cognitive Appraisal Theory (Lazarus, 1991)
  情绪产生经历"初级评估→次级评估"的层级过程
- 不确定性建模：Dempster-Shafer Evidence Theory (Shafer, 1976)
  信息熵度量证据的"不确定度"，用于证据权重分配
- 年龄差异：Erikson's Psychosocial Development (1968)
  不同发展阶段的认知能力和情绪表达方式存在质差异
- 时序一致性：Kalman Filtering 思想 (Kalman, 1960)
  当前状态估计应结合历史先验与当前观测

================================================================
使用方式
================================================================
    from perception.fusion.hierarchical_fusion import (
        HierarchicalFusionEngine,
        ModalityResult,
    )

    engine = HierarchicalFusionEngine()
    result = engine.fuse(modality_results, age_group="early_adolescent")
    # result.fused_probs → 融合后的 5 维情绪概率
    # result.confidence  → 融合置信度
    # result.evidence    → 融合证据链
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ModalityResult:
    """单个模态的融合输入

    Attributes:
        name: 模态唯一标识符
        probs: 5 维情绪概率分布 [快乐, 悲伤, 焦虑, 愤怒, 中性]
            与 EmotionResult.text_emotion_probs 格式一致。
            若模态仅输出 risk_score 而无概率分布，应构造近似分布。
        risk_score: 标量风险分 [0, 1]，可与 probs 互为补充。
        confidence: 模态自评置信度 [0, 1]。
        evidence: 该模态产生的证据字符串列表。
    """
    name: str
    probs: list[float] = field(default_factory=lambda: [0.2] * 5)
    risk_score: float = 0.0
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)


@dataclass
class HUAFConfig:
    """HUAF-TC 算法超参数

    Attributes:
        entropy_weight: 熵不确定度在综合不确定度中的占比。
        confidence_weight: 置信度不确定度在综合不确定度中的占比。
        temporal_window: 时序平滑保留的历史帧数。
        layer_priors: 三层先验权重 [L1, L2, L3]，
            按年龄段预设，可在运行时被信号驱动修正。
        signal_deviation_sensitivity: 信号驱动修正的灵敏度，
            控制某层信号偏离均值时对门控权重的修正幅度。
        min_modality_confidence: 模态最低置信度阈值，
            低于此值的模态在层内融合时被过滤。
    """
    entropy_weight: float = 0.6
    confidence_weight: float = 0.4
    temporal_window: int = 5
    layer_priors: list[float] = field(default_factory=lambda: [0.40, 0.35, 0.25])
    signal_deviation_sensitivity: float = 0.15
    min_modality_confidence: float = 0.2


# ============================================================
# 年龄自适应先验配置
# ============================================================
# 基于发展心理学研究 (Erikson, 1968; Casey et al., 2008)：
# - 初中生情绪表达直接 → L1 感知层最可信
# - 高中生抽象思维发展 → L2 认知层信号增强
# - 大学生掩饰能力强 → L2 认知层最重要，L1 感知层可能被误导

AGE_PRIORS: dict[str, list[float]] = {
    #                       L1感知   L2认知   L3生理
    "early_adolescent":  [0.50,    0.28,    0.22],   # 初中：信感知
    "mid_adolescent":    [0.38,    0.38,    0.24],   # 高中：感知+认知均衡
    "young_adult":       [0.28,    0.45,    0.27],   # 大学：信认知
}

# 三层包含的模态名称集合
LAYER_MODALITIES: dict[str, list[str]] = {
    "L1_perception":  ["text", "voice", "face"],
    "L2_cognitive":   ["circadian", "cognitive", "behavior"],
    "L3_physiological": ["hrv", "breathing", "behavioral_act", "eye", "voice_semantics"],
}


# ============================================================
# 融合结果
# ============================================================

@dataclass
class LayerResult:
    """单层融合结果"""
    name: str
    fused_probs: list[float]
    uncertainty: float
    modality_count: int
    modality_weights: dict[str, float]


@dataclass
class HierarchicalFusionResult:
    """HUAF-TC 完整融合结果

    Attributes:
        fused_probs: 融合后的 5 维情绪概率 [快乐, 悲伤, 焦虑, 愤怒, 中性]
        confidence: 综合置信度 [0, 1]
        uncertainty: 综合不确定度 [0, 1]
        layer_results: 三层各自的融合结果
        gating_weights: 门控机制输出的三层权重 [w_L1, w_L2, w_L3]
        temporal_alpha: 当前时序平滑因子（大=信任历史，小=信任当前）
        evidence: 完整证据链
        timestamp: 融合完成时间戳
    """
    fused_probs: list[float]
    confidence: float
    uncertainty: float
    layer_results: list[LayerResult]
    gating_weights: list[float]
    temporal_alpha: float
    evidence: list[str]
    timestamp: float


# ============================================================
# 核心算法引擎
# ============================================================

class HierarchicalFusionEngine:
    """HUAF-TC 融合引擎

    实现完整的四阶段融合流水线：
      阶段 1 → 熵驱动不确定性量化
      阶段 2 → 层内不确定性加权融合
      阶段 3 → 年龄自适应门控
      阶段 4 → 时序一致性平滑
    """

    def __init__(self, config: Optional[HUAFConfig] = None) -> None:
        self._config = config or HUAFConfig()
        self._history: list[np.ndarray] = []

    # ----------------------------------------------------------
    # 阶段 1：熵驱动不确定性量化
    # ----------------------------------------------------------

    @staticmethod
    def _entropy(probs: np.ndarray) -> float:
        """归一化 Shannon 熵  H ∈ [0, 1]

        H = -Σ p·log₂(p) / log₂(K)
        K=5 时，log₂(5) ≈ 2.322
        """
        p = np.clip(probs, 1e-10, 1.0)
        h = -np.sum(p * np.log2(p))
        return float(h / np.log2(len(probs)))

    def compute_uncertainty(self, mod: ModalityResult) -> float:
        """综合不确定度 = 熵不确定度 × w_ent + (1-置信度) × w_conf"""
        cfg = self._config
        ent = self._entropy(np.array(mod.probs))
        conf_unc = 1.0 - mod.confidence
        return float(cfg.entropy_weight * ent + cfg.confidence_weight * conf_unc)

    # ----------------------------------------------------------
    # 阶段 2：层内不确定性加权融合
    # ----------------------------------------------------------

    def _fuse_layer(
        self,
        mods: list[ModalityResult],
        layer_name: str,
    ) -> LayerResult:
        """对一层内的模态做不确定性加权融合

        权重公式：w_i = (1 - u_i) / Σ(1 - u_j)
        即不确定度越低的模态获得越高权重。
        """
        n = len(mods)
        if n == 0:
            return LayerResult(
                name=layer_name,
                fused_probs=[0.2] * 5,
                uncertainty=1.0,
                modality_count=0,
                modality_weights={},
            )

        if n == 1:
            u = self.compute_uncertainty(mods[0])
            return LayerResult(
                name=layer_name,
                fused_probs=list(mods[0].probs),
                uncertainty=u,
                modality_count=1,
                modality_weights={mods[0].name: 1.0},
            )

        uncertainties = [self.compute_uncertainty(m) for m in mods]
        reliabilities = [max(1.0 - u, 0.01) for u in uncertainties]
        total_r = sum(reliabilities)
        weights = [r / total_r for r in reliabilities]

        fused = np.zeros(5)
        for w, m in zip(weights, mods):
            fused += w * np.array(m.probs)
        s = fused.sum()
        if s > 0:
            fused /= s

        layer_unc = sum(w * u for w, u in zip(weights, uncertainties))

        return LayerResult(
            name=layer_name,
            fused_probs=fused.tolist(),
            uncertainty=layer_unc,
            modality_count=n,
            modality_weights={m.name: round(w, 4) for m, w in zip(mods, weights)},
        )

    # ----------------------------------------------------------
    # 阶段 3：年龄自适应门控
    # ----------------------------------------------------------

    def _gate(
        self,
        layers: list[LayerResult],
        age_group: Optional[str],
    ) -> list[float]:
        """年龄自适应门控：先验 + 信号驱动修正

        Step 1: 取年龄先验 gate_prior = AGE_PRIORS[age_group]
        Step 2: 计算各层"信号强度" = 1 - layer.uncertainty
        Step 3: 若某层信号显著偏离均值，增加/减少其门控权重
        Step 4: softmax 归一化输出最终门控权重
        """
        cfg = self._config
        prior = list(AGE_PRIORS.get(age_group, cfg.layer_priors))

        signals = [max(1.0 - lr.uncertainty, 0.01) for lr in layers]
        mean_sig = sum(signals) / max(len(signals), 1)

        adjusted = []
        for g, s in zip(prior, signals):
            deviation = (s - mean_sig) / max(mean_sig, 0.01)
            correction = cfg.signal_deviation_sensitivity * deviation
            adjusted.append(max(g + correction, 0.05))

        total = sum(adjusted)
        return [a / total for a in adjusted]

    # ----------------------------------------------------------
    # 阶段 4：时序一致性平滑
    # ----------------------------------------------------------

    def _temporal_smooth(self, current: np.ndarray) -> tuple[np.ndarray, float]:
        """自适应 EMA 平滑

        alpha = clip(0.5 + (0.5 - volatility), 0.3, 0.9)
        volatility = mean(|current - history_mean|)
        alpha 大 → 更信任历史（情绪稳定期）
        alpha 小 → 更信任当前（情绪突变期）
        """
        self._history.append(current.copy())
        window = self._config.temporal_window
        if len(self._history) > window:
            self._history = self._history[-window:]

        if len(self._history) <= 1:
            return current.copy(), 0.5

        hist_mean = np.mean(self._history[:-1], axis=0)
        volatility = float(np.mean(np.abs(current - hist_mean)))
        alpha = max(0.3, min(0.9, 0.5 + (0.5 - volatility)))

        smoothed = alpha * hist_mean + (1 - alpha) * current
        s = smoothed.sum()
        if s > 0:
            smoothed /= s
        return smoothed, alpha

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def fuse(
        self,
        modality_results: list[ModalityResult],
        age_group: Optional[str] = None,
    ) -> HierarchicalFusionResult:
        """执行完整的 HUAF-TC 四阶段融合

        Args:
            modality_results: 各模态的分析结果列表。
            age_group: 年龄段标识
                ("early_adolescent" / "mid_adolescent" / "young_adult")。

        Returns:
            HierarchicalFusionResult: 包含融合概率、置信度、证据链等。
        """
        if not modality_results:
            return HierarchicalFusionResult(
                fused_probs=[0.2] * 5,
                confidence=0.0,
                uncertainty=1.0,
                layer_results=[],
                gating_weights=[0.34, 0.33, 0.33],
                temporal_alpha=0.5,
                evidence=["无可用模态数据"],
                timestamp=time.time(),
            )

        # 构建模态名称 → 结果映射
        mod_map: dict[str, ModalityResult] = {
            m.name: m for m in modality_results
        }

        # ── 阶段 2：层内融合 ──
        layer_results: list[LayerResult] = []
        for layer_name, mod_names in LAYER_MODALITIES.items():
            layer_mods = [mod_map[n] for n in mod_names if n in mod_map]
            lr = self._fuse_layer(layer_mods, layer_name)
            layer_results.append(lr)

        # ── 阶段 3：年龄自适应门控 ──
        gating_weights = self._gate(layer_results, age_group)

        # 门控加权组合
        fused = np.zeros(5)
        for w, lr in zip(gating_weights, layer_results):
            fused += w * np.array(lr.fused_probs)
        s = fused.sum()
        if s > 0:
            fused /= s

        # ── 阶段 4：时序一致性平滑 ──
        smoothed, alpha = self._temporal_smooth(fused)

        # ── 计算综合指标 ──
        mean_unc = float(
            sum(w * lr.uncertainty for w, lr in zip(gating_weights, layer_results))
        )
        active_mods = sum(lr.modality_count for lr in layer_results)
        total_possible = sum(len(v) for v in LAYER_MODALITIES.values())
        coverage = active_mods / total_possible

        layer_probs = [np.array(lr.fused_probs) for lr in layer_results if lr.modality_count > 0]
        if len(layer_probs) >= 2:
            pairwise_agree = []
            for i in range(len(layer_probs)):
                for j in range(i + 1, len(layer_probs)):
                    pairwise_agree.append(
                        1.0 - float(np.mean(np.abs(layer_probs[i] - layer_probs[j])))
                    )
            agreement = sum(pairwise_agree) / len(pairwise_agree)
        else:
            agreement = 0.5

        confidence = float(
            0.30 * coverage
            + 0.30 * (1.0 - mean_unc)
            + 0.20 * agreement
            + 0.20 * (1.0 - abs(alpha - 0.5) * 2)
        )
        confidence = max(0.0, min(1.0, confidence))

        # ── 构建证据链 ──
        evidence: list[str] = []
        layer_labels = {
            "L1_perception": "基础感知层",
            "L2_cognitive": "认知解释层",
            "L3_physiological": "生理背景层",
        }
        for lr, gw in zip(layer_results, gating_weights):
            if lr.modality_count > 0:
                label = layer_labels.get(lr.name, lr.name)
                mods_str = ", ".join(lr.modality_weights.keys())
                evidence.append(
                    f"{label}: {lr.modality_count}个模态({mods_str}), "
                    f"不确定度={lr.uncertainty:.3f}, 门控权重={gw:.3f}"
                )
        if age_group:
            evidence.append(f"年龄自适应门控: age_group={age_group}")
        evidence.append(f"时序平滑因子: α={alpha:.3f}")
        evidence.append(f"融合策略: HUAF-TC (Hierarchical Uncertainty-Aware)")

        return HierarchicalFusionResult(
            fused_probs=smoothed.tolist(),
            confidence=round(confidence, 4),
            uncertainty=round(mean_unc, 4),
            layer_results=layer_results,
            gating_weights=[round(w, 4) for w in gating_weights],
            temporal_alpha=round(alpha, 4),
            evidence=evidence,
            timestamp=time.time(),
        )


# ============================================================
# 便捷函数
# ============================================================

def risk_score_to_probs(risk: float) -> list[float]:
    """将标量 risk_score 近似转换为 5 维情绪概率分布

    风险分主要映射到焦虑(dim=2)和悲伤(dim=1)维度，
    其余维度按风险程度递减分配。

    Args:
        risk: 风险分 [0, 1]

    Returns:
        [快乐, 悲伤, 焦虑, 愤怒, 中性] 概率分布（归一化）
    """
    anxiety = risk * 0.45
    sadness = risk * 0.30
    anger = risk * 0.10
    neutral = max(0.05, (1.0 - risk) * 0.6)
    happy = max(0.02, 1.0 - anxiety - sadness - anger - neutral)
    probs = [happy, sadness, anxiety, anger, neutral]
    s = sum(probs)
    return [p / s for p in probs]


def hierarchical_fuse(
    modality_results: list[ModalityResult],
    age_group: Optional[str] = None,
) -> HierarchicalFusionResult:
    """HUAF-TC 融合便捷入口

    Args:
        modality_results: 各模态结果列表
        age_group: 年龄段

    Returns:
        HierarchicalFusionResult
    """
    engine = HierarchicalFusionEngine()
    return engine.fuse(modality_results, age_group)
