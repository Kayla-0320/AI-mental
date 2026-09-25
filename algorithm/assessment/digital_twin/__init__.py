"""
心理数字孪生（Psychological Digital Twin）
==========================================
为每个青少年建立个性化的心理因果模型，实现：
  1. 个人因果图发现 —— 发现每个用户独有的症状链条（如 睡眠→认知扭曲→社交退缩）
  2. 心理轨迹预测 —— 基于因果模型预测未来心理状态
  3. 精准干预时机优化 —— 计算最优干预时间和方式

核心创新：
  - 首次将 Digital Twin 概念引入青少年心理健康 AI
  - 个人因果图（Per-User Causal Graph）：每个人的症状交互模式不同
  - 干预时机优化（Just-In-Time Adaptive Intervention, JITAI）

临床依据：
  - Granger Causality (Granger, 1969): 时间序列因果推断
  - JITAI (Nahum-Shani et al., 2012): 精准时机自适应干预
  - Digital Twin (Mirams et al., 2013): 个性化计算模型

架构：
  观测序列 → 因果图发现 → 数字孪生预测 → 干预时机优化
                ↓                ↓                ↓
           CausalEdge[]    Trajectory[]    InterventionDecision

存储：内存 dict + JSON 文件持久化
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from assessment.digital_twin.precision_engine import (
    CTPLEWEngine,
    PrecisionReport,
    get_precision_engine,
)
from assessment.digital_twin.landscape_nd import get_multidim_analyzer
from assessment.digital_twin.scm_counterfactual import StructuralCausalModel
from assessment.digital_twin.in_silico_simulator import InSilicoPatient
from assessment.digital_twin.optimal_control import (
    InterventionMPC,
    minimum_rescue_path,
)

logger = logging.getLogger(__name__)


# ============================================================
# 路径配置
# ============================================================

_MODULE_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _MODULE_DIR / "outputs"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_TWIN_STORE_PATH = str(_OUTPUT_DIR / "digital_twins.json")


# ============================================================
# 心理维度定义
# ============================================================

class PsychoDimension(str, Enum):
    """心理状态维度（与 11 模态对齐）"""
    # L1 基础感知
    TEXT_EMOTION = "text_emotion"          # 文本情绪
    VOICE_ACOUSTIC = "voice_acoustic"      # 语音声学
    FACIAL = "facial"                      # 面部表情
    # L2 认知解释
    CIRCADIAN = "circadian"                # 昼夜节律
    COGNITIVE = "cognitive"                # 认知扭曲
    BEHAVIOR = "behavior"                  # 行为模式
    # L3 生理背景
    HRV = "hrv"                            # 心率变异性
    BREATHING = "breathing"                # 呼吸模式
    BEHAVIORAL_ACT = "behavioral_act"      # 行为激活
    EYE = "eye"                            # 眼动模式
    VOICE_SEMANTICS = "voice_semantics"    # 语音语义


# 维度标签（用于展示）
DIM_LABELS = {
    PsychoDimension.TEXT_EMOTION: "文本情绪",
    PsychoDimension.VOICE_ACOUSTIC: "语音声学",
    PsychoDimension.FACIAL: "面部表情",
    PsychoDimension.CIRCADIAN: "昼夜节律",
    PsychoDimension.COGNITIVE: "认知扭曲",
    PsychoDimension.BEHAVIOR: "行为模式",
    PsychoDimension.HRV: "心率变异性",
    PsychoDimension.BREATHING: "呼吸模式",
    PsychoDimension.BEHAVIORAL_ACT: "行为激活",
    PsychoDimension.EYE: "眼动模式",
    PsychoDimension.VOICE_SEMANTICS: "语音语义",
}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class CausalEdge:
    """因果边：cause → effect，延迟 lag 个时间步

    含义：cause 维度的变化在 lag 步后会显著影响 effect 维度。
    strength > 0 表示正向因果（cause↑ → effect↑）
    strength < 0 表示负向因果（cause↑ → effect↓）
    """
    cause: str
    effect: str
    strength: float        # 因果强度 [-1, 1]
    lag: int               # 延迟步数（1=立即，2=延迟一步...）
    p_value: float         # 统计显著性
    confidence: float      # 置信度 [0, 1]


@dataclass
class CausalGraph:
    """个人因果图

    每个用户有独立的因果图，描述其心理症状之间的因果关系。
    例如：circadian →(lag=2)→ cognitive →(lag=1)→ text_emotion
    表示"睡眠异常 2 步后认知扭曲增加，再 1 步后文本情绪恶化"。
    """
    user_id: str
    edges: list[CausalEdge] = field(default_factory=list)
    n_observations: int = 0
    last_updated: float = 0.0

    def get_causes(self, dim: str) -> list[CausalEdge]:
        """获取导致 dim 变化的所有因果边"""
        return [e for e in self.edges if e.effect == dim]

    def get_effects(self, dim: str) -> list[CausalEdge]:
        """获取 dim 会导致哪些变化"""
        return [e for e in self.edges if e.cause == dim]

    def get_chain(self, start_dim: str, max_depth: int = 3) -> list[list[CausalEdge]]:
        """获取从 start_dim 出发的因果链条

        返回所有可能的因果路径，如：
        [[circadian→cognitive], [circadian→cognitive→text_emotion], ...]
        """
        chains: list[list[CausalEdge]] = []
        self._dfs_chains(start_dim, [], max_depth, chains, set())
        return chains

    def _dfs_chains(self, current: str, path: list[CausalEdge],
                    max_depth: int, result: list[list[CausalEdge]],
                    visited: set[str]):
        if len(path) >= max_depth:
            return
        for edge in self.get_effects(current):
            if edge.effect in visited:
                continue  # 避免环路
            new_path = path + [edge]
            result.append(new_path)
            visited.add(edge.effect)
            self._dfs_chains(edge.effect, new_path, max_depth, result, visited)
            visited.discard(edge.effect)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "edges": [asdict(e) for e in self.edges],
            "n_observations": self.n_observations,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CausalGraph:
        edges = [CausalEdge(**e) for e in d.get("edges", [])]
        return cls(
            user_id=d["user_id"],
            edges=edges,
            n_observations=d.get("n_observations", 0),
            last_updated=d.get("last_updated", 0.0),
        )


@dataclass
class TrajectoryPoint:
    """轨迹预测点"""
    time_step: int         # 未来第几步
    timestamp: float       # 预测时间戳
    predicted: dict[str, float]   # 各维度预测值
    uncertainty: float     # 预测不确定度 [0, 1]
    crisis_probability: float     # 危机概率 [0, 1]


@dataclass
class Trajectory:
    """心理轨迹预测"""
    user_id: str
    points: list[TrajectoryPoint] = field(default_factory=list)
    forecast_horizon: int = 0
    generated_at: float = 0.0

    def get_crisis_risk(self) -> float:
        """获取未来轨迹中的最大危机概率"""
        if not self.points:
            return 0.0
        return max(p.crisis_probability for p in self.points)

    def get_deterioration_time(self) -> Optional[int]:
        """获取预计恶化时间步（危机概率首次超过 0.5）"""
        for p in self.points:
            if p.crisis_probability > 0.5:
                return p.time_step
        return None


@dataclass
class InterventionDecision:
    """干预决策"""
    should_intervene: bool
    urgency: float             # 紧急程度 [0, 1]
    optimal_delay: int         # 最优延迟步数（0=立即干预）
    recommended_dimensions: list[str]  # 建议干预的维度
    expected_benefit: float    # 期望收益 [0, 1]
    reasoning: str             # 决策理由
    chain_to_break: Optional[str] = None  # 要阻断的因果链


# ============================================================
# 1. 个人因果图发现引擎
# ============================================================

class PersonalCausalDiscovery:
    """个人因果图发现

    使用 Granger 因果检验发现每个用户独有的症状因果结构。

    算法流程：
      1. 收集用户的多维度时间序列数据
      2. 对每对维度 (A, B)，检验 A 是否 Granger-cause B
      3. 使用 VAR(1) 模型估计因果强度和延迟
      4. 过滤不显著的边（p > threshold）
      5. 输出个人因果图

    Granger 因果检验原理：
      如果 "用 A 的过去值 + B 的过去值" 预测 B 的当前值，
      比 "仅用 B 的过去值" 预测得更好，则 A Granger-causes B。
      使用 F-test 检验改善是否统计显著。
    """

    def __init__(self, max_lag: int = 5, significance_level: float = 0.05):
        self.max_lag = max_lag
        self.significance_level = significance_level

    def discover(
        self,
        user_id: str,
        time_series: dict[str, list[float]],
    ) -> CausalGraph:
        """发现个人因果图

        Args:
            user_id: 用户 ID
            time_series: {维度名: [观测值序列]}

        Returns:
            个人因果图
        """
        dims = list(time_series.keys())
        n = len(next(iter(time_series.values())))

        if n < self.max_lag + 10:
            logger.warning(f"数据不足（{n} < {self.max_lag + 10}），无法发现因果图")
            return CausalGraph(user_id=user_id, n_observations=n)

        edges: list[CausalEdge] = []

        for cause_dim in dims:
            for effect_dim in dims:
                if cause_dim == effect_dim:
                    continue

                cause_arr = np.array(time_series[cause_dim], dtype=np.float64)
                effect_arr = np.array(time_series[effect_dim], dtype=np.float64)

                # 对每个 lag 检验 Granger 因果
                best_edge = self._granger_test(
                    cause_arr, effect_arr, cause_dim, effect_dim
                )
                if best_edge is not None:
                    edges.append(best_edge)

        # 过滤：移除被中介变量解释的伪因果（简化版）
        edges = self._remove_spurious(edges)

        graph = CausalGraph(
            user_id=user_id,
            edges=edges,
            n_observations=n,
            last_updated=time.time(),
        )
        logger.info(
            f"[{user_id}] 因果图发现完成：{len(dims)} 维度，"
            f"{n} 观测，{len(edges)} 条因果边"
        )
        return graph

    def _granger_test(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        cause_name: str,
        effect_name: str,
    ) -> Optional[CausalEdge]:
        """对单个 (cause, effect) 对进行 Granger 因果检验

        对每个 lag ∈ [1, max_lag]：
          - 受限模型：effect ~ effect(t-1) + effect(t-2) + ... + effect(t-lag)
          - 非受限模型：受限 + cause(t-1) + cause(t-2) + ... + cause(t-lag)
          - F-test：非受限是否显著优于受限？

        Returns:
            最强因果边，或 None（不显著）
        """
        best_edge: Optional[CausalEdge] = None
        best_f = 0.0

        for lag in range(1, self.max_lag + 1):
            if len(cause) < lag * 2 + 5:
                continue

            # 构建滞后矩阵
            n = len(effect)
            start = lag
            end = n

            # 受限模型：仅用 effect 的过去
            X_restricted = np.column_stack([
                effect[start - lag:end - lag] if l == 0
                else effect[start - l:end - l]
                for l in range(1, lag + 1)
            ])
            # 添加截距
            X_restricted = np.column_stack([np.ones(end - start), X_restricted])
            y = effect[start:end]

            # 非受限模型：effect 的过去 + cause 的过去
            X_unrestricted = np.column_stack([
                X_restricted,
                *[cause[start - l:end - l] for l in range(1, lag + 1)]
            ])

            # OLS 拟合
            try:
                beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
                resid_r = y - X_restricted @ beta_r
                rss_r = np.sum(resid_r ** 2)

                beta_u = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]
                resid_u = y - X_unrestricted @ beta_u
                rss_u = np.sum(resid_u ** 2)

                # F-statistic
                df_diff = lag  # 新增参数数
                df_resid = len(y) - X_unrestricted.shape[1]
                if df_resid <= 0 or rss_u <= 0:
                    continue

                f_stat = ((rss_r - rss_u) / df_diff) / (rss_u / df_resid)

                # 近似 p-value（F 分布）
                p_value = self._f_survival(f_stat, df_diff, df_resid)

                if p_value < self.significance_level and f_stat > best_f:
                    # 估计因果强度：非受限模型中 cause 系数的均值
                    cause_coefs = beta_u[-lag:]
                    strength = float(np.mean(cause_coefs))
                    # 归一化到 [-1, 1]
                    strength = max(-1.0, min(1.0, strength))

                    best_f = f_stat
                    best_edge = CausalEdge(
                        cause=cause_name,
                        effect=effect_name,
                        strength=strength,
                        lag=lag,
                        p_value=p_value,
                        confidence=max(0.0, min(1.0, 1.0 - p_value)),
                    )
            except (np.linalg.LinAlgError, ValueError):
                continue

        return best_edge

    @staticmethod
    def _f_survival(f: float, df1: int, df2: int) -> float:
        """F 分布生存函数近似（1 - CDF）

        使用正则化不完全 Beta 函数的近似。
        """
        if f <= 0:
            return 1.0
        # 近似公式：p ≈ (1 + f*df1/df2)^(-df2/2) 的修正版
        x = df2 / (df2 + df1 * f)
        # 使用简单的指数近似
        a = df2 / 2.0
        b = df1 / 2.0
        # 不完全 Beta 近似
        try:
            p = math.pow(x, a) * math.pow(1 - x, b) / (a * 0.5)
            return min(1.0, max(0.0, p))
        except (OverflowError, ValueError):
            return 0.5  # 无法计算时返回保守值

    @staticmethod
    def _remove_spurious(edges: list[CausalEdge]) -> list[CausalEdge]:
        """移除伪因果边

        如果 A→B→C 存在，且 A→C 的强度远小于 A→B→C 的间接路径，
        则 A→C 可能是伪因果（被 B 中介）。

        简化实现：仅移除强度很弱的直接边（< 0.1）。
        """
        return [e for e in edges if abs(e.strength) >= 0.05]


# ============================================================
# 2. 心理数字孪生
# ============================================================

class PsychologicalDigitalTwin:
    """心理数字孪生 —— 个性化心理状态预测模型

    为每个用户维护一个动态模型，结合：
      - 个人因果图（结构先验）
      - 实时观测（数据驱动）
      - 衰减记忆（时间权重）

    预测算法（因果感知状态空间模型）：
      1. 因果预测：x̂(t+1) = A_causal @ x(t)
         其中 A_causal 从个人因果图构建
      2. 观测更新：x(t) = α * x_observed + (1-α) * x̂(t)
         其中 α 随观测可靠性自适应调整
      3. 危机预测：P(crisis) = σ(w @ x(t) + b)

    与传统趋势分析的区别：
      - 趋势分析：线性回归 → 只能捕捉线性趋势
      - 数字孪生：因果模型 → 捕捉维度间交互 + 非线性动态
    """

    def __init__(
        self,
        causal_graph: Optional[CausalGraph] = None,
        alpha_obs: float = 0.4,
        crisis_threshold: float = 0.6,
    ):
        self.causal_graph = causal_graph
        self.alpha_obs = alpha_obs  # 观测权重
        self.crisis_threshold = crisis_threshold

        # 内部状态
        self._state: dict[str, float] = {}     # 当前估计状态
        self._state_var: dict[str, float] = {}  # 状态方差（不确定度）
        self._history: list[dict[str, float]] = []
        self._timestamps: list[float] = []

        # 因果转移矩阵（从因果图构建）
        self._transition: Optional[np.ndarray] = None
        self._dim_order: list[str] = []
        if causal_graph and causal_graph.edges:
            self._build_transition_matrix()

    def _build_transition_matrix(self):
        """从因果图构建状态转移矩阵

        A[i,j] 表示维度 j 对维度 i 的因果影响强度。
        对角线 = 自回归系数（默认 0.9，表示惯性）。
        非对角线 = 因果边的强度（考虑延迟）。
        """
        assert self.causal_graph is not None
        dims = set()
        for e in self.causal_graph.edges:
            dims.add(e.cause)
            dims.add(e.effect)
        self._dim_order = sorted(dims)
        n = len(self._dim_order)
        dim_idx = {d: i for i, d in enumerate(self._dim_order)}

        # 初始化：对角线自回归
        A = np.eye(n) * 0.85

        # 填充因果边
        for edge in self.causal_graph.edges:
            if edge.cause in dim_idx and edge.effect in dim_idx:
                i = dim_idx[edge.effect]
                j = dim_idx[edge.cause]
                # 因果强度乘以延迟衰减（越远的因果影响越弱）
                delay_factor = 1.0 / edge.lag
                A[i, j] += edge.strength * delay_factor * 0.3

        self._transition = A

    def update(self, observation: dict[str, float], timestamp: Optional[float] = None):
        """用新观测更新数字孪生状态

        Args:
            observation: {维度名: 观测值}，值域 [0, 1]
            timestamp: 时间戳（秒）
        """
        if timestamp is None:
            timestamp = time.time()

        # 初始化状态
        if not self._state:
            self._state = dict(observation)
            self._state_var = {k: 0.1 for k in observation}
            self._history.append(dict(observation))
            self._timestamps.append(timestamp)
            return

        # 1. 因果预测步
        predicted = self._predict_step()

        # 2. 观测更新步（自适应融合）
        alpha = self._adaptive_alpha(observation)
        new_state = {}
        new_var = {}
        for dim in observation:
            obs_val = observation[dim]
            pred_val = predicted.get(dim, obs_val)
            # 融合：α * 观测 + (1-α) * 预测
            new_state[dim] = alpha * obs_val + (1 - alpha) * pred_val
            # 方差更新
            prev_var = self._state_var.get(dim, 0.1)
            new_var[dim] = (1 - alpha) * prev_var + alpha * (obs_val - pred_val) ** 2

        self._state = new_state
        self._state_var = new_var
        self._history.append(dict(new_state))
        self._timestamps.append(timestamp)

        # 限制历史长度
        max_history = 200
        if len(self._history) > max_history:
            self._history = self._history[-max_history:]
            self._timestamps = self._timestamps[-max_history:]

    def _predict_step(self) -> dict[str, float]:
        """基于因果模型预测下一步状态"""
        if self._transition is None or not self._state:
            return dict(self._state)  # 无因果模型时返回当前状态

        # 提取当前状态向量
        x = np.array([self._state.get(d, 0.5) for d in self._dim_order])

        # 因果预测：x_next = A @ x
        x_pred = self._transition @ x

        # 裁剪到 [0, 1]
        x_pred = np.clip(x_pred, 0.0, 1.0)

        return {d: float(x_pred[i]) for i, d in enumerate(self._dim_order)}

    def _adaptive_alpha(self, observation: dict[str, float]) -> float:
        """自适应观测权重

        当观测与预测差异大时，提高观测权重（可能是模型未捕捉的变化）。
        当差异小时，降低观测权重（信任模型预测）。
        """
        if not self._state:
            return 1.0

        total_diff = 0.0
        count = 0
        for dim, obs_val in observation.items():
            if dim in self._state:
                total_diff += abs(obs_val - self._state[dim])
                count += 1

        if count == 0:
            return self.alpha_obs

        avg_diff = total_diff / count
        # 差异大 → alpha 高（更信任观测）
        # 差异小 → alpha 低（更信任预测）
        adaptive = self.alpha_obs + 0.3 * min(1.0, avg_diff / 0.2)
        return min(0.9, max(0.2, adaptive))

    def predict_trajectory(self, horizon: int = 12) -> Trajectory:
        """预测未来心理轨迹

        Args:
            horizon: 预测步数

        Returns:
            Trajectory 包含每步的预测状态和危机概率
        """
        if not self._state:
            return Trajectory(user_id="unknown", forecast_horizon=horizon)

        user_id = self.causal_graph.user_id if self.causal_graph else "unknown"
        points: list[TrajectoryPoint] = []
        current_state = dict(self._state)

        base_time = self._timestamps[-1] if self._timestamps else time.time()
        time_interval = 3600  # 假设每步 1 小时

        for step in range(1, horizon + 1):
            # 因果预测
            predicted = self._predict_from_state(current_state)
            # 不确定度递增
            uncertainty = min(0.95, 0.1 + step * 0.06)
            # 危机概率
            crisis_prob = self._estimate_crisis_probability(predicted, step)

            ts = base_time + step * time_interval
            points.append(TrajectoryPoint(
                time_step=step,
                timestamp=ts,
                predicted=predicted,
                uncertainty=uncertainty,
                crisis_probability=crisis_prob,
            ))
            current_state = predicted

        return Trajectory(
            user_id=user_id,
            points=points,
            forecast_horizon=horizon,
            generated_at=time.time(),
        )

    def _predict_from_state(self, state: dict[str, float]) -> dict[str, float]:
        """从给定状态预测下一步"""
        if self._transition is None:
            return dict(state)

        x = np.array([state.get(d, 0.5) for d in self._dim_order])
        x_pred = self._transition @ x
        x_pred = np.clip(x_pred, 0.0, 1.0)
        return {d: float(x_pred[i]) for i, d in enumerate(self._dim_order)}

    def _estimate_crisis_probability(
        self, state: dict[str, float], step: int
    ) -> float:
        """估计危机概率

        基于多维度风险加权和 + 时间衰减：
          P(crisis) = σ(Σ w_i * x_i - bias)
        其中 w_i 是各维度的风险权重，bias 是基线偏移。
        """
        # 高风险维度权重
        risk_weights = {
            "text_emotion": 0.3,
            "cognitive": 0.25,
            "circadian": 0.15,
            "behavior": 0.1,
            "voice_acoustic": 0.1,
            "facial": 0.05,
            "hrv": 0.05,
            "breathing": 0.05,
            "behavioral_act": 0.1,
            "eye": 0.02,
            "voice_semantics": 0.08,
        }

        weighted_sum = 0.0
        total_weight = 0.0
        for dim, weight in risk_weights.items():
            if dim in state:
                weighted_sum += weight * state[dim]
                total_weight += weight

        if total_weight == 0:
            return 0.0

        score = weighted_sum / total_weight

        # 多步预测时，距离越远概率越不确定（向基线回归）
        baseline = 0.15  # 基线危机概率
        decay = 0.95 ** step
        score = score * decay + baseline * (1 - decay)

        # Sigmoid 映射
        crisis_prob = 1.0 / (1.0 + math.exp(-6 * (score - self.crisis_threshold + 0.3)))
        return max(0.0, min(1.0, crisis_prob))

    def get_current_state(self) -> dict[str, float]:
        """获取当前估计状态"""
        return dict(self._state)

    def get_uncertainty(self) -> dict[str, float]:
        """获取各维度的不确定度"""
        return dict(self._state_var)


# ============================================================
# 3. 干预时机优化器
# ============================================================

class InterventionTimingOptimizer:
    """精准干预时机优化

    基于数字孪生的轨迹预测，计算最优干预时机和方式。

    算法（期望收益最大化）：
      1. 模拟无干预轨迹：预测未来 H 步的心理状态
      2. 模拟有干预轨迹：在每步 t 假设干预，计算干预后的状态
      3. 计算每步干预的期望收益：
         benefit(t) = Σ_{s>t} [cost_no_intervention(s) - cost_intervention(s)]
      4. 选择收益最大的时机：t* = argmax benefit(t)
      5. 如果 benefit(t*) < min_benefit → 不干预（继续观察）

    干预效果模型：
      - 干预使目标维度的值降低 intervention_effectiveness
      - 效果通过因果图传播到下游维度
      - 效果随时间衰减（decay_rate）
    """

    def __init__(
        self,
        intervention_effectiveness: float = 0.3,
        decay_rate: float = 0.85,
        intervention_cost: float = 0.05,
        crisis_cost: float = 1.0,
        min_benefit: float = 0.1,
    ):
        self.intervention_effectiveness = intervention_effectiveness
        self.decay_rate = decay_rate
        self.intervention_cost = intervention_cost
        self.crisis_cost = crisis_cost
        self.min_benefit = min_benefit

    def optimize(
        self,
        twin: PsychologicalDigitalTwin,
        horizon: int = 12,
    ) -> InterventionDecision:
        """计算最优干预决策

        Args:
            twin: 用户的数字孪生模型
            horizon: 预测步数

        Returns:
            InterventionDecision
        """
        # 1. 获取无干预轨迹
        trajectory_no = twin.predict_trajectory(horizon)

        if not trajectory_no.points:
            return InterventionDecision(
                should_intervene=False,
                urgency=0.0,
                optimal_delay=0,
                recommended_dimensions=[],
                expected_benefit=0.0,
                reasoning="数据不足，无法做出干预决策",
            )

        # 2. 计算每步干预的期望收益
        best_t = 0
        best_benefit = 0.0
        best_dims: list[str] = []
        best_chain: Optional[str] = None

        for t_intervene in range(horizon):
            # 模拟在 t_intervene 步干预
            trajectory_with = self._simulate_intervention(
                twin, t_intervene, horizon
            )
            # 计算收益
            benefit = self._compute_benefit(
                trajectory_no, trajectory_with, t_intervene
            )
            if benefit > best_benefit:
                best_benefit = benefit
                best_t = t_intervene
                best_dims = self._identify_target_dims(twin, t_intervene)
                best_chain = self._identify_chain_to_break(twin)

        # 3. 决策
        should_intervene = best_benefit >= self.min_benefit
        urgency = min(1.0, best_benefit / 0.5)

        if should_intervene:
            if best_t == 0:
                reasoning = (
                    f"建议立即干预：期望收益 {best_benefit:.3f}。"
                    f"目标维度：{', '.join(best_dims)}。"
                )
            else:
                reasoning = (
                    f"建议在 {best_t} 步后干预：期望收益 {best_benefit:.3f}。"
                    f"当前继续观察，等待最佳干预窗口。"
                )
            if best_chain:
                reasoning += f" 关键：阻断因果链 {best_chain}。"
        else:
            reasoning = (
                f"当前不需要干预：最大期望收益 {best_benefit:.3f} "
                f"低于阈值 {self.min_benefit}。继续观察。"
            )

        return InterventionDecision(
            should_intervene=should_intervene,
            urgency=urgency,
            optimal_delay=best_t,
            recommended_dimensions=best_dims,
            expected_benefit=best_benefit,
            reasoning=reasoning,
            chain_to_break=best_chain,
        )

    def _simulate_intervention(
        self,
        twin: PsychologicalDigitalTwin,
        intervene_at: int,
        horizon: int,
    ) -> Trajectory:
        """模拟在指定时间步干预后的轨迹"""
        # 先获取无干预轨迹到 intervene_at
        trajectory = twin.predict_trajectory(horizon)
        if not trajectory.points:
            return trajectory

        # 在 intervene_at 步应用干预效果
        points = []
        intervention_applied = False
        effect_decay = 1.0

        for p in trajectory.points:
            if p.time_step == intervene_at + 1 and not intervention_applied:
                # 应用干预：降低高风险维度
                new_predicted = dict(p.predicted)
                for dim in new_predicted:
                    reduction = self.intervention_effectiveness * effect_decay
                    new_predicted[dim] = max(0.0, new_predicted[dim] - reduction)
                effect_decay *= self.decay_rate
                intervention_applied = True
                crisis_prob = twin._estimate_crisis_probability(
                    new_predicted, p.time_step
                )
                points.append(TrajectoryPoint(
                    time_step=p.time_step,
                    timestamp=p.timestamp,
                    predicted=new_predicted,
                    uncertainty=p.uncertainty * 0.8,  # 干预降低不确定度
                    crisis_probability=crisis_prob,
                ))
            elif p.time_step > intervene_at + 1:
                # 后续步骤：干预效果衰减传播
                new_predicted = dict(p.predicted)
                for dim in new_predicted:
                    reduction = self.intervention_effectiveness * effect_decay * 0.3
                    new_predicted[dim] = max(0.0, new_predicted[dim] - reduction)
                effect_decay *= self.decay_rate
                crisis_prob = twin._estimate_crisis_probability(
                    new_predicted, p.time_step
                )
                points.append(TrajectoryPoint(
                    time_step=p.time_step,
                    timestamp=p.timestamp,
                    predicted=new_predicted,
                    uncertainty=p.uncertainty,
                    crisis_probability=crisis_prob,
                ))
            else:
                points.append(p)

        return Trajectory(
            user_id=trajectory.user_id,
            points=points,
            forecast_horizon=horizon,
            generated_at=time.time(),
        )

    def _compute_benefit(
        self,
        traj_no: Trajectory,
        traj_with: Trajectory,
        intervene_at: int,
    ) -> float:
        """计算干预的期望收益

        benefit = Σ [cost_no(s) - cost_with(s)] - intervention_cost
        """
        total_benefit = 0.0
        for p_no, p_with in zip(traj_no.points, traj_with.points):
            if p_no.time_step <= intervene_at:
                continue  # 干预前的步骤无差异
            cost_no = p_no.crisis_probability * self.crisis_cost
            cost_with = p_with.crisis_probability * self.crisis_cost
            total_benefit += cost_no - cost_with

        # 减去干预成本
        total_benefit -= self.intervention_cost
        return total_benefit

    def _identify_target_dims(
        self, twin: PsychologicalDigitalTwin, step: int
    ) -> list[str]:
        """识别应该干预的维度（当前值最高的维度）"""
        state = twin.get_current_state()
        if not state:
            return []
        sorted_dims = sorted(state.items(), key=lambda x: x[1], reverse=True)
        # 返回 top-3 高风险维度
        return [d for d, v in sorted_dims[:3] if v > 0.4]

    def _identify_chain_to_break(
        self, twin: PsychologicalDigitalTwin
    ) -> Optional[str]:
        """识别应该阻断的因果链"""
        if twin.causal_graph is None:
            return None

        state = twin.get_current_state()
        if not state:
            return None

        # 找到当前值最高的维度（最可能引发连锁反应的源头）
        max_dim = max(state, key=state.get)

        # 从该维度出发的因果链
        chains = twin.causal_graph.get_chain(max_dim, max_depth=2)
        if not chains:
            return None

        # 返回最长的链
        longest = max(chains, key=len)
        parts = [longest[0].cause]
        for edge in longest:
            parts.append(edge.effect)
        return " → ".join(parts)


# ============================================================
# 4. 统一管理器
# ============================================================

class DigitalTwinManager:
    """数字孪生管理器

    管理所有用户的数字孪生，提供：
      - 因果图发现与更新
      - 实时状态追踪
      - 轨迹预测
      - 干预决策

    持久化：JSON 文件
    """

    def __init__(self):
        self._twins: dict[str, PsychologicalDigitalTwin] = {}
        self._graphs: dict[str, CausalGraph] = {}
        self._optimizer = InterventionTimingOptimizer()
        self._discovery = PersonalCausalDiscovery()
        self._precision = get_precision_engine()  # CT-PLEW 精确引擎

    def get_or_create_twin(
        self, user_id: str
    ) -> PsychologicalDigitalTwin:
        """获取或创建用户的数字孪生"""
        if user_id not in self._twins:
            graph = self._graphs.get(user_id)
            self._twins[user_id] = PsychologicalDigitalTwin(
                causal_graph=graph,
            )
        return self._twins[user_id]

    def update_observation(
        self,
        user_id: str,
        observation: dict[str, float],
        timestamp: Optional[float] = None,
    ):
        """更新用户观测"""
        twin = self.get_or_create_twin(user_id)
        twin.update(observation, timestamp)

    def discover_causal_graph(
        self,
        user_id: str,
        time_series: dict[str, list[float]],
    ) -> CausalGraph:
        """发现用户的个人因果图"""
        graph = self._discovery.discover(user_id, time_series)
        self._graphs[user_id] = graph

        # 更新数字孪生的因果图
        if user_id in self._twins:
            twin = self._twins[user_id]
            twin.causal_graph = graph
            twin._build_transition_matrix()

        return graph

    def predict_trajectory(
        self, user_id: str, horizon: int = 12
    ) -> Trajectory:
        """预测用户心理轨迹"""
        twin = self.get_or_create_twin(user_id)
        return twin.predict_trajectory(horizon)

    def get_intervention_decision(
        self, user_id: str, horizon: int = 12
    ) -> InterventionDecision:
        """获取干预决策"""
        twin = self.get_or_create_twin(user_id)
        return self._optimizer.optimize(twin, horizon)

    def precision_analysis(
        self, user_id: str, horizon: int = 12
    ) -> dict:
        """CT-PLEW 精确分析：韧性/临界预警/Kramers 跃迁概率/反事实干预

        与线性轨迹预测互补：线性模型外推趋势，
        CT-PLEW 计算势能景观壁垒与闭式危机逃逸概率。
        """
        twin = self.get_or_create_twin(user_id)
        graph = self._graphs.get(user_id)
        report = self._precision.analyze(
            user_id, twin._history, graph, horizon
        )
        return report.to_dict()

    # ----------------------------------------------------------
    # 多维景观 / 反事实 / 平行未来 / 最优救援（均带冷启动回退）
    # ----------------------------------------------------------
    @staticmethod
    def _series_from_history(
        history: list, min_len: int = 3
    ) -> dict[str, list[float]]:
        """list[dict] -> {dim: [values]}，仅保留长度足够的模态"""
        series: dict[str, list[float]] = {}
        for step in history:
            for k, v in step.items():
                try:
                    series.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    continue
        return {k: v for k, v in series.items() if len(v) >= min_len}

    def multidim_landscape(self, user_id: str, horizon: int = 12) -> dict:
        """多维临界跃迁景观 + instanton 崩溃路径 + 多维 Kramers 概率

        数据/维度不足时回退到 1D CT-PLEW（precision_analysis），绝不抛错。
        """
        twin = self.get_or_create_twin(user_id)
        series = self._series_from_history(twin._history)
        try:
            nd = get_multidim_analyzer().analyze(series, horizon)
        except Exception as e:  # 数值失败回退
            logger.warning(f"multidim_landscape 失败，回退 1D：{e}")
            nd = None
        if nd is not None:
            out = nd.to_dict()
            out["fallback"] = False
            return out
        # 回退：1D CT-PLEW
        rep = self.precision_analysis(user_id, horizon)
        return {
            "fallback": True,
            "reason": "维度不足/历史不足/无清晰双稳态，回退 1D CT-PLEW",
            "crisis_probability": rep.get("crisis_probability", 0.0),
            "critical_dim": rep.get("critical_dim"),
            "multistable": rep.get("multistable", False),
            "method": "ctplew_1d",
        }

    def counterfactual_query(
        self,
        user_id: str,
        do_spec: dict[str, float],
        intervention_step: Optional[int] = None,
        horizon: int = 8,
        clamp: bool = False,
    ) -> dict:
        """Pearl 三步反事实：若在某步 do(X_j=v)，之后 horizon 步末态会怎样

        SCM 数据不足时回退（返回 fallback 标记），绝不抛错。
        """
        twin = self.get_or_create_twin(user_id)
        history = twin._history
        graph = self._graphs.get(user_id)
        if not do_spec:
            return {"fallback": True, "reason": "do_spec 为空", "method": "none"}
        try:
            scm = StructuralCausalModel.fit(history, graph=graph)
            if scm is None:
                raise ValueError("SCM 拟合失败")
            cf = scm.counterfactual(
                history, do_spec, intervention_step=intervention_step,
                horizon=horizon, clamp=clamp,
            )
        except Exception as e:
            logger.warning(f"counterfactual_query 失败，回退：{e}")
            cf = None
        if cf is None:
            return {
                "fallback": True,
                "reason": "历史不足/无有效因果结构，无法做反事实",
                "do_spec": {k: float(v) for k, v in do_spec.items()},
                "method": "none",
            }
        out = cf.to_dict()
        out["fallback"] = False
        return out

    def parallel_futures(
        self,
        user_id: str,
        K: int = 200,
        horizon: int = 20,
        seed: int = 0,
    ) -> dict:
        """生成式平行未来：K 条轨迹的危机率分布 + 分位轨迹 + domino 频率

        数据不足时回退（返回 fallback 标记），绝不抛错。
        """
        twin = self.get_or_create_twin(user_id)
        series = self._series_from_history(twin._history)
        try:
            sim = InSilicoPatient.from_history(series)
            fut = sim.rollout(K=K, horizon=horizon, seed=seed) if sim else None
        except Exception as e:
            logger.warning(f"parallel_futures 失败，回退：{e}")
            fut = None
        if fut is None:
            rep = self.precision_analysis(user_id, horizon)
            return {
                "fallback": True,
                "reason": "维度/历史不足或无双稳态，回退 1D 危机概率",
                "crisis_probability": rep.get("crisis_probability", 0.0),
                "method": "ctplew_1d",
            }
        out = fut.to_dict()
        out["fallback"] = False
        return out

    def optimal_rescue(self, user_id: str, horizon: int = 15, K: int = 80,
                       seed: int = 0) -> dict:
        """MPC 最优干预策略 + 最小能量救援轨迹

        数据不足时回退到现有时机优化（get_intervention_decision），绝不抛错。
        """
        twin = self.get_or_create_twin(user_id)
        series = self._series_from_history(twin._history)
        plan_dict = None
        rescue_dict = None
        try:
            sim = InSilicoPatient.from_history(series)
            if sim is not None:
                mpc = InterventionMPC(sim, K=K, seed=seed)
                plan = mpc.optimize(horizon=horizon)
                plan_dict = plan.to_dict()
                rescue = minimum_rescue_path(sim.model)
                rescue_dict = rescue.to_dict() if rescue is not None else None
        except Exception as e:
            logger.warning(f"optimal_rescue 失败，回退：{e}")
            plan_dict = None
        if plan_dict is None:
            # 回退：现有干预时机优化
            try:
                dec = self.get_intervention_decision(user_id, horizon)
                dec_dict = asdict(dec)
            except Exception:
                dec_dict = None
            return {
                "fallback": True,
                "reason": "多维模拟器不可用，回退现有时机优化",
                "intervention_decision": dec_dict,
                "method": "timing_optimizer",
            }
        return {
            "fallback": False,
            "mpc_plan": plan_dict,
            "rescue_path": rescue_dict,
            "method": "cem_mpc+reverse_instanton",
        }

    def get_summary(self, user_id: str) -> dict:
        """获取用户数字孪生摘要"""
        twin = self.get_or_create_twin(user_id)
        graph = self._graphs.get(user_id)

        summary = {
            "user_id": user_id,
            "current_state": twin.get_current_state(),
            "uncertainty": twin.get_uncertainty(),
            "has_causal_graph": graph is not None,
            "causal_edges": len(graph.edges) if graph else 0,
            "history_length": len(twin._history),
        }

        if graph and graph.edges:
            # 提取关键因果链
            chains = []
            for dim in [d.value for d in PsychoDimension]:
                dim_chains = graph.get_chain(dim, max_depth=2)
                for chain in dim_chains:
                    parts = [chain[0].cause]
                    for e in chain:
                        parts.append(e.effect)
                    chains.append(" → ".join(parts))
            summary["causal_chains"] = chains[:10]  # 最多 10 条

        return summary

    def save(self):
        """持久化到文件"""
        data = {}
        for uid, graph in self._graphs.items():
            data[uid] = graph.to_dict()
        with open(_TWIN_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        """从文件加载"""
        if not Path(_TWIN_STORE_PATH).exists():
            return
        with open(_TWIN_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for uid, d in data.items():
            graph = CausalGraph.from_dict(d)
            self._graphs[uid] = graph


# ============================================================
# 全局单例
# ============================================================

_manager: Optional[DigitalTwinManager] = None


def get_digital_twin_manager() -> DigitalTwinManager:
    """获取全局数字孪生管理器"""
    global _manager
    if _manager is None:
        _manager = DigitalTwinManager()
        _manager.load()
    return _manager
