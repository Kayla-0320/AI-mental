"""
心理数字孪生 API —— 轨迹预测 + 干预决策 + 因果图查询

端点：
    POST /digital-twin/predict      → 预测用户未来心理轨迹（含 CT-PLEW 精确分析）
    POST /digital-twin/precision    → CT-PLEW 势能景观/临界跃迁精确分析
    POST /digital-twin/intervention  → 获取干预决策
    POST /digital-twin/causal-graph  → 查询个人因果图
    POST /digital-twin/summary       → 获取数字孪生摘要
    POST /digital-twin/discover      → 触发因果图发现（需足够历史数据）
    POST /digital-twin/landscape     → 多维临界跃迁景观 + instanton 崩溃路径
    POST /digital-twin/counterfactual → do/反事实查询（Pearl 三步）
    POST /digital-twin/futures       → 生成式平行未来（K 条轨迹）
    POST /digital-twin/rescue        → MPC 最优干预 + 最小能量救援
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from assessment.digital_twin import (
    get_digital_twin_manager,
    PersonalCausalDiscovery,
    PsychoDimension,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/digital-twin", tags=["心理数字孪生"])


# ============================================================
# 请求/响应模型
# ============================================================

class PredictRequest(BaseModel):
    """轨迹预测请求"""
    user_id: str = Field(..., description="用户 ID")
    horizon: int = Field(12, description="预测步数（默认 12，每步约 1 小时）", ge=1, le=48)


class PredictResponse(BaseModel):
    """轨迹预测响应"""
    user_id: str
    trajectory: dict = Field(..., description="轨迹预测点列表")
    intervention: dict = Field(..., description="干预决策")
    twin_summary: dict = Field(..., description="数字孪生摘要")
    precision: dict = Field(
        default_factory=dict,
        description="CT-PLEW 精确分析：势能景观韧性/临界预警/Kramers 跃迁概率/反事实干预",
    )


class PrecisionRequest(BaseModel):
    """CT-PLEW 精确分析请求"""
    user_id: str = Field(..., description="用户 ID")
    horizon: int = Field(12, description="跃迁概率计算窗口步数", ge=1, le=48)


class InterventionRequest(BaseModel):
    """干预决策请求"""
    user_id: str = Field(..., description="用户 ID")
    horizon: int = Field(12, description="优化窗口步数", ge=1, le=48)


class InterventionResponse(BaseModel):
    """干预决策响应"""
    user_id: str
    should_intervene: bool
    urgency: float
    optimal_delay: int
    recommended_dimensions: list[str]
    expected_benefit: float
    reasoning: str
    chain_to_break: Optional[str] = None


class CausalGraphRequest(BaseModel):
    """因果图查询请求"""
    user_id: str = Field(..., description="用户 ID")


class CausalGraphResponse(BaseModel):
    """因果图响应"""
    user_id: str
    edges: list[dict]
    n_observations: int
    causal_chains: list[str]
    root_causes: list[str]


class DiscoverRequest(BaseModel):
    """因果图发现请求"""
    user_id: str = Field(..., description="用户 ID")
    time_series: dict[str, list[float]] = Field(
        ..., description="{维度名: [观测值序列]}，至少需要 20 个时间步"
    )
    max_lag: int = Field(5, description="最大因果延迟步数", ge=1, le=10)
    significance_level: float = Field(0.05, description="显著性水平", ge=0.001, le=0.2)


class DiscoverResponse(BaseModel):
    """因果图发现响应"""
    user_id: str
    n_edges: int
    edges: list[dict]
    n_observations: int
    message: str


class SummaryRequest(BaseModel):
    """摘要请求"""
    user_id: str = Field(..., description="用户 ID")


class LandscapeRequest(BaseModel):
    """多维临界跃迁景观请求"""
    user_id: str = Field(..., description="用户 ID")
    horizon: int = Field(12, description="危机概率窗口步数", ge=1, le=48)


class CounterfactualRequest(BaseModel):
    """do/反事实查询请求"""
    user_id: str = Field(..., description="用户 ID")
    do_spec: dict[str, float] = Field(
        ..., description="被干预变量 -> 目标值，如 {\"sleep\": 0.8}"
    )
    intervention_step: Optional[int] = Field(
        None, description="干预发生的时间步（默认为历史中点）", ge=0
    )
    horizon: int = Field(8, description="干预后观察窗口步数", ge=1, le=48)
    clamp: bool = Field(False, description="True=持续钳制干预，False=单次脉冲")


class FuturesRequest(BaseModel):
    """平行未来请求"""
    user_id: str = Field(..., description="用户 ID")
    n_particles: int = Field(200, description="平行未来轨迹数 K", ge=10, le=500)
    horizon: int = Field(20, description="前向模拟步数", ge=1, le=100)
    seed: int = Field(0, description="随机种子（可复现）")


class RescueRequest(BaseModel):
    """最优救援请求"""
    user_id: str = Field(..., description="用户 ID")
    horizon: int = Field(15, description="MPC 优化时域步数", ge=1, le=40)
    n_particles: int = Field(80, description="每次评估的粒子数 K", ge=10, le=300)
    seed: int = Field(0, description="随机种子")


# ============================================================
# 路由端点
# ============================================================

@router.post("/predict", response_model=PredictResponse, summary="心理轨迹预测")
async def predict_trajectory(request: PredictRequest) -> PredictResponse:
    """预测用户未来心理轨迹

    基于心理数字孪生的因果感知状态空间模型，结合个人因果图，
    预测用户未来 N 步的心理状态轨迹和危机概率。

    核心创新：
    - 不是简单的线性外推，而是基于因果关系的动态预测
    - 不确定度随预测步数递增
    - 危机概率基于多维度风险加权
    - 附带 CT-PLEW 精确分析（势能景观韧性 + Kramers 临界跃迁概率）
    """
    mgr = get_digital_twin_manager()
    result = _get_prediction(mgr, request.user_id, request.horizon)
    return PredictResponse(**result)


@router.post("/precision", summary="CT-PLEW 精确分析（势能景观+临界跃迁）")
async def precision_analysis(request: PrecisionRequest) -> dict:
    """CT-PLEW：心理势能景观与临界跃迁预警算法（原创精确引擎）

    与线性轨迹预测互补，提供解析精确的非线性分析：
    - 势能景观：个体化吸引子（心理稳态）与势能壁垒（临界点）
    - 韧性度量：ΔU/D 无量纲壁垒高度（越大越难崩溃）
    - Kramers 逃逸率：闭式计算 horizon 内危机概率，无需蒙特卡洛
    - 临界慢化预警：自相关/方差趋势捕捉跃迁前兆
    - 反事实干预：闭式重算各候选靶点的干预收益
    """
    mgr = get_digital_twin_manager()
    return mgr.precision_analysis(request.user_id, request.horizon)


@router.post("/intervention", response_model=InterventionResponse, summary="干预时机优化")
async def get_intervention(request: InterventionRequest) -> InterventionResponse:
    """获取最优干预决策

    基于数字孪生轨迹预测，使用期望收益最大化算法计算：
    - 是否应该干预
    - 最优干预时机（立即 vs 等待）
    - 建议干预的维度
    - 应该阻断的因果链

    核心创新：
    - JITAI（Just-In-Time Adaptive Intervention）精准时机干预
    - 基于因果图的干预效果传播模型
    - 期望收益 = Σ(无干预成本 - 有干预成本) - 干预代价
    """
    mgr = get_digital_twin_manager()
    decision = mgr.get_intervention_decision(request.user_id, request.horizon)
    return InterventionResponse(
        user_id=request.user_id,
        should_intervene=decision.should_intervene,
        urgency=round(decision.urgency, 4),
        optimal_delay=decision.optimal_delay,
        recommended_dimensions=decision.recommended_dimensions,
        expected_benefit=round(decision.expected_benefit, 4),
        reasoning=decision.reasoning,
        chain_to_break=decision.chain_to_break,
    )


@router.post("/causal-graph", response_model=CausalGraphResponse, summary="查询个人因果图")
async def get_causal_graph(request: CausalGraphRequest) -> CausalGraphResponse:
    """查询用户的个人因果图

    返回该用户独有的症状因果关系图：
    - 哪些症状会导致哪些症状
    - 因果延迟（lag）
    - 因果强度

    核心创新：
    - 每个人的因果图不同——个性化心理建模
    - 基于 Granger 因果检验，不是简单的相关性
    """
    mgr = get_digital_twin_manager()
    summary = mgr.get_summary(request.user_id)

    edges = []
    chains = summary.get("causal_chains", [])
    graph = mgr._graphs.get(request.user_id)

    if graph:
        edges = [
            {
                "cause": e.cause,
                "effect": e.effect,
                "strength": round(e.strength, 4),
                "lag": e.lag,
                "p_value": round(e.p_value, 4),
                "confidence": round(e.confidence, 4),
            }
            for e in graph.edges
        ]

    # 识别根因（只有出边没有入边的维度）
    root_causes = _find_root_causes(graph) if graph else []

    return CausalGraphResponse(
        user_id=request.user_id,
        edges=edges,
        n_observations=summary.get("causal_edges", 0),
        causal_chains=chains[:10],
        root_causes=root_causes,
    )


@router.post("/discover", response_model=DiscoverResponse, summary="触发因果图发现")
async def discover_causal_graph(request: DiscoverRequest) -> DiscoverResponse:
    """基于历史数据发现用户个人因果图

    使用 Granger 因果检验分析多维度时间序列数据，
    发现症状之间的因果关系。

    需要至少 20 个时间步的观测数据。
    """
    mgr = get_digital_twin_manager()
    discovery = PersonalCausalDiscovery(
        max_lag=request.max_lag,
        significance_level=request.significance_level,
    )
    graph = discovery.discover(request.user_id, request.time_series)

    # 存储到管理器
    mgr._graphs[request.user_id] = graph

    edges = [
        {
            "cause": e.cause,
            "effect": e.effect,
            "strength": round(e.strength, 4),
            "lag": e.lag,
            "p_value": round(e.p_value, 4),
        }
        for e in graph.edges
    ]

    msg = (
        f"因果图发现完成：{len(request.time_series)} 维度，"
        f"{graph.n_observations} 观测，{len(edges)} 条因果边"
    )

    return DiscoverResponse(
        user_id=request.user_id,
        n_edges=len(edges),
        edges=edges,
        n_observations=graph.n_observations,
        message=msg,
    )


@router.post("/summary", summary="数字孪生摘要")
async def get_summary(request: SummaryRequest) -> dict:
    """获取用户数字孪生完整摘要

    包含：当前状态估计、各维度不确定度、因果图信息、历史长度。
    """
    mgr = get_digital_twin_manager()
    return mgr.get_summary(request.user_id)


@router.post("/landscape", summary="多维临界跃迁景观（心理地震预警）")
async def multidim_landscape(request: LandscapeRequest) -> dict:
    """多维势能景观 + instanton 崩溃路径 + 多维 Kramers 逃逸概率

    将 11 模态降维到 2-3 维潜在风险流形，估计多维耦合动力系统：
    - 拟势 U 与 Helmholtz 梯度-通量分解（gradient_flux_ratio）
    - 吸引子/鞍点与壁垒高度
    - 最小能量崩溃路径 → 反投影为个体化 domino 链（哪个模态先崩）
    - 多维 Kramers/Langer 逃逸率 + 粒子群经验交叉验证（escape_ci）

    数据/维度不足时自动回退 1D CT-PLEW（fallback=true）。
    """
    mgr = get_digital_twin_manager()
    return mgr.multidim_landscape(request.user_id, request.horizon)


@router.post("/counterfactual", summary="do/反事实查询（Pearl 三步）")
async def counterfactual_query(request: CounterfactualRequest) -> dict:
    """结构因果模型上的 do-干预与 Pearl 三步反事实

    Abduction（由观测反推外生噪声）→ Action（do）→ Prediction（重算），
    回答"若这个孩子多睡 2h，其情绪轨迹会怎样"。

    SCM 数据不足时回退（fallback=true）。
    """
    mgr = get_digital_twin_manager()
    return mgr.counterfactual_query(
        request.user_id, request.do_spec,
        intervention_step=request.intervention_step,
        horizon=request.horizon, clamp=request.clamp,
    )


@router.post("/futures", summary="生成式平行未来")
async def parallel_futures(request: FuturesRequest) -> dict:
    """In-Silico Patient：在同一个体化 f、Σ 上模拟 K 条平行未来

    返回危机率分布 + Wilson 置信区间、各模态分位轨迹扇形带、
    domino 命中频率（哪个模态先崩）与平均首次崩溃时间。

    数据不足时回退 1D 危机概率（fallback=true）。
    """
    mgr = get_digital_twin_manager()
    return mgr.parallel_futures(
        request.user_id, K=request.n_particles,
        horizon=request.horizon, seed=request.seed,
    )


@router.post("/rescue", summary="MPC 最优干预 + 最小能量救援")
async def optimal_rescue(request: RescueRequest) -> dict:
    """在虚拟世界里排练如何阻止崩溃

    - InterventionMPC（交叉熵法）：滚动优化干预时机+靶点+剂量，
      目标 min E[危机成本] + λ·干预能量；返回基线 vs 干预后危机率。
    - 最小能量救援轨迹：崩溃 instanton 的逆向，闭式给出救援剂量方向
      与救援后重新抬高的壁垒。

    多维模拟器不可用时回退现有干预时机优化（fallback=true）。
    """
    mgr = get_digital_twin_manager()
    return mgr.optimal_rescue(
        request.user_id, horizon=request.horizon,
        K=request.n_particles, seed=request.seed,
    )


# ============================================================
# 辅助函数
# ============================================================

def _get_prediction(mgr, user_id: str, horizon: int) -> dict:
    """获取完整的数字孪生预测（线性轨迹 + CT-PLEW 精确分析）"""
    traj = mgr.predict_trajectory(user_id, horizon)
    decision = mgr.get_intervention_decision(user_id, horizon)
    summary = mgr.get_summary(user_id)
    precision = mgr.precision_analysis(user_id, horizon)

    return {
        "user_id": user_id,
        "trajectory": {
            "points": [
                {
                    "time_step": p.time_step,
                    "predicted": {k: round(v, 4) for k, v in p.predicted.items()},
                    "uncertainty": round(p.uncertainty, 4),
                    "crisis_probability": round(p.crisis_probability, 4),
                }
                for p in traj.points
            ],
            "max_crisis_risk": round(traj.get_crisis_risk(), 4),
            "deterioration_time": traj.get_deterioration_time(),
        },
        "intervention": {
            "should_intervene": decision.should_intervene,
            "urgency": round(decision.urgency, 4),
            "optimal_delay": decision.optimal_delay,
            "recommended_dimensions": decision.recommended_dimensions,
            "expected_benefit": round(decision.expected_benefit, 4),
            "reasoning": decision.reasoning,
            "chain_to_break": decision.chain_to_break,
        },
        "twin_summary": summary,
        "precision": precision,
    }


def _find_root_causes(graph) -> list[str]:
    """识别因果图中的根因维度（只有出边没有入边）"""
    if not graph or not graph.edges:
        return []

    causes = set()
    effects = set()
    for e in graph.edges:
        causes.add(e.cause)
        effects.add(e.effect)

    # 根因：是 cause 但不是 effect
    roots = causes - effects
    return sorted(roots)
