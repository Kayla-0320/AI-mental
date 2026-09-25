"""
SIM-VAIL 动态测试路由

端点：
  - POST /sim-vail/run     一键运行测试
  - GET  /sim-vail/report  获取最近测试报告
  - GET  /sim-vail/profiles 获取画像列表
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from sim_vail import (
    SimVailTestEngine,
    get_profiles,
    OUTPUT_DIR,
)
from sim_vail.report import (
    generate_json_report,
    generate_markdown_report,
    save_reports,
    generate_comparison_report,
    save_comparison_report,
)

router = APIRouter(prefix="/sim-vail", tags=["SIM-VAIL 动态测试"])


# ============================================================
# 请求/响应模型
# ============================================================

class SimVailRunRequest(BaseModel):
    """SIM-VAIL 测试运行请求"""
    use_safety_loop: bool = Field(
        False, description="是否使用 SafetyLoop 闭环"
    )
    max_turns: Optional[int] = Field(
        None, description="每个画像的最大对话轮次（默认使用画像定义）"
    )
    save_to_file: bool = Field(
        True, description="是否保存报告到文件"
    )
    output_format: str = Field(
        "both", description="输出格式: json / markdown / both"
    )


class SimVailRunResponse(BaseModel):
    """SIM-VAIL 测试运行响应"""
    run_id: str
    total_profiles: int
    total_turns: int
    interception_rate: float
    miss_rate: float
    false_alarm_rate: float
    avg_likert_score: float
    profile_summaries: list[dict]
    report_files: dict[str, str] = Field(default_factory=dict)
    json_report: Optional[dict] = None
    markdown_report: Optional[str] = None


class SimVailProfileInfo(BaseModel):
    """画像信息"""
    profile_id: str
    name: str
    description: str
    clinical_basis: str
    initial_emotion: str
    dimensions: dict[str, float]
    total_turns: int
    expected_crisis_turns: list[int]


class SimVailComparisonResponse(BaseModel):
    """对比实验响应"""
    run_id: str
    overall_baseline_interception_rate: float
    overall_no_baseline_interception_rate: float
    overall_improvement: float
    total_unique_interceptions: int
    profile_comparisons: list[dict]
    markdown_report: str
    report_file: str = ""


# ============================================================
# 最近报告缓存
# ============================================================

_last_report = None


# ============================================================
# 路由端点
# ============================================================

@router.post("/run", summary="一键运行 SIM-VAIL 测试")
async def run_sim_vail(request: SimVailRunRequest) -> SimVailRunResponse:
    """运行 SIM-VAIL 动态测试

    流程：
      1. 加载 5 个青少年脆弱性画像
      2. 对每个画像运行多轮模拟对话
      3. 每轮调用五轴审计器
      4. 计算拦截率/漏判率/误报率
      5. 生成 JSON + Markdown 报告
    """
    engine = SimVailTestEngine(
        use_safety_loop=request.use_safety_loop,
        max_turns=request.max_turns,
    )

    report = engine.run_all()

    # 保存报告
    report_files = {}
    if request.save_to_file:
        report_files = save_reports(report)

    # 构建响应
    json_report = None
    markdown_report = None

    if request.output_format in ("json", "both"):
        import json as json_lib
        json_report = json_lib.loads(generate_json_report(report))

    if request.output_format in ("markdown", "both"):
        markdown_report = generate_markdown_report(report)

    # 缓存报告
    global _last_report
    _last_report = report

    # 画像摘要
    profile_summaries = []
    for pr in report.profile_results:
        profile_summaries.append({
            "profile_id": pr.profile_id,
            "profile_name": pr.profile_name,
            "total_turns": pr.total_turns,
            "intercepted": pr.interception_count,
            "missed": pr.miss_count,
            "false_alarms": pr.false_alarm_count,
            "avg_likert": round(pr.avg_likert, 2),
            "max_escalation": round(pr.max_escalation_reached, 2),
        })

    return SimVailRunResponse(
        run_id=report.run_id,
        total_profiles=report.total_profiles,
        total_turns=report.total_turns,
        interception_rate=round(report.overall_interception_rate, 4),
        miss_rate=round(report.overall_miss_rate, 4),
        false_alarm_rate=round(report.overall_false_alarm_rate, 4),
        avg_likert_score=round(report.avg_likert_score, 2),
        profile_summaries=profile_summaries,
        report_files=report_files,
        json_report=json_report,
        markdown_report=markdown_report,
    )


@router.get("/report", summary="获取最近测试报告")
async def get_latest_report(format: str = "json"):
    """获取最近一次 SIM-VAIL 测试报告

    Args:
        format: 报告格式 (json / markdown)
    """
    if _last_report is None:
        return {"error": "暂无测试报告，请先运行 POST /sim-vail/run"}

    if format == "markdown":
        return {
            "format": "markdown",
            "content": generate_markdown_report(_last_report),
        }
    else:
        import json as json_lib
        return json_lib.loads(generate_json_report(_last_report))


@router.get("/profiles", summary="获取画像列表")
async def get_profile_list() -> list[SimVailProfileInfo]:
    """获取所有青少年脆弱性画像信息"""
    profiles = get_profiles()
    return [
        SimVailProfileInfo(
            profile_id=p.profile_id,
            name=p.name,
            description=p.description,
            clinical_basis=p.clinical_basis,
            initial_emotion=p.initial_emotion,
            dimensions=p.dimensions,
            total_turns=len(p.turn_scripts),
            expected_crisis_turns=p.expected_crisis_turns,
        )
        for p in profiles
    ]


@router.post("/comparison", summary="运行基线驱动对比实验")
async def run_comparison() -> SimVailComparisonResponse:
    """运行基线驱动 vs 非基线驱动对比实验

    对 5 个画像分别跑两次（baseline_driven=True/False），
    对比拦截率、漏判率、误报率，生成对比报告。
    报告保存到 sim_vail/outputs/comparison_report.md。
    """
    engine = SimVailTestEngine()
    comparison = engine.run_comparison()

    # 保存报告
    report_file = save_comparison_report(comparison)

    # 构建画像对比摘要
    profile_comparisons = []
    for pc in comparison.profile_comparisons:
        profile_comparisons.append({
            "profile_id": pc.profile_id,
            "profile_name": pc.profile_name,
            "baseline_interception_rate": pc.baseline_interception_rate,
            "no_baseline_interception_rate": pc.no_baseline_interception_rate,
            "baseline_miss_rate": pc.baseline_miss_rate,
            "no_baseline_miss_rate": pc.no_baseline_miss_rate,
            "baseline_false_alarm_rate": pc.baseline_false_alarm_rate,
            "no_baseline_false_alarm_rate": pc.no_baseline_false_alarm_rate,
            "unique_interceptions": pc.unique_interceptions,
        })

    return SimVailComparisonResponse(
        run_id=comparison.run_id,
        overall_baseline_interception_rate=comparison.overall_baseline_interception_rate,
        overall_no_baseline_interception_rate=comparison.overall_no_baseline_interception_rate,
        overall_improvement=comparison.overall_improvement,
        total_unique_interceptions=comparison.total_unique_interceptions,
        profile_comparisons=profile_comparisons,
        markdown_report=generate_comparison_report(comparison),
        report_file=report_file,
    )
