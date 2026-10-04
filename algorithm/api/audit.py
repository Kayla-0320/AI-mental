"""
公平性审计路由 —— 算法公平性 API 端点

暴露底层 audit/fairness.py 的功能：
- POST /audit/fairness       → run_full_audit()
- GET  /audit/report/{id}    → render_markdown_report()
- GET  /audit/chart/{id}     → generate_chart()
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from audit.fairness import (
    run_full_audit,
    render_markdown_report,
    generate_chart,
    FairnessReport,
    GroupMetrics,
    REPORT_PATH,
    CHART_PATH,
)

router = APIRouter(prefix="/audit", tags=["公平性审计"])

# 内存报告存储（演示用）
_report_store: dict[str, dict] = {}


# ============================================================
# 请求 / 响应模型
# ============================================================

class FairnessAuditRequest(BaseModel):
    """公平性审计请求"""
    predictions: list[int] = Field(..., description="模型预测标签")
    labels: list[int] = Field(..., description="真实标签")
    demographics: dict = Field(
        ...,
        description="人口统计学信息，包含 ages/genders/regions 列表",
    )


class GroupMetricsResponse(BaseModel):
    group_name: str
    dimension: str
    sample_size: int
    accuracy: float
    f1: float
    tpr: float
    fpr: float
    insufficient_sample: bool = False


class FairnessAuditResponse(BaseModel):
    """公平性审计响应"""
    report_id: str
    demographic_parity_diff: float
    equalized_odds_diff: float
    worst_group: str
    recommendations: list[str]
    group_metrics: list[GroupMetricsResponse]
    timestamp: str


class ReportResponse(BaseModel):
    report_id: str
    content: str


# ============================================================
# 端点
# ============================================================

@router.post("/fairness", response_model=FairnessAuditResponse)
async def audit_fairness(request: FairnessAuditRequest):
    """运行公平性审计

    调用 audit/fairness.py 的 run_full_audit()，按年龄、性别、地域
    分组计算各组指标，生成 Markdown 报告和性能对比图。
    """
    if len(request.predictions) != len(request.labels):
        raise HTTPException(
            status_code=400,
            detail="predictions 和 labels 长度必须相同",
        )

    # 检查 demographics 必要字段
    demos = request.demographics
    for key in ("ages", "genders", "regions"):
        if key not in demos:
            raise HTTPException(
                status_code=400,
                detail=f"demographics 缺少必要字段: {key}",
            )

    n = len(request.predictions)
    for key in ("ages", "genders", "regions"):
        if len(demos[key]) != n:
            raise HTTPException(
                status_code=400,
                detail=f"demographics.{key} 长度必须与 predictions 相同",
            )

    predictions = np.array(request.predictions)
    labels = np.array(request.labels)

    report: FairnessReport = run_full_audit(predictions, labels, demos)

    report_id = uuid.uuid4().hex[:8]
    _report_store[report_id] = {
        "report": report,
        "demographics": demos,
    }

    return FairnessAuditResponse(
        report_id=report_id,
        demographic_parity_diff=report.demographic_parity_diff,
        equalized_odds_diff=report.equalized_odds_diff,
        worst_group=report.worst_group,
        recommendations=report.recommendations,
        group_metrics=[
            GroupMetricsResponse(**asdict(m)) for m in report.group_metrics
        ],
        timestamp=report.timestamp,
    )


@router.get("/report/{report_id}", response_model=ReportResponse)
async def get_audit_report(report_id: str):
    """获取审计报告（Markdown 格式）"""
    if report_id not in _report_store:
        raise HTTPException(status_code=404, detail="报告不存在")

    stored = _report_store[report_id]
    report: FairnessReport = stored["report"]
    content = render_markdown_report(report)

    return ReportResponse(report_id=report_id, content=content)


@router.get("/chart/{report_id}")
async def get_audit_chart(report_id: str):
    """获取可视化图表（PNG 图片）"""
    if report_id not in _report_store:
        raise HTTPException(status_code=404, detail="报告不存在")

    stored = _report_store[report_id]
    report: FairnessReport = stored["report"]

    chart_path = generate_chart(report)
    if not chart_path:
        raise HTTPException(status_code=500, detail="图表生成失败")

    return FileResponse(
        chart_path,
        media_type="image/png",
        filename=f"fairness_chart_{report_id}.png",
    )
