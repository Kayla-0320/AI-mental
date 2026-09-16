"""
公平性审计路由 —— 算法公平性 API 端点
"""
from fastapi import APIRouter

router = APIRouter(prefix="/audit", tags=["公平性审计"])

# TODO: 偏差检测接口
# TODO: 公平性指标接口
# TODO: 可解释性报告接口
