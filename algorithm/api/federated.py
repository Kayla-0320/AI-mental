"""
联邦学习路由 —— 分布式训练 API 端点
"""
from fastapi import APIRouter

router = APIRouter(prefix="/federated", tags=["联邦学习"])

# TODO: 客户端注册接口
# TODO: 模型聚合接口
# TODO: 训练状态查询接口
