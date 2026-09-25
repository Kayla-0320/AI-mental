"""
联邦学习路由 —— 分布式训练 API 端点

暴露底层 federated/server.py + federated/client.py 的功能：
- GET  /federated/status     → 服务端状态
- POST /federated/aggregate  → FedAvg 聚合 / 仿真
- POST /federated/register   → 客户端注册
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, field
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from federated.server import (
    FederatedServer,
    ServerConfig,
    SimulationResult,
    create_simulation_clients,
)
from federated.client import FederatedClient, ClientData, generate_client_data

router = APIRouter(prefix="/federated", tags=["联邦学习"])

# 内存状态存储（演示用）
_server_state: dict = {
    "initialized": False,
    "config": None,
    "clients": [],
    "global_test_data": None,
    "simulation_result": None,
}
_registered_clients: dict[str, dict] = {}


# ============================================================
# 请求 / 响应模型
# ============================================================

class ClientRegisterRequest(BaseModel):
    client_id: str = Field(..., description="客户端 ID")
    n_samples: int = Field(500, description="样本数量")
    n_features: int = Field(10, description="特征数量")


class ClientRegisterResponse(BaseModel):
    client_id: str
    status: str
    n_samples: int
    n_features: int


class AggregationRequest(BaseModel):
    num_rounds: int = Field(5, description="训练轮次")
    n_samples_per_client: int = Field(500, description="每客户端样本数")


class RoundMetricsResponse(BaseModel):
    round_num: int
    global_accuracy: float
    global_f1: float
    client_metrics: list[dict]


class AggregationResponse(BaseModel):
    status: str
    num_rounds: int
    final_accuracy: float
    final_f1: float
    round_results: list[RoundMetricsResponse]


class StatusResponse(BaseModel):
    initialized: bool
    num_clients: int
    client_ids: list[str]
    has_global_model: bool
    config: Optional[dict] = None


# ============================================================
# 端点
# ============================================================

@router.get("/status", response_model=StatusResponse)
async def federated_status():
    """返回联邦学习服务器状态"""
    return StatusResponse(
        initialized=_server_state["initialized"],
        num_clients=len(_server_state.get("clients", [])),
        client_ids=[c.client_id for c in _server_state.get("clients", [])],
        has_global_model=_server_state.get("simulation_result") is not None,
        config=(
            asdict(_server_state["config"])
            if _server_state.get("config")
            else None
        ),
    )


@router.post("/aggregate", response_model=AggregationResponse)
async def federated_aggregate(request: AggregationRequest):
    """触发 FedAvg 聚合（运行完整仿真）

    创建模拟客户端，运行多轮 FedAvg 训练，返回全局模型指标。
    """
    try:
        config = ServerConfig(
            num_rounds=request.num_rounds,
            fraction_fit=1.0,
        )
        server = FederatedServer(config)

        clients, global_test_data = create_simulation_clients(
            n_samples_per_client=request.n_samples_per_client,
        )

        result: SimulationResult = server.run_simulation(clients, global_test_data)

        # 更新状态
        _server_state["initialized"] = True
        _server_state["config"] = config
        _server_state["clients"] = clients
        _server_state["global_test_data"] = global_test_data
        _server_state["simulation_result"] = result

        # 组装响应
        round_results = []
        for rr in result.round_results:
            round_results.append(
                RoundMetricsResponse(
                    round_num=rr.round_num,
                    global_accuracy=rr.global_metrics["accuracy"],
                    global_f1=rr.global_metrics["f1"],
                    client_metrics=rr.client_metrics,
                )
            )

        final_acc = result.round_results[-1].global_metrics["accuracy"] if result.round_results else 0.0
        final_f1 = result.round_results[-1].global_metrics["f1"] if result.round_results else 0.0

        return AggregationResponse(
            status="completed",
            num_rounds=request.num_rounds,
            final_accuracy=final_acc,
            final_f1=final_f1,
            round_results=round_results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚合失败: {str(e)}")


@router.post("/register", response_model=ClientRegisterResponse)
async def federated_register(request: ClientRegisterRequest):
    """注册联邦学习客户端"""
    if request.client_id in _registered_clients:
        raise HTTPException(
            status_code=409,
            detail=f"客户端 {request.client_id} 已注册",
        )

    try:
        data: ClientData = generate_client_data(
            client_id=request.client_id,
            n_samples=request.n_samples,
            n_features=request.n_features,
        )
        client = FederatedClient(request.client_id, data)

        _registered_clients[request.client_id] = {
            "client": client,
            "data": data,
        }

        return ClientRegisterResponse(
            client_id=request.client_id,
            status="registered",
            n_samples=request.n_samples,
            n_features=request.n_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")
