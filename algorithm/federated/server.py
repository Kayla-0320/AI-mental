"""
联邦学习服务端 —— FedAvg 聚合策略

⚠️ 声明：此 Demo 为联邦学习流程模拟，非真实跨机构部署。

聚合策略：
    - FedAvg (Federated Averaging)
    - 按客户端样本量加权平均模型参数

训练配置：
    - 5 轮训练
    - fraction_fit = 1.0（每轮选择所有客户端）
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .client import FederatedClient, ClientData, generate_client_data


# 固定随机种子
RANDOM_SEED = 42


@dataclass
class ServerConfig:
    """服务端配置"""
    num_rounds: int = 5
    fraction_fit: float = 1.0
    min_available_clients: int = 2
    random_seed: int = RANDOM_SEED


@dataclass
class RoundResult:
    """单轮训练结果"""
    round_num: int
    global_metrics: dict
    client_metrics: list[dict]


@dataclass
class SimulationResult:
    """仿真结果"""
    round_results: list[RoundResult] = field(default_factory=list)
    final_global_model: Optional[LogisticRegression] = None
    global_test_data: Optional[ClientData] = None
    config: Optional[ServerConfig] = None


def fed_avg_aggregate(
    client_params: list[dict],
    client_sizes: list[int],
) -> dict:
    """FedAvg 聚合策略

    按客户端样本量加权平均模型参数。

    Args:
        client_params: 各客户端模型参数列表
        client_sizes: 各客户端样本量列表

    Returns:
        dict: 聚合后的模型参数
    """
    total_size = sum(client_sizes)
    weights = [n / total_size for n in client_sizes]

    # 加权平均 coef
    avg_coef = np.zeros_like(client_params[0]["coef"])
    for w, params in zip(weights, client_params):
        avg_coef += w * params["coef"]

    # 加权平均 intercept
    avg_intercept = np.zeros_like(client_params[0]["intercept"])
    for w, params in zip(weights, client_params):
        avg_intercept += w * params["intercept"]

    return {
        "coef": avg_coef,
        "intercept": avg_intercept,
        "classes": client_params[0]["classes"],  # 类别标签相同
    }


class FederatedServer:
    """联邦学习服务端

    模拟 Flower 框架中的 Server 角色。

    ⚠️ 此 Demo 为联邦学习流程模拟，非真实跨机构部署。
    """

    def __init__(self, config: ServerConfig):
        """初始化服务端

        Args:
            config: 服务端配置
        """
        self.config = config
        self.global_model: Optional[LogisticRegression] = None
        self.round_results: list[RoundResult] = []

    def initialize_global_model(self) -> None:
        """初始化全局模型

        使用第一个客户端的参数初始化。
        """
        self.global_model = LogisticRegression(
            random_state=self.config.random_seed,
            max_iter=200,
            solver="lbfgs",
        )

    def configure_round(self, round_num: int) -> list[int]:
        """配置本轮训练

        选择参与训练的客户端。

        Args:
            round_num: 轮次编号

        Returns:
            list[int]: 参与训练的客户端索引
        """
        # fraction_fit = 1.0 表示选择所有客户端
        num_clients = 2  # 固定 2 个客户端
        num_sample = max(1, int(num_clients * self.config.fraction_fit))
        return list(range(num_sample))

    def aggregate(self, client_params: list[dict], client_sizes: list[int]) -> dict:
        """聚合客户端参数

        Args:
            client_params: 客户端参数列表
            client_sizes: 客户端样本量列表

        Returns:
            dict: 聚合后的参数
        """
        return fed_avg_aggregate(client_params, client_sizes)

    def evaluate_global_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """评估全局模型

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            dict: 测试指标
        """
        if self.global_model is None:
            return {"accuracy": 0.0, "f1": 0.0}

        y_pred = self.global_model.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="binary")),
        }

    def run_simulation(
        self,
        clients: list[FederatedClient],
        global_test_data: ClientData,
    ) -> SimulationResult:
        """运行联邦学习仿真

        Args:
            clients: 客户端列表
            global_test_data: 全局测试数据

        Returns:
            SimulationResult: 仿真结果
        """
        self.initialize_global_model()

        result = SimulationResult(
            config=self.config,
            global_test_data=global_test_data,
        )

        for round_num in range(1, self.config.num_rounds + 1):
            print(f"\n{'='*50}")
            print(f"Round {round_num}/{self.config.num_rounds}")
            print(f"{'='*50}")

            # 选择参与客户端
            selected_indices = self.configure_round(round_num)
            print(f"[Server] 选择 {len(selected_indices)} 个客户端参与训练")

            # 客户端本地训练
            client_params = []
            client_sizes = []
            round_client_metrics = []

            for idx in selected_indices:
                client = clients[idx]
                print(f"\n[Client {client.client_id}] 开始本地训练...")

                # 如果有全局模型参数，先下载
                if self.global_model is not None and hasattr(self.global_model, 'coef_'):
                    try:
                        from .client import get_model_parameters
                        global_params = get_model_parameters(self.global_model)
                        client.set_parameters(global_params)
                    except Exception:
                        pass  # 首轮跳过

                # 本地训练
                client_model = client.fit()
                print(f"[Client {client.client_id}] "
                      f"Train Acc={client_model.train_metrics['accuracy']:.4f}, "
                      f"Test Acc={client_model.test_metrics['accuracy']:.4f}")

                # 上传参数
                params = client.get_parameters()
                client_params.append(params)
                client_sizes.append(client.data.n_samples)
                round_client_metrics.append({
                    "client_id": client.client_id,
                    "train": client_model.train_metrics,
                    "test": client_model.test_metrics,
                })

            # FedAvg 聚合
            print(f"\n[Server] FedAvg 聚合 {len(client_params)} 个客户端参数...")
            aggregated_params = self.aggregate(client_params, client_sizes)

            # 更新全局模型
            from .client import set_model_parameters
            self.global_model = set_model_parameters(self.global_model, aggregated_params)

            # 评估全局模型
            global_metrics = self.evaluate_global_model(
                global_test_data.X_test,
                global_test_data.y_test,
            )
            print(f"[Server] 全局模型 Test Acc={global_metrics['accuracy']:.4f}, "
                  f"F1={global_metrics['f1']:.4f}")

            # 记录本轮结果
            round_result = RoundResult(
                round_num=round_num,
                global_metrics=global_metrics,
                client_metrics=round_client_metrics,
            )
            self.round_results.append(round_result)
            result.round_results.append(round_result)

        result.final_global_model = self.global_model
        return result


def create_simulation_clients(
    n_samples_per_client: int = 500,
    n_features: int = 10,
) -> tuple[list[FederatedClient], ClientData]:
    """创建仿真客户端和全局测试数据

    Args:
        n_samples_per_client: 每个客户端的样本数
        n_features: 特征数量

    Returns:
        (clients, global_test_data): 客户端列表和全局测试数据
    """
    # 客户端 A
    data_a = generate_client_data(
        client_id="client_a",
        n_samples=n_samples_per_client,
        n_features=n_features,
    )
    client_a = FederatedClient("client_a", data_a)

    # 客户端 B
    data_b = generate_client_data(
        client_id="client_b",
        n_samples=n_samples_per_client,
        n_features=n_features,
    )
    client_b = FederatedClient("client_b", data_b)

    # 全局测试数据（混合两个客户端的测试集）
    X_global_test = np.vstack([data_a.X_test, data_b.X_test])
    y_global_test = np.concatenate([data_a.y_test, data_b.y_test])

    global_test_data = ClientData(
        X_train=np.vstack([data_a.X_train, data_b.X_train]),
        X_test=X_global_test,
        y_train=np.concatenate([data_a.y_train, data_b.y_train]),
        y_test=y_global_test,
        client_id="global",
        n_samples=len(y_global_test),
    )

    return [client_a, client_b], global_test_data
