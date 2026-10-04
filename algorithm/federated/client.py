"""
联邦学习客户端 —— 2 个客户端各用 sklearn LogisticRegression 训练

⚠️ 声明：此 Demo 为联邦学习流程模拟，非真实跨机构部署。

数据分布：
    - 客户端 A：城市青少年群体（特征分布 1）
    - 客户端 B：县镇青少年群体（特征分布 2）

模型：
    - sklearn LogisticRegression
    - 每轮本地训练后上传模型参数
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# 固定随机种子
RANDOM_SEED = 42


@dataclass
class ClientData:
    """客户端数据"""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    client_id: str
    n_samples: int


@dataclass
class ClientModel:
    """客户端模型"""
    client_id: str
    model: LogisticRegression
    train_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)


def generate_client_data(
    client_id: str,
    n_samples: int = 500,
    n_features: int = 10,
    random_state: int = RANDOM_SEED,
) -> ClientData:
    """为客户端生成模拟数据

    不同客户端使用不同的数据分布参数，模拟真实场景中
    不同机构的非独立同分布（Non-IID）数据。

    Args:
        client_id: 客户端 ID（"client_a" 或 "client_b"）
        n_samples: 样本数量
        n_features: 特征数量
        random_state: 随机种子

    Returns:
        ClientData: 客户端数据
    """
    # 不同客户端使用不同的分布参数
    if client_id == "client_a":
        # 客户端 A：城市群体，特征分布更集中
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=7,
            n_redundant=2,
            n_clusters_per_class=2,
            flip_y=0.05,  # 噪声较少
            class_sep=1.2,
            random_state=random_state,
        )
    else:
        # 客户端 B：县镇群体，特征分布更分散
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=5,
            n_redundant=3,
            n_clusters_per_class=3,
            flip_y=0.1,  # 噪声较多
            class_sep=0.8,
            random_state=random_state + 1,  # 不同种子
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    return ClientData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        client_id=client_id,
        n_samples=n_samples,
    )


def create_client_model(client_id: str) -> LogisticRegression:
    """创建客户端模型

    Args:
        client_id: 客户端 ID

    Returns:
        LogisticRegression: sklearn 模型
    """
    return LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=200,
        solver="lbfgs",
    )


def train_local_model(
    model: LogisticRegression,
    data: ClientData,
) -> ClientModel:
    """本地训练模型

    Args:
        model: sklearn 模型
        data: 客户端数据

    Returns:
        ClientModel: 训练后的客户端模型
    """
    model.fit(data.X_train, data.y_train)

    # 训练集指标
    train_pred = model.predict(data.X_train)
    train_metrics = {
        "accuracy": float(accuracy_score(data.y_train, train_pred)),
        "f1": float(f1_score(data.y_train, train_pred, average="binary")),
    }

    # 测试集指标
    test_pred = model.predict(data.X_test)
    test_metrics = {
        "accuracy": float(accuracy_score(data.y_test, test_pred)),
        "f1": float(f1_score(data.y_test, test_pred, average="binary")),
    }

    return ClientModel(
        client_id=data.client_id,
        model=model,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )


def get_model_parameters(model: LogisticRegression) -> dict:
    """提取模型参数

    用于上传到服务器进行聚合。

    Args:
        model: sklearn LogisticRegression 模型

    Returns:
        dict: 模型参数字典
    """
    return {
        "coef": model.coef_.copy(),
        "intercept": model.intercept_.copy(),
        "classes": model.classes_.copy(),
    }


def set_model_parameters(
    model: LogisticRegression,
    params: dict,
) -> LogisticRegression:
    """设置模型参数

    用于从服务器下载聚合后的参数。

    Args:
        model: sklearn LogisticRegression 模型
        params: 模型参数字典

    Returns:
        LogisticRegression: 参数已更新的模型
    """
    model.coef_ = params["coef"].copy()
    model.intercept_ = params["intercept"].copy()
    model.classes_ = params["classes"].copy()
    return model


class FederatedClient:
    """联邦学习客户端

    模拟 Flower 框架中的 Client 角色。

    ⚠️ 此 Demo 为联邦学习流程模拟，非真实跨机构部署。
    """

    def __init__(self, client_id: str, data: ClientData):
        """初始化客户端

        Args:
            client_id: 客户端 ID
            data: 客户端数据
        """
        self.client_id = client_id
        self.data = data
        self.model = create_client_model(client_id)
        self.round_metrics = []

    def get_parameters(self) -> dict:
        """获取模型参数（用于上传）"""
        return get_model_parameters(self.model)

    def set_parameters(self, params: dict) -> None:
        """设置模型参数（从服务器下载）"""
        self.model = set_model_parameters(self.model, params)

    def fit(self) -> ClientModel:
        """本地训练

        Returns:
            ClientModel: 训练后的模型
        """
        client_model = train_local_model(self.model, self.data)
        self.round_metrics.append({
            "train": client_model.train_metrics,
            "test": client_model.test_metrics,
        })
        return client_model

    def evaluate(self) -> dict:
        """评估模型

        Returns:
            dict: 测试集指标
        """
        y_pred = self.model.predict(self.data.X_test)
        return {
            "accuracy": float(accuracy_score(self.data.y_test, y_pred)),
            "f1": float(f1_score(self.data.y_test, y_pred, average="binary")),
        }
