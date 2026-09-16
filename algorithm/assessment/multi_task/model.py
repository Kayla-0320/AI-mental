"""
多任务风险分级模型 —— 共享编码器 + 三任务头

架构：
    输入特征 (感知层输出) → 共享编码器 (2层MLP) → 共享语义特征
                                                        ↓
                                    ┌──────────────────┼──────────────────┐
                                    ↓                  ↓                  ↓
                              抑郁任务头          焦虑任务头          睡眠任务头
                            (2层MLP+sigmoid)   (2层MLP+sigmoid)   (2层MLP+sigmoid)

使用纯 numpy 实现，保存为 joblib 文件。
"""
from __future__ import annotations

import numpy as np

from assessment.multi_task.config import FEATURE_DIM, HIDDEN_DIM, SEED, TASK_HIDDEN_DIM, TASK_NAMES


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _init_weights(rows: int, cols: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Xavier 初始化"""
    scale = np.sqrt(2.0 / (rows + cols))
    W = rng.randn(rows, cols).astype(np.float64) * scale
    b = np.zeros(cols, dtype=np.float64)
    return W, b


class TaskHead:
    """单任务头 —— 2层 MLP + sigmoid

    Args:
        input_dim: 输入维度（共享编码器输出维度）
        hidden_dim: 隐藏层维度
    """

    def __init__(self, input_dim: int = HIDDEN_DIM, hidden_dim: int = TASK_HIDDEN_DIM) -> None:
        rng = np.random.RandomState(SEED)
        self.W1, self.b1 = _init_weights(input_dim, hidden_dim, rng)
        self.W2, self.b2 = _init_weights(hidden_dim, 1, rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播 → sigmoid 概率"""
        h = _relu(x @ self.W1 + self.b1)
        logit = (h @ self.W2 + self.b2).ravel()
        return _sigmoid(logit)

    def get_params(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, params: dict) -> None:
        self.W1, self.b1 = params["W1"], params["b1"]
        self.W2, self.b2 = params["W2"], params["b2"]


class SharedEncoder:
    """共享语义特征编码器 —— 2层 MLP

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
    """

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM) -> None:
        rng = np.random.RandomState(SEED)
        self.W1, self.b1 = _init_weights(input_dim, hidden_dim, rng)
        self.W2, self.b2 = _init_weights(hidden_dim, hidden_dim, rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """编码 → ReLU 激活的语义特征"""
        h1 = _relu(x @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        return h2

    def get_params(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, params: dict) -> None:
        self.W1, self.b1 = params["W1"], params["b1"]
        self.W2, self.b2 = params["W2"], params["b2"]


class MultiTaskModel:
    """多任务风险分级模型

    共享编码器 + 三个独立任务头。

    Args:
        feature_dim: 输入特征维度
        hidden_dim: 共享编码器隐藏层维度
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        self.encoder = SharedEncoder(feature_dim, hidden_dim)
        self.heads = {task: TaskHead(hidden_dim) for task in TASK_NAMES}

    def forward(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """前向传播

        Args:
            X: 输入特征 (n_samples, feature_dim)

        Returns:
            {task_name: probabilities (n_samples,)}
        """
        shared_features = self.encoder.forward(X)
        return {task: head.forward(shared_features) for task, head in self.heads.items()}

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """预测概率（同 forward）"""
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> dict[str, np.ndarray]:
        """预测二分类标签"""
        proba = self.forward(X)
        return {task: (p >= threshold).astype(int) for task, p in proba.items()}

    def get_shared_features(self, X: np.ndarray) -> np.ndarray:
        """提取共享语义特征（供下游模块使用）"""
        return self.encoder.forward(X)

    def save(self, path: str) -> None:
        """保存模型为 joblib"""
        import joblib
        state = {
            "encoder": self.encoder.get_params(),
            "heads": {task: head.get_params() for task, head in self.heads.items()},
            "feature_dim": self.encoder.W1.shape[0],
            "hidden_dim": self.encoder.W1.shape[1],
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "MultiTaskModel":
        """从 joblib 加载模型"""
        import joblib
        state = joblib.load(path)
        model = cls(
            feature_dim=state["feature_dim"],
            hidden_dim=state["hidden_dim"],
        )
        model.encoder.set_params(state["encoder"])
        for task in TASK_NAMES:
            model.heads[task].set_params(state["heads"][task])
        return model


class SingleTaskModel:
    """单任务基线模型 —— 独立编码器 + 单任务头

    用于对比实验，每个任务有完全独立的参数。
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        task_name: str = "depression",
    ) -> None:
        self.task_name = task_name
        self.encoder = SharedEncoder(feature_dim, hidden_dim)
        self.head = TaskHead(hidden_dim)

    def forward(self, X: np.ndarray) -> np.ndarray:
        features = self.encoder.forward(X)
        return self.head.forward(features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.forward(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        import joblib
        state = {
            "encoder": self.encoder.get_params(),
            "head": self.head.get_params(),
            "feature_dim": self.encoder.W1.shape[0],
            "hidden_dim": self.encoder.W1.shape[1],
            "task_name": self.task_name,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "SingleTaskModel":
        import joblib
        state = joblib.load(path)
        model = cls(
            feature_dim=state["feature_dim"],
            hidden_dim=state["hidden_dim"],
            task_name=state["task_name"],
        )
        model.encoder.set_params(state["encoder"])
        model.head.set_params(state["head"])
        return model
