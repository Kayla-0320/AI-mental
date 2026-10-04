"""
联邦学习单元测试

覆盖：
- 客户端数据生成
- FedAvg 聚合
- 服务端仿真
- 收敛曲线生成
- Markdown 报告生成
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from federated.client import (
    ClientData,
    FederatedClient,
    generate_client_data,
    create_client_model,
    train_local_model,
    get_model_parameters,
    set_model_parameters,
)
from federated.server import (
    ServerConfig,
    FederatedServer,
    fed_avg_aggregate,
    create_simulation_clients,
)
from federated.run_simulation import (
    train_centralized_model,
    run_federated_simulation,
    generate_convergence_chart,
    generate_markdown_report,
)


# ============================================================
# 客户端测试
# ============================================================

class TestClientData:
    def test_generate_client_a(self):
        """生成客户端 A 数据"""
        data = generate_client_data("client_a", n_samples=100)
        assert data.client_id == "client_a"
        assert data.X_train.shape[0] == 80  # 80% 训练集
        assert data.X_test.shape[0] == 20   # 20% 测试集

    def test_generate_client_b(self):
        """生成客户端 B 数据（不同分布）"""
        data = generate_client_data("client_b", n_samples=100)
        assert data.client_id == "client_b"
        assert data.X_train.shape[0] == 80

    def test_different_distributions(self):
        """两个客户端数据分布不同"""
        data_a = generate_client_data("client_a", n_samples=100)
        data_b = generate_client_data("client_b", n_samples=100)
        # 数据应该不同
        assert not np.allclose(data_a.X_train, data_b.X_train)


class TestClientModel:
    def test_create_model(self):
        """创建客户端模型"""
        model = create_client_model("client_a")
        assert model is not None

    def test_train_model(self):
        """训练本地模型"""
        data = generate_client_data("client_a", n_samples=100)
        model = create_client_model("client_a")
        client_model = train_local_model(model, data)
        assert client_model.train_metrics["accuracy"] > 0
        assert client_model.test_metrics["accuracy"] > 0

    def test_get_set_parameters(self):
        """获取和设置模型参数"""
        data = generate_client_data("client_a", n_samples=100)
        model = create_client_model("client_a")
        trained = train_local_model(model, data)
        params = get_model_parameters(trained.model)
        assert "coef" in params
        assert "intercept" in params

        # 设置参数到新模型
        new_model = create_client_model("client_a")
        set_model_parameters(new_model, params)
        assert np.allclose(new_model.coef_, trained.model.coef_)


class TestFederatedClient:
    def test_client_creation(self):
        """创建联邦客户端"""
        data = generate_client_data("client_a", n_samples=100)
        client = FederatedClient("client_a", data)
        assert client.client_id == "client_a"

    def test_client_fit(self):
        """客户端训练"""
        data = generate_client_data("client_a", n_samples=100)
        client = FederatedClient("client_a", data)
        result = client.fit()
        assert result.train_metrics["accuracy"] > 0
        assert len(client.round_metrics) == 1

    def test_client_evaluate(self):
        """客户端评估"""
        data = generate_client_data("client_a", n_samples=100)
        client = FederatedClient("client_a", data)
        client.fit()
        metrics = client.evaluate()
        assert "accuracy" in metrics
        assert "f1" in metrics


# ============================================================
# 服务端测试
# ============================================================

class TestFedAvg:
    def test_aggregate_two_clients(self):
        """聚合两个客户端参数"""
        params_a = {
            "coef": np.array([[1.0, 2.0]]),
            "intercept": np.array([0.5]),
            "classes": np.array([0, 1]),
        }
        params_b = {
            "coef": np.array([[3.0, 4.0]]),
            "intercept": np.array([1.5]),
            "classes": np.array([0, 1]),
        }
        sizes = [100, 100]

        avg = fed_avg_aggregate([params_a, params_b], sizes)
        # 等权重平均
        expected_coef = np.array([[2.0, 3.0]])
        assert np.allclose(avg["coef"], expected_coef)

    def test_aggregate_weighted(self):
        """加权聚合"""
        params_a = {
            "coef": np.array([[1.0, 0.0]]),
            "intercept": np.array([0.0]),
            "classes": np.array([0, 1]),
        }
        params_b = {
            "coef": np.array([[3.0, 0.0]]),
            "intercept": np.array([0.0]),
            "classes": np.array([0, 1]),
        }
        sizes = [300, 100]  # A 样本更多

        avg = fed_avg_aggregate([params_a, params_b], sizes)
        # 加权：0.75 * 1 + 0.25 * 3 = 1.5
        assert avg["coef"][0, 0] == pytest.approx(1.5)


class TestFederatedServer:
    def test_server_creation(self):
        """创建服务端"""
        config = ServerConfig(num_rounds=3)
        server = FederatedServer(config)
        assert server.config.num_rounds == 3

    def test_server_simulation(self):
        """运行仿真"""
        config = ServerConfig(num_rounds=2, fraction_fit=1.0)
        server = FederatedServer(config)

        clients, global_test = create_simulation_clients(n_samples_per_client=50)
        result = server.run_simulation(clients, global_test)

        assert len(result.round_results) == 2
        assert result.final_global_model is not None


# ============================================================
# 仿真测试
# ============================================================

class TestSimulation:
    def test_centralized_model(self):
        """训练中心化模型"""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randint(0, 2, 20)

        result = train_centralized_model(X_train, y_train, X_test, y_test)
        assert result["metrics"]["accuracy"] > 0

    def test_full_simulation(self):
        """完整仿真流程"""
        result = run_federated_simulation(
            num_rounds=2,
            n_samples_per_client=50,
        )
        assert "federated" in result
        assert "centralized" in result
        assert "local_models" in result


# ============================================================
# 输出测试
# ============================================================

class TestOutput:
    def test_convergence_chart(self, tmp_path):
        """生成收敛曲线"""
        result = run_federated_simulation(num_rounds=2, n_samples_per_client=50)
        chart_path = str(tmp_path / "chart.png")
        path = generate_convergence_chart(result["federated"], chart_path)
        assert Path(path).exists()

    def test_markdown_report(self, tmp_path):
        """生成 Markdown 报告"""
        result = run_federated_simulation(num_rounds=2, n_samples_per_client=50)
        report_path = str(tmp_path / "report.md")
        content = generate_markdown_report(result, report_path)
        assert "# 联邦学习仿真报告" in content
        assert "FedAvg" in content
        assert "声明" in content
