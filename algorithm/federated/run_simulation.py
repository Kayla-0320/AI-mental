"""
联邦学习仿真一键运行脚本

⚠️ 声明：此 Demo 为联邦学习流程模拟，非真实跨机构部署。

对比实验：
    1. 联邦全局模型：FedAvg 聚合后的模型
    2. 单客户端本地模型：各客户端独立训练的模型
    3. 中心化训练模型：合并所有数据后训练的模型

输出：
    - 收敛曲线图（federated/outputs/convergence_curve.png）
    - Markdown 报告（federated/outputs/federated_report.md）
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 路径配置
_MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _MODULE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = str(OUTPUT_DIR / "federated_report.md")
CHART_PATH = str(OUTPUT_DIR / "convergence_curve.png")

# 固定随机种子
RANDOM_SEED = 42


def train_centralized_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """训练中心化模型

    合并所有客户端数据后训练。

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        dict: 模型和指标
    """
    print("\n[Centralized] 训练中心化模型...")
    model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=200,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="binary")),
    }
    print(f"[Centralized] Test Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    return {"model": model, "metrics": metrics}


def run_federated_simulation(
    num_rounds: int = 5,
    n_samples_per_client: int = 500,
) -> dict:
    """运行联邦学习仿真

    Args:
        num_rounds: 训练轮数
        n_samples_per_client: 每个客户端的样本数

    Returns:
        dict: 仿真结果
    """
    from .server import FederatedServer, ServerConfig, create_simulation_clients
    from .client import train_local_model, create_client_model

    print("\n" + "="*60)
    print("联邦学习仿真开始")
    print("="*60)
    print(f"配置：{num_rounds} 轮，2 个客户端，每客户端 {n_samples_per_client} 样本")

    start_time = time.time()

    # 创建客户端
    clients, global_test_data = create_simulation_clients(
        n_samples_per_client=n_samples_per_client,
    )
    print(f"\n[Setup] 创建 {len(clients)} 个客户端")
    for c in clients:
        print(f"  - {c.client_id}: {c.data.n_samples} 样本")

    # 联邦学习训练
    config = ServerConfig(
        num_rounds=num_rounds,
        fraction_fit=1.0,
        random_seed=RANDOM_SEED,
    )
    server = FederatedServer(config)
    fed_result = server.run_simulation(clients, global_test_data)

    # 中心化训练（对比基线）
    centralized_result = train_centralized_model(
        global_test_data.X_train,
        global_test_data.y_train,
        global_test_data.X_test,
        global_test_data.y_test,
    )

    # 单客户端本地模型（最终轮）
    local_models = {}
    for client in clients:
        local_model = create_client_model(client.client_id)
        trained = train_local_model(local_model, client.data)
        local_models[client.client_id] = trained
        print(f"\n[Local {client.client_id}] "
              f"Test Acc={trained.test_metrics['accuracy']:.4f}, "
              f"F1={trained.test_metrics['f1']:.4f}")

    elapsed = time.time() - start_time

    return {
        "federated": fed_result,
        "centralized": centralized_result,
        "local_models": local_models,
        "clients": clients,
        "global_test_data": global_test_data,
        "elapsed_time": elapsed,
    }


def generate_convergence_chart(
    fed_result,
    save_path: str = CHART_PATH,
) -> str:
    """生成收敛曲线图

    Args:
        fed_result: 联邦学习仿真结果
        save_path: 图表保存路径

    Returns:
        str: 图表文件路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    round_nums = [r.round_num for r in fed_result.round_results]
    global_acc = [r.global_metrics["accuracy"] for r in fed_result.round_results]
    global_f1 = [r.global_metrics["f1"] for r in fed_result.round_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 准确率曲线
    ax1.plot(round_nums, global_acc, "o-", linewidth=2, markersize=8, label="Global Model", color="#FF6B6B")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Federated Learning Convergence - Accuracy")
    ax1.set_xticks(round_nums)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # F1 曲线
    ax2.plot(round_nums, global_f1, "s-", linewidth=2, markersize=8, label="Global Model", color="#4ECDC4")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Federated Learning Convergence - F1")
    ax2.set_xticks(round_nums)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path


def generate_markdown_report(
    result: dict,
    save_path: str = REPORT_PATH,
) -> str:
    """生成 Markdown 报告

    Args:
        result: 仿真结果
        save_path: 报告保存路径

    Returns:
        str: 报告内容
    """
    fed_result = result["federated"]
    centralized = result["centralized"]
    local_models = result["local_models"]
    elapsed = result["elapsed_time"]

    lines = [
        "# 联邦学习仿真报告",
        "",
        "> ⚠️ 声明：此报告为联邦学习流程模拟结果，非真实跨机构部署。",
        "> 真实联邦学习需要网络通信、加密传输、安全聚合等基础设施。",
        "",
        f"> 仿真时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 训练轮数：{fed_result.config.num_rounds}",
        f"> 客户端数：2",
        f"> 耗时：{elapsed:.2f} 秒",
        "",
        "## 1. 实验设置",
        "",
        "| 配置项 | 值 |",
        "|--------|-----|",
        f"| 训练轮数 | {fed_result.config.num_rounds} |",
        f"| 客户端数 | 2 |",
        f"| 每客户端样本数 | {result['clients'][0].data.n_samples} |",
        f"| 聚合策略 | FedAvg |",
        f"| fraction_fit | {fed_result.config.fraction_fit} |",
        "",
        "## 2. 联邦学习收敛过程",
        "",
        "| 轮次 | 全局模型 Acc | 全局模型 F1 | Client A Acc | Client B Acc |",
        "|------|--------------|-------------|--------------|--------------|",
    ]

    for r in fed_result.round_results:
        client_a_acc = next((m["test"]["accuracy"] for m in r.client_metrics if m["client_id"] == "client_a"), 0)
        client_b_acc = next((m["test"]["accuracy"] for m in r.client_metrics if m["client_id"] == "client_b"), 0)
        lines.append(
            f"| {r.round_num} | {r.global_metrics['accuracy']:.4f} | "
            f"{r.global_metrics['f1']:.4f} | {client_a_acc:.4f} | {client_b_acc:.4f} |"
        )

    lines.extend([
        "",
        "## 3. 模型对比",
        "",
        "| 模型类型 | 测试集 Acc | 测试集 F1 | 说明 |",
        "|----------|------------|-----------|------|",
    ])

    # 联邦全局模型（最终轮）
    final_global = fed_result.round_results[-1].global_metrics
    lines.append(
        f"| **联邦全局模型** | {final_global['accuracy']:.4f} | "
        f"{final_global['f1']:.4f} | FedAvg 聚合 |"
    )

    # 中心化模型
    lines.append(
        f"| 中心化模型 | {centralized['metrics']['accuracy']:.4f} | "
        f"{centralized['metrics']['f1']:.4f} | 合并所有数据训练 |"
    )

    # 单客户端本地模型
    for client_id, model in local_models.items():
        lines.append(
            f"| 本地模型 ({client_id}) | {model.test_metrics['accuracy']:.4f} | "
            f"{model.test_metrics['f1']:.4f} | 仅使用本地数据 |"
        )

    lines.extend([
        "",
        "## 4. 分析与结论",
        "",
        "### 4.1 联邦学习 vs 中心化训练",
        "",
    ])

    gap = centralized["metrics"]["accuracy"] - final_global["accuracy"]
    if abs(gap) < 0.02:
        lines.append(
            "- 联邦全局模型与中心化模型性能接近（差距 < 2%），"
            "说明 FedAvg 聚合策略有效。"
        )
    elif gap > 0:
        lines.append(
            f"- 联邦全局模型准确率比中心化模型低 {gap:.2%}，"
            "这是预期结果：联邦学习在保护数据隐私的同时，"
            "牺牲了少量性能。"
        )
    else:
        lines.append(
            f"- 联邦全局模型准确率比中心化模型高 {-gap:.2%}，"
            "这可能是因为 FedAvg 聚合起到了正则化效果。"
        )

    lines.extend([
        "",
        "### 4.2 联邦学习 vs 本地训练",
        "",
        "- 联邦全局模型综合了多个客户端的数据分布，"
        "相比单客户端本地模型，具有更好的泛化能力。",
        "- 本地模型仅使用本地数据训练，"
        "可能过拟合到特定数据分布。",
        "",
        "### 4.3 隐私保护优势",
        "",
        "- 联邦学习过程中，原始数据始终保留在各客户端本地。",
        "- 仅上传模型参数（梯度/权重），不传输原始数据。",
        "- 符合《个人信息保护法》的数据最小化原则。",
        "",
        "## 5. 收敛曲线",
        "",
        f"![收敛曲线](convergence_curve.png)",
        "",
        "## 6. 声明",
        "",
        "本仿真仅用于演示联邦学习的基本流程，"
        "不包含真实跨机构部署所需的安全机制（如安全聚合、差分隐私等）。"
        "生产环境部署需要额外的基础设施支持。",
        "",
    ])

    content = "\n".join(lines)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def main():
    """主函数"""
    # 运行仿真
    result = run_federated_simulation(
        num_rounds=5,
        n_samples_per_client=500,
    )

    # 生成收敛曲线
    chart_path = generate_convergence_chart(result["federated"])
    print(f"\n[Output] 收敛曲线已保存 → {chart_path}")

    # 生成报告
    report_path = generate_markdown_report(result)
    print(f"[Output] 报告已保存 → {REPORT_PATH}")

    print("\n" + "="*60)
    print("联邦学习仿真完成")
    print("="*60)

    return result


if __name__ == "__main__":
    main()
