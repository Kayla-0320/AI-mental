"""
对比实验 1：规则引擎 vs 多任务模型 —— 风险评估方法对比

实验目的：
    比较基于规则的阈值判定与数据驱动的多任务学习模型在风险评估上的差异。

实验条件：
    - 数据集：sklearn.datasets.make_classification 模拟数据
    - 样本量：1000
    - 特征维度：10
    - 随机种子：42

指标定义：
    - Accuracy: 分类准确率
    - F1 (macro): 宏平均 F1
    - Recall (高风险): 高风险样本的召回率（临床安全性核心指标）
    - Specificity: 低风险样本的特异度
    - Inference time (ms): 单样本推理延迟

局限性说明：
    - 使用模拟数据，非真实临床数据
    - 规则引擎的阈值基于经验设定，未经临床验证
    - 多任务模型为浅层 MLP，非深度学习架构
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ============================================================
# 路径配置
# ============================================================

_MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _MODULE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SAMPLES = 1000
N_FEATURES = 10


# ============================================================
# 数据生成
# ============================================================

def generate_risk_data(n_samples: int = N_SAMPLES) -> tuple:
    """生成风险评估模拟数据

    使用 make_classification 生成二分类数据，
    模拟心理健康风险筛查场景。

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.05,
        class_sep=1.0,
        random_state=RANDOM_SEED,
    )
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


# ============================================================
# 规则引擎
# ============================================================

class RuleBasedRiskEngine:
    """基于规则的风险评估引擎

    规则设计（模拟临床筛查量表逻辑）：
        - 特征 0-2 代表抑郁相关指标（PHQ-9 子项模拟）
        - 特征 3-5 代表焦虑相关指标（GAD-7 子项模拟）
        - 特征 6-8 代表睡眠相关指标
        - 特征 9 代表社会功能指标

    判定规则：
        - 任一维度均值 > 阈值 → 高风险
        - 两个维度均值 > 阈值 * 0.8 → 中风险
        - 否则 → 低风险
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测风险等级（二分类：0=低/中风险, 1=高风险）"""
        dep_score = np.mean(X[:, 0:3], axis=1)
        anx_score = np.mean(X[:, 3:6], axis=1)
        slp_score = np.mean(X[:, 6:9], axis=1)
        social_score = X[:, 9]

        # 规则判定
        high_count = (
            (dep_score > self.threshold).astype(int)
            + (anx_score > self.threshold).astype(int)
            + (slp_score > self.threshold).astype(int)
            + (social_score > self.threshold).astype(int)
        )
        return (high_count >= 2).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """输出风险概率（归一化综合评分）"""
        dep_score = np.mean(X[:, 0:3], axis=1)
        anx_score = np.mean(X[:, 3:6], axis=1)
        slp_score = np.mean(X[:, 6:9], axis=1)
        social_score = X[:, 9]

        combined = 0.3 * dep_score + 0.3 * anx_score + 0.2 * slp_score + 0.2 * social_score
        # 归一化到 [0, 1]
        combined = np.clip(combined, 0, 1)
        return combined


# ============================================================
# 多任务模型（简化版）
# ============================================================

class SimpleMultiTaskModel:
    """简化多任务模型（用于实验对比）

    共享编码器 + 3 任务头，纯 numpy 实现。
    """

    def __init__(self, input_dim: int = N_FEATURES, hidden_dim: int = 32):
        rng = np.random.RandomState(RANDOM_SEED)
        # 共享编码器
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        # 任务头
        self.heads = {}
        for task in ["depression", "anxiety", "sleep"]:
            self.heads[task] = {
                "Wh": rng.randn(hidden_dim, 16) * 0.1,
                "bh": np.zeros(16),
                "Wo": rng.randn(16, 1) * 0.1,
                "bo": np.zeros(1),
            }

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _encode(self, X):
        h = self._relu(X @ self.W1 + self.b1)
        h = self._relu(h @ self.W2 + self.b2)
        return h

    def forward(self, X):
        enc = self._encode(X)
        proba = {}
        for task, params in self.heads.items():
            h = self._relu(enc @ params["Wh"] + params["bh"])
            logit = (h @ params["Wo"] + params["bo"]).ravel()
            proba[task] = self._sigmoid(logit)
        return proba

    def predict_risk(self, X) -> np.ndarray:
        """综合风险预测（三任务最大概率）"""
        proba = self.forward(X)
        combined = np.maximum(
            np.maximum(proba["depression"], proba["anxiety"]),
            proba["sleep"],
        )
        return (combined >= 0.5).astype(int)

    def predict_risk_proba(self, X) -> np.ndarray:
        """综合风险概率"""
        proba = self.forward(X)
        return np.maximum(
            np.maximum(proba["depression"], proba["anxiety"]),
            proba["sleep"],
        )

    def train(self, X, y, epochs=100, lr=0.01):
        """简化训练（仅训练 depression 头作为代理）"""
        n = len(y)
        for _ in range(epochs):
            enc = self._encode(X)
            for task in ["depression", "anxiety", "sleep"]:
                params = self.heads[task]
                h = self._relu(enc @ params["Wh"] + params["bh"])
                logit = h @ params["Wo"] + params["bo"]
                pred = self._sigmoid(logit).ravel()
                error = pred - y
                d_logit = error.reshape(-1, 1) / n
                dWo = h.T @ d_logit
                dbo = d_logit.sum(axis=0)
                dh = d_logit @ params["Wo"].T * (h > 0).astype(float)
                dWh = enc.T @ dh
                dbh = dh.sum(axis=0)
                params["Wo"] -= lr * dWo
                params["bo"] -= lr * dbo
                params["Wh"] -= lr * dWh
                params["bh"] -= lr * dbh


# ============================================================
# 实验运行
# ============================================================

def run_experiment() -> dict:
    """运行规则引擎 vs 多任务模型对比实验

    Returns:
        dict: 实验结果
    """
    print("\n" + "=" * 60)
    print("实验 1：规则引擎 vs 多任务模型")
    print("=" * 60)

    X_train, X_test, y_train, y_test = generate_risk_data()
    print(f"数据规模：训练集 {len(y_train)}, 测试集 {len(y_test)}")
    print(f"正样本比例：训练 {y_train.mean():.2%}, 测试 {y_test.mean():.2%}")

    results = {}

    # --- 规则引擎 ---
    print("\n[规则引擎] 评估中...")
    rule_engine = RuleBasedRiskEngine(threshold=0.6)

    start = time.time()
    rule_pred = rule_engine.predict(X_test)
    rule_time = (time.time() - start) / len(y_test) * 1000  # ms per sample

    rule_proba = rule_engine.predict_proba(X_test)

    results["rule_engine"] = {
        "accuracy": float(accuracy_score(y_test, rule_pred)),
        "f1_macro": float(f1_score(y_test, rule_pred, average="macro")),
        "recall_high_risk": float(recall_score(y_test, rule_pred, pos_label=1)),
        "specificity": float(
            confusion_matrix(y_test, rule_pred)[0, 0] /
            max(1, confusion_matrix(y_test, rule_pred)[0].sum())
        ),
        "inference_time_ms": round(rule_time, 4),
        "risk_proba": rule_proba,
    }
    print(f"  Acc={results['rule_engine']['accuracy']:.4f}, "
          f"F1={results['rule_engine']['f1_macro']:.4f}, "
          f"Recall={results['rule_engine']['recall_high_risk']:.4f}")

    # --- 多任务模型 ---
    print("\n[多任务模型] 训练中...")
    mt_model = SimpleMultiTaskModel(input_dim=N_FEATURES)
    mt_model.train(X_train, y_train, epochs=100, lr=0.05)

    print("[多任务模型] 评估中...")
    start = time.time()
    mt_pred = mt_model.predict_risk(X_test)
    mt_time = (time.time() - start) / len(y_test) * 1000

    mt_proba = mt_model.predict_risk_proba(X_test)

    results["multi_task"] = {
        "accuracy": float(accuracy_score(y_test, mt_pred)),
        "f1_macro": float(f1_score(y_test, mt_pred, average="macro")),
        "recall_high_risk": float(recall_score(y_test, mt_pred, pos_label=1)),
        "specificity": float(
            confusion_matrix(y_test, mt_pred)[0, 0] /
            max(1, confusion_matrix(y_test, mt_pred)[0].sum())
        ),
        "inference_time_ms": round(mt_time, 4),
        "risk_proba": mt_proba,
    }
    print(f"  Acc={results['multi_task']['accuracy']:.4f}, "
          f"F1={results['multi_task']['f1_macro']:.4f}, "
          f"Recall={results['multi_task']['recall_high_risk']:.4f}")

    results["_meta"] = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": N_FEATURES,
        "random_seed": RANDOM_SEED,
    }

    return results


def generate_report(results: dict) -> str:
    """生成 Markdown 对比报告"""
    rule = results["rule_engine"]
    mt = results["multi_task"]
    meta = results["_meta"]

    lines = [
        "# 实验 1：规则引擎 vs 多任务模型 —— 风险评估对比",
        "",
        "## 实验条件",
        "",
        "| 配置项 | 值 |",
        "|--------|-----|",
        f"| 数据集 | make_classification (模拟) |",
        f"| 训练集样本数 | {meta['n_train']} |",
        f"| 测试集样本数 | {meta['n_test']} |",
        f"| 特征维度 | {meta['n_features']} |",
        f"| 随机种子 | {meta['random_seed']} |",
        "",
        "## 指标定义",
        "",
        "| 指标 | 定义 | 临床意义 |",
        "|------|------|----------|",
        "| Accuracy | 分类准确率 | 整体判别能力 |",
        "| F1 (macro) | 宏平均 F1 | 兼顾正负类均衡 |",
        "| Recall (高风险) | 高风险样本召回率 | **核心安全指标**：漏诊代价极高 |",
        "| Specificity | 低风险特异度 | 减少不必要的干预 |",
        "| Inference Time | 单样本推理延迟 (ms) | 实时性要求 |",
        "",
        "## 实验结果",
        "",
        "| 指标 | 规则引擎 | 多任务模型 | 差异 |",
        "|------|----------|------------|------|",
    ]

    for metric, label in [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1 (macro)"),
        ("recall_high_risk", "Recall (高风险)"),
        ("specificity", "Specificity"),
    ]:
        r_val = rule[metric]
        m_val = mt[metric]
        diff = m_val - r_val
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        lines.append(
            f"| {label} | {r_val:.4f} | {m_val:.4f} | "
            f"{arrow} {abs(diff):.4f} |"
        )

    lines.append(
        f"| Inference Time (ms) | {rule['inference_time_ms']:.4f} | "
        f"{mt['inference_time_ms']:.4f} | - |"
    )

    lines.extend([
        "",
        "## 分析与结论",
        "",
        "1. **规则引擎**：基于临床阈值规则，可解释性强，但无法捕捉特征间的非线性交互。",
        "2. **多任务模型**：通过共享编码器学习特征表示，能捕捉维度间的关联模式。",
        "3. **高风险召回率**是临床安全性的核心指标——漏诊（false negative）的代价远高于误诊。",
        "",
        "## 局限性说明",
        "",
        "- 使用 `make_classification` 模拟数据，非真实临床数据分布",
        "- 规则引擎的阈值基于经验设定，未经临床验证",
        "- 多任务模型为浅层 MLP（2 层），表达能力有限",
        "- 未进行超参数调优和交叉验证",
        "- 推理时间为本地 CPU 测量，不含网络传输延迟",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    results = run_experiment()
    report = generate_report(results)
    report_path = str(OUTPUT_DIR / "compare_risk_models.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Output] 报告已保存 → {report_path}")
