"""
对比实验 2：单模态 vs 双模态 —— 情感感知融合对比

实验目的：
    比较仅使用文本模态与文本+语音双模态融合在情感识别上的差异。

实验条件：
    - 数据集：模拟文本情绪概率 + 语音风险概率
    - 样本量：800
    - 分类类别：5 类情绪（快乐/悲伤/焦虑/愤怒/中性）
    - 融合策略：加权平均 + Stacking
    - 随机种子：42

指标定义：
    - Accuracy: 情绪分类准确率
    - F1 (macro): 宏平均 F1
    - Confidence: 平均预测置信度
    - Robustness: 噪声条件下的性能稳定性

局限性说明：
    - 使用模拟概率分布，非真实模型输出
    - 语音特征为单维风险概率，非真实语音嵌入
    - 融合权重为经验值，未经调优
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ============================================================
# 路径配置
# ============================================================

_MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _MODULE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_SAMPLES = 800
N_TEXT_CLASSES = 5
EMOTION_LABELS = ["快乐", "悲伤", "焦虑", "愤怒", "中性"]


# ============================================================
# 数据生成
# ============================================================

def generate_emotion_data(n_samples: int = N_SAMPLES):
    """生成模拟情感数据

    模拟文本和语音两个模态的概率分布输出。

    Returns:
        (text_probs, audio_probs, labels, text_probs_noisy, audio_probs_noisy)
    """
    rng = np.random.RandomState(RANDOM_SEED)

    # 生成真实标签
    labels = rng.randint(0, N_TEXT_CLASSES, n_samples)

    # 文本模态概率分布
    text_probs = np.zeros((n_samples, N_TEXT_CLASSES))
    for i in range(n_samples):
        true_label = labels[i]
        # 真实类别的概率较高，其余较低
        probs = rng.dirichlet([0.5] * N_TEXT_CLASSES)
        probs[true_label] += rng.uniform(0.3, 0.6)
        probs /= probs.sum()
        text_probs[i] = probs

    # 语音模态概率分布（与文本有一定相关性但不完全一致）
    audio_probs = np.zeros((n_samples, N_TEXT_CLASSES))
    for i in range(n_samples):
        true_label = labels[i]
        # 语音模态的区分度略低于文本
        probs = rng.dirichlet([0.8] * N_TEXT_CLASSES)
        probs[true_label] += rng.uniform(0.2, 0.5)
        probs /= probs.sum()
        audio_probs[i] = probs

    # 添加噪声版本（模拟真实场景的干扰）
    text_noisy = text_probs + rng.randn(*text_probs.shape) * 0.05
    text_noisy = np.clip(text_noisy, 0, 1)
    text_noisy /= text_noisy.sum(axis=1, keepdims=True)

    audio_noisy = audio_probs + rng.randn(*audio_probs.shape) * 0.08
    audio_noisy = np.clip(audio_noisy, 0, 1)
    audio_noisy /= audio_noisy.sum(axis=1, keepdims=True)

    return text_probs, audio_probs, labels, text_noisy, audio_noisy


# ============================================================
# 单模态模型
# ============================================================

class TextOnlyModel:
    """纯文本情感分类模型"""

    def __init__(self):
        self.model = LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=200,
        )

    def train(self, text_probs, labels):
        self.model.fit(text_probs, labels)

    def predict(self, text_probs):
        return self.model.predict(text_probs)

    def predict_proba(self, text_probs):
        return self.model.predict_proba(text_probs)


class WeightedFusionModel:
    """加权融合模型"""

    def __init__(self, text_weight: float = 0.6, audio_weight: float = 0.4):
        self.text_weight = text_weight
        self.audio_weight = audio_weight
        self.model = LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=200,
        )

    def fuse(self, text_probs, audio_probs):
        """加权平均融合"""
        return (
            self.text_weight * text_probs
            + self.audio_weight * audio_probs
        )

    def train(self, text_probs, audio_probs, labels):
        fused = self.fuse(text_probs, audio_probs)
        self.model.fit(fused, labels)

    def predict(self, text_probs, audio_probs):
        fused = self.fuse(text_probs, audio_probs)
        return self.model.predict(fused)


class StackingFusionModel:
    """Stacking 融合模型"""

    def __init__(self):
        self.meta_model = LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=200,
        )

    def train(self, text_probs, audio_probs, labels):
        # 拼接两个模态的概率作为特征
        features = np.concatenate([text_probs, audio_probs], axis=1)
        self.meta_model.fit(features, labels)

    def predict(self, text_probs, audio_probs):
        features = np.concatenate([text_probs, audio_probs], axis=1)
        return self.meta_model.predict(features)


# ============================================================
# 实验运行
# ============================================================

def run_experiment() -> dict:
    """运行单模态 vs 双模态对比实验"""
    print("\n" + "=" * 60)
    print("实验 2：单模态 vs 双模态 —— 情感感知融合对比")
    print("=" * 60)

    text_probs, audio_probs, labels, text_noisy, audio_noisy = generate_emotion_data()

    # 划分训练/测试集
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_SEED
    )

    text_train, text_test = text_probs[train_idx], text_probs[test_idx]
    audio_train, audio_test = audio_probs[train_idx], audio_probs[test_idx]
    labels_train, labels_test = labels[train_idx], labels[test_idx]
    text_noisy_test = text_noisy[test_idx]
    audio_noisy_test = audio_noisy[test_idx]

    print(f"数据规模：训练集 {len(labels_train)}, 测试集 {len(labels_test)}")
    print(f"分类类别：{N_TEXT_CLASSES} 类 ({', '.join(EMOTION_LABELS)})")

    results = {}

    # --- 1. 纯文本单模态 ---
    print("\n[纯文本] 训练中...")
    text_model = TextOnlyModel()
    start = time.time()
    text_model.train(text_train, labels_train)
    train_time = time.time() - start

    start = time.time()
    text_pred = text_model.predict(text_test)
    text_time = (time.time() - start) / len(labels_test) * 1000

    # 噪声测试
    text_pred_noisy = text_model.predict(text_noisy_test)

    results["text_only"] = {
        "accuracy": float(accuracy_score(labels_test, text_pred)),
        "f1_macro": float(f1_score(labels_test, text_pred, average="macro")),
        "inference_time_ms": round(text_time, 4),
        "robustness_noisy": float(accuracy_score(labels_test, text_pred_noisy)),
    }
    print(f"  Acc={results['text_only']['accuracy']:.4f}, "
          f"F1={results['text_only']['f1_macro']:.4f}")

    # --- 2. 加权融合双模态 ---
    print("\n[加权融合] 训练中...")
    weighted_model = WeightedFusionModel(text_weight=0.6, audio_weight=0.4)
    start = time.time()
    weighted_model.train(text_train, audio_train, labels_train)
    train_time = time.time() - start

    start = time.time()
    weighted_pred = weighted_model.predict(text_test, audio_test)
    weighted_time = (time.time() - start) / len(labels_test) * 1000

    # 噪声测试
    weighted_pred_noisy = weighted_model.predict(text_noisy_test, audio_noisy_test)

    results["weighted_fusion"] = {
        "accuracy": float(accuracy_score(labels_test, weighted_pred)),
        "f1_macro": float(f1_score(labels_test, weighted_pred, average="macro")),
        "inference_time_ms": round(weighted_time, 4),
        "robustness_noisy": float(accuracy_score(labels_test, weighted_pred_noisy)),
    }
    print(f"  Acc={results['weighted_fusion']['accuracy']:.4f}, "
          f"F1={results['weighted_fusion']['f1_macro']:.4f}")

    # --- 3. Stacking 融合双模态 ---
    print("\n[Stacking 融合] 训练中...")
    stacking_model = StackingFusionModel()
    start = time.time()
    stacking_model.train(text_train, audio_train, labels_train)
    train_time = time.time() - start

    start = time.time()
    stacking_pred = stacking_model.predict(text_test, audio_test)
    stacking_time = (time.time() - start) / len(labels_test) * 1000

    # 噪声测试
    stacking_features_noisy = np.concatenate([text_noisy_test, audio_noisy_test], axis=1)
    stacking_pred_noisy = stacking_model.meta_model.predict(stacking_features_noisy)

    results["stacking_fusion"] = {
        "accuracy": float(accuracy_score(labels_test, stacking_pred)),
        "f1_macro": float(f1_score(labels_test, stacking_pred, average="macro")),
        "inference_time_ms": round(stacking_time, 4),
        "robustness_noisy": float(accuracy_score(labels_test, stacking_pred_noisy)),
    }
    print(f"  Acc={results['stacking_fusion']['accuracy']:.4f}, "
          f"F1={results['stacking_fusion']['f1_macro']:.4f}")

    results["_meta"] = {
        "n_train": len(labels_train),
        "n_test": len(labels_test),
        "n_classes": N_TEXT_CLASSES,
        "class_names": EMOTION_LABELS,
        "random_seed": RANDOM_SEED,
    }

    return results


def generate_report(results: dict) -> str:
    """生成 Markdown 对比报告"""
    text = results["text_only"]
    weighted = results["weighted_fusion"]
    stacking = results["stacking_fusion"]
    meta = results["_meta"]

    lines = [
        "# 实验 2：单模态 vs 双模态 —— 情感感知融合对比",
        "",
        "## 实验条件",
        "",
        "| 配置项 | 值 |",
        "|--------|-----|",
        f"| 数据集 | 模拟情绪概率分布 |",
        f"| 训练集样本数 | {meta['n_train']} |",
        f"| 测试集样本数 | {meta['n_test']} |",
        f"| 分类类别 | {meta['n_classes']} 类 ({', '.join(meta['class_names'])}) |",
        f"| 文本模态 | 5 维情绪概率分布 |",
        f"| 语音模态 | 5 维情绪概率分布 |",
        f"| 融合权重 | 文本 0.6 + 语音 0.4 |",
        f"| 随机种子 | {meta['random_seed']} |",
        "",
        "## 指标定义",
        "",
        "| 指标 | 定义 | 意义 |",
        "|------|------|------|",
        "| Accuracy | 情绪分类准确率 | 整体识别能力 |",
        "| F1 (macro) | 宏平均 F1 | 各类别均衡性能 |",
        "| Inference Time | 单样本推理延迟 (ms) | 实时性 |",
        "| Robustness | 噪声条件下的准确率 | 鲁棒性（模拟真实干扰） |",
        "",
        "## 实验结果",
        "",
        "| 模型 | Accuracy | F1 (macro) | Inference (ms) | Robustness (噪声) |",
        "|------|----------|------------|----------------|-------------------|",
        f"| 纯文本（单模态） | {text['accuracy']:.4f} | {text['f1_macro']:.4f} | "
        f"{text['inference_time_ms']:.4f} | {text['robustness_noisy']:.4f} |",
        f"| 加权融合（双模态） | {weighted['accuracy']:.4f} | {weighted['f1_macro']:.4f} | "
        f"{weighted['inference_time_ms']:.4f} | {weighted['robustness_noisy']:.4f} |",
        f"| Stacking 融合（双模态） | {stacking['accuracy']:.4f} | {stacking['f1_macro']:.4f} | "
        f"{stacking['inference_time_ms']:.4f} | {stacking['robustness_noisy']:.4f} |",
        "",
        "## 分析与结论",
        "",
        "1. **双模态融合**通过结合文本和语音信息，能提供更全面的情感判断。",
        "2. **加权融合**实现简单，适合实时性要求高的场景。",
        "3. **Stacking 融合**通过 meta-learner 学习最优组合策略，通常性能更好。",
        "4. **鲁棒性**：双模态在噪声条件下通常更稳定，因为一个模态受干扰时另一个可提供补偿。",
        "",
        "## 局限性说明",
        "",
        "- 使用模拟概率分布，非真实 NLP/语音模型输出",
        "- 语音特征为 5 维概率分布，非真实语音嵌入（如 wav2vec 2.0）",
        "- 融合权重为经验值（0.6/0.4），未经网格搜索调优",
        "- 噪声为高斯噪声添加，非真实环境噪声模式",
        "- 未考虑模态缺失时的降级策略",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    results = run_experiment()
    report = generate_report(results)
    report_path = str(OUTPUT_DIR / "compare_emotion_models.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Output] 报告已保存 → {report_path}")
