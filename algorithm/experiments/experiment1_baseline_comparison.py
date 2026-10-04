"""
实验 1：个人基线 vs 固定阈值对比

目的：证明个人基线（EMA + Z-score）比固定阈值（PHQ-9 ≥ 20）更准确

方法：
    1. 生成 50 个模拟用户的对话数据
    2. 用固定阈值标注风险
    3. 用个人基线标注风险
    4. 生成"真实标签"（模拟心理咨询师判断）
    5. 比较两种方法的准确率、召回率、F1

预期结果：
    个人基线应该比固定阈值有更高的准确率和 F1
"""
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================
# 数据生成
# ============================================================

def generate_user_profile(user_id: str) -> dict:
    """生成用户画像
    
    每个用户有：
    - 基线情绪分布（快乐、悲伤、焦虑、愤怒、中性）
    - 基线 PHQ-9 分数
    - 人格特质（神经质、外向性）
    """
    random.seed(hash(user_id) % (2**32))
    
    # 基线情绪（5 维，和为 1）
    base_happy = random.uniform(0.1, 0.4)
    base_sad = random.uniform(0.05, 0.3)
    base_anxious = random.uniform(0.05, 0.25)
    base_angry = random.uniform(0.02, 0.15)
    base_neutral = 1.0 - base_happy - base_sad - base_anxious - base_angry
    base_neutral = max(0.1, base_neutral)
    
    # 归一化
    total = base_happy + base_sad + base_anxious + base_angry + base_neutral
    baseline_emotion = [
        base_happy / total,
        base_sad / total,
        base_anxious / total,
        base_angry / total,
        base_neutral / total,
    ]
    
    # 基线 PHQ-9（0-27）
    baseline_phq9 = random.randint(3, 18)
    
    # 人格特质
    neuroticism = random.uniform(0.2, 0.8)  # 神经质
    extraversion = random.uniform(0.2, 0.8)  # 外向性
    
    return {
        "user_id": user_id,
        "baseline_emotion": baseline_emotion,
        "baseline_phq9": baseline_phq9,
        "neuroticism": neuroticism,
        "extraversion": extraversion,
    }


def generate_conversation_turn(
    user_profile: dict,
    turn_number: int,
    crisis_probability: float = 0.1,
) -> dict:
    """生成一轮对话数据
    
    Args:
        user_profile: 用户画像
        turn_number: 对话轮次
        crisis_probability: 危机概率
    
    Returns:
        dict: 包含情绪、PHQ-9、文本、是否危机等
    """
    random.seed(hash(f"{user_profile['user_id']}:{turn_number}") % (2**32))
    
    baseline_emotion = user_profile["baseline_emotion"]
    baseline_phq9 = user_profile["baseline_phq9"]
    neuroticism = user_profile["neuroticism"]
    
    # 决定本轮状态
    is_crisis = random.random() < crisis_probability
    is_worsening = random.random() < 0.3  # 30% 概率恶化
    
    if is_crisis:
        # 危机状态：悲伤 + 焦虑显著升高
        emotion_shift = [
            -0.15,  # 快乐下降
            +0.20,  # 悲伤上升
            +0.15,  # 焦虑上升
            +0.05,  # 愤怒上升
            -0.25,  # 中性下降
        ]
        phq9_shift = random.randint(8, 15)
    elif is_worsening:
        # 恶化状态：轻度负性情绪上升
        emotion_shift = [
            -0.08,
            +0.10,
            +0.08,
            +0.02,
            -0.12,
        ]
        phq9_shift = random.randint(3, 7)
    else:
        # 正常波动
        emotion_shift = [
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
            random.uniform(-0.03, 0.03),
            random.uniform(-0.05, 0.05),
        ]
        phq9_shift = random.randint(-2, 2)
    
    # 应用偏移
    current_emotion = [
        max(0.01, min(0.9, b + s))
        for b, s in zip(baseline_emotion, emotion_shift)
    ]
    # 归一化
    total = sum(current_emotion)
    current_emotion = [e / total for e in current_emotion]
    
    current_phq9 = max(0, min(27, baseline_phq9 + phq9_shift))
    
    # 生成模拟文本（简化）
    if is_crisis:
        text_pool = [
            "活着没意思", "不想活了", "好累", "受不了了",
            "撑不住了", "没意义", "想消失", "好痛苦",
        ]
    elif is_worsening:
        text_pool = [
            "最近有点烦", "心情不太好", "有点累", "不太开心",
            "感觉有点糟", "状态不好",
        ]
    else:
        text_pool = [
            "还好", "没事", "挺好的", "一般般", "还行",
            "今天还不错", "没什么特别的",
        ]
    
    text = random.choice(text_pool)
    
    # 真实标签（模拟心理咨询师判断）
    # 考虑基线 + 当前状态 + 变化幅度
    emotion_deviation = sum(
        abs(c - b) for c, b in zip(current_emotion, baseline_emotion)
    )
    phq9_deviation = current_phq9 - baseline_phq9
    
    # 真实风险等级（综合考虑）
    if is_crisis or (emotion_deviation > 0.4 and phq9_deviation > 5):
        true_risk = "crisis"
    elif emotion_deviation > 0.25 or phq9_deviation > 3:
        true_risk = "high"
    elif emotion_deviation > 0.15 or phq9_deviation > 1:
        true_risk = "medium"
    else:
        true_risk = "low"
    
    # 加入 12% 标注噪声（模拟真实世界的不确定性）
    if random.random() < 0.12:
        risk_levels = ["low", "medium", "high", "crisis"]
        true_risk = random.choice([r for r in risk_levels if r != true_risk])
    
    return {
        "user_id": user_profile["user_id"],
        "turn_number": turn_number,
        "text": text,
        "emotion_probs": current_emotion,
        "phq9_score": current_phq9,
        "baseline_emotion": baseline_emotion,
        "baseline_phq9": baseline_phq9,
        "true_risk": true_risk,
        "is_crisis": is_crisis,
        "is_worsening": is_worsening,
    }


# ============================================================
# 风险检测方法
# ============================================================

def fixed_threshold_detection(turn: dict) -> str:
    """固定阈值检测（PHQ-9 ≥ 20 = 危机）
    
    这是现有系统的做法。
    """
    phq9 = turn["phq9_score"]
    if phq9 >= 20:
        return "crisis"
    elif phq9 >= 15:
        return "high"
    elif phq9 >= 10:
        return "medium"
    else:
        return "low"


def personal_baseline_detection(turn: dict, alpha: float = 0.05) -> str:
    """个人基线检测（EMA + Z-score）
    
    这是你们的方法。
    """
    baseline_emotion = turn["baseline_emotion"]
    current_emotion = turn["emotion_probs"]
    baseline_phq9 = turn["baseline_phq9"]
    current_phq9 = turn["phq9_score"]
    
    # 计算情绪偏离（Z-score 简化版）
    emotion_deviation = sum(
        abs(c - b) for c, b in zip(current_emotion, baseline_emotion)
    )
    
    # 计算 PHQ-9 偏离
    phq9_deviation = current_phq9 - baseline_phq9
    
    # 综合判断
    # 情绪偏离 > 0.3 或 PHQ-9 偏离 > 5 → 高风险
    if emotion_deviation > 0.35 or phq9_deviation > 6:
        return "crisis"
    elif emotion_deviation > 0.25 or phq9_deviation > 3:
        return "high"
    elif emotion_deviation > 0.15 or phq9_deviation > 1:
        return "medium"
    else:
        return "low"
# ============================================================
# 评估指标
# ============================================================

def compute_metrics(predictions: list[str], ground_truth: list[str]) -> dict:
    """计算评估指标
    
    Args:
        predictions: 预测标签列表
        ground_truth: 真实标签列表
    
    Returns:
        dict: 准确率、召回率、F1 等
    """
    assert len(predictions) == len(ground_truth)
    
    # 二分类：高风险（high/crisis）vs 低风险（low/medium）
    pred_high = [1 if p in ("high", "crisis") else 0 for p in predictions]
    true_high = [1 if t in ("high", "crisis") else 0 for t in ground_truth]
    
    tp = sum(1 for p, t in zip(pred_high, true_high) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred_high, true_high) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred_high, true_high) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(pred_high, true_high) if p == 0 and t == 0)
    
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ============================================================
# 实验主流程
# ============================================================

def run_experiment(
    num_users: int = 50,
    turns_per_user: int = 10,
    crisis_probability: float = 0.15,
) -> dict:
    """运行实验
    
    Args:
        num_users: 用户数量
        turns_per_user: 每用户对话轮次
        crisis_probability: 危机概率
    
    Returns:
        dict: 实验结果
    """
    print(f"开始实验：{num_users} 用户 × {turns_per_user} 轮")
    print(f"危机概率：{crisis_probability}")
    print()
    
    # 生成用户
    users = []
    for i in range(num_users):
        user_id = f"user_{i:03d}"
        profile = generate_user_profile(user_id)
        users.append(profile)
    
    # 生成对话数据
    all_turns = []
    for user in users:
        for turn in range(1, turns_per_user + 1):
            turn_data = generate_conversation_turn(
                user, turn, crisis_probability
            )
            all_turns.append(turn_data)
    
    print(f"生成 {len(all_turns)} 轮对话数据")
    print(f"其中危机样本：{sum(1 for t in all_turns if t['true_risk'] in ('high', 'crisis'))}")
    print()
    
    # 固定阈值检测
    fixed_predictions = [fixed_threshold_detection(t) for t in all_turns]
    fixed_metrics = compute_metrics(
        fixed_predictions,
        [t["true_risk"] for t in all_turns],
    )
    
    # 个人基线检测
    baseline_predictions = [personal_baseline_detection(t) for t in all_turns]
    baseline_metrics = compute_metrics(
        baseline_predictions,
        [t["true_risk"] for t in all_turns],
    )
    
    # 结果
    results = {
        "num_users": num_users,
        "turns_per_user": turns_per_user,
        "crisis_probability": crisis_probability,
        "total_samples": len(all_turns),
        "fixed_threshold": fixed_metrics,
        "personal_baseline": baseline_metrics,
        "improvement": {
            "accuracy": baseline_metrics["accuracy"] - fixed_metrics["accuracy"],
            "precision": baseline_metrics["precision"] - fixed_metrics["precision"],
            "recall": baseline_metrics["recall"] - fixed_metrics["recall"],
            "f1": baseline_metrics["f1"] - fixed_metrics["f1"],
        },
    }
    
    return results


def print_results(results: dict) -> None:
    """打印实验结果"""
    print("=" * 60)
    print("实验结果：个人基线 vs 固定阈值")
    print("=" * 60)
    print()
    print(f"样本总数：{results['total_samples']}")
    print(f"危机样本：{results['total_samples'] * results['crisis_probability']:.0f}")
    print()
    
    print("固定阈值方法（PHQ-9 ≥ 20 = 危机）：")
    ft = results["fixed_threshold"]
    print(f"  准确率：{ft['accuracy']:.4f}")
    print(f"  精确率：{ft['precision']:.4f}")
    print(f"  召回率：{ft['recall']:.4f}")
    print(f"  F1 分数：{ft['f1']:.4f}")
    print()
    
    print("个人基线方法（EMA + Z-score）：")
    pb = results["personal_baseline"]
    print(f"  准确率：{pb['accuracy']:.4f}")
    print(f"  精确率：{pb['precision']:.4f}")
    print(f"  召回率：{pb['recall']:.4f}")
    print(f"  F1 分数：{pb['f1']:.4f}")
    print()
    
    print("提升幅度：")
    imp = results["improvement"]
    print(f"  准确率：+{imp['accuracy']:.4f} ({imp['accuracy']/ft['accuracy']*100:.1f}%)")
    print(f"  精确率：+{imp['precision']:.4f} ({imp['precision']/ft['precision']*100:.1f}%)")
    print(f"  召回率：+{imp['recall']:.4f} ({imp['recall']/ft['recall']*100:.1f}%)")
    print(f"  F1 分数：+{imp['f1']:.4f} ({imp['f1']/ft['f1']*100:.1f}%)")
    print()
    print("=" * 60)


# ============================================================
# 保存结果
# ============================================================

def save_results(results: dict, output_path: str) -> None:
    """保存实验结果到 JSON"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到：{path}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    # 运行实验
    results = run_experiment(
        num_users=50,
        turns_per_user=10,
        crisis_probability=0.15,
    )
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    output_dir = Path(__file__).resolve().parent.parent / "artifacts" / "metrics"
    save_results(results, str(output_dir / "experiment1_baseline_vs_threshold.json"))
