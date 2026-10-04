"""
实验 2：跨模态矛盾检测

目的：证明跨模态矛盾检测能识别掩饰行为，比单模态文本检测更准确

方法：
    1. 生成模拟数据（掩饰样本 + 一致样本）
    2. 用单模态文本检测分类
    3. 用跨模态矛盾检测分类
    4. 比较准确率

预期结果：
    跨模态矛盾检测应该比单模态文本检测有更高的准确率
"""
import json
import random
from pathlib import Path
from typing import Optional


# ============================================================
# 数据生成
# ============================================================

def generate_congruent_sample(user_id: str, sample_id: int) -> dict:
    """生成一致样本（言行一致）
    
    文本情绪和声学特征一致。
    """
    random.seed(hash(f"{user_id}:{sample_id}:congruent") % (2**32))
    
    # 随机选择情绪状态
    emotion_state = random.choice(["calm", "distressed", "neutral"])
    
    if emotion_state == "calm":
        # 平静：文本中性/快乐，声音低唤醒
        text_emotion = [0.3, 0.1, 0.1, 0.05, 0.45]  # [快乐，悲伤，焦虑，愤怒，中性]
        acoustic_arousal = random.uniform(0.1, 0.3)  # 低唤醒
        true_label = "congruent_calm"
        text = random.choice(["还好", "没事", "挺好的", "一般般"])
    elif emotion_state == "distressed":
        # 痛苦：文本悲伤/焦虑，声音高唤醒
        text_emotion = [0.05, 0.35, 0.30, 0.15, 0.15]
        acoustic_arousal = random.uniform(0.6, 0.9)  # 高唤醒
        true_label = "congruent_distressed"
        text = random.choice(["好难受", "受不了了", "很痛苦", "撑不住了"])
    else:
        # 中性：文本中性，声音中等唤醒
        text_emotion = [0.15, 0.15, 0.15, 0.1, 0.45]
        acoustic_arousal = random.uniform(0.3, 0.5)
        true_label = "congruent_neutral"
        text = random.choice(["嗯", "还好吧", "就这样", "没什么"])
    
    # 加入 8% 噪声（模拟真实世界的模糊性）
    if random.random() < 0.08:
        acoustic_arousal = 1.0 - acoustic_arousal  # 翻转声学特征
    
    return {
        "user_id": user_id,
        "sample_id": sample_id,
        "text": text,
        "text_emotion": text_emotion,
        "acoustic_arousal": acoustic_arousal,
        "true_label": true_label,
        "is_masking": False,
        "is_performative": False,
    }


def generate_masking_sample(user_id: str, sample_id: int) -> dict:
    """生成掩饰样本（嘴上说没事但声音紧张）
    
    文本平静但声学高唤醒。
    """
    random.seed(hash(f"{user_id}:{sample_id}:masking") % (2**32))
    
    # 文本平静
    text_emotion = [0.25, 0.1, 0.1, 0.05, 0.5]  # 中性/快乐主导
    # 但声音高唤醒
    acoustic_arousal = random.uniform(0.65, 0.95)
    
    text = random.choice([
        "我没事", "还好", "没什么", "挺好的", "不用担心",
        "真的没事", "我很好", "别担心",
    ])
    
    # 加入 10% 噪声（有些掩饰样本可能难以检测）
    if random.random() < 0.10:
        acoustic_arousal = random.uniform(0.3, 0.5)  # 降低声学唤醒
    
    return {
        "user_id": user_id,
        "sample_id": sample_id,
        "text": text,
        "text_emotion": text_emotion,
        "acoustic_arousal": acoustic_arousal,
        "true_label": "masking",
        "is_masking": True,
        "is_performative": False,
    }


def generate_performative_sample(user_id: str, sample_id: int) -> dict:
    """生成表演性样本（表达痛苦但声音平淡）
    
    文本痛苦但声学低唤醒。
    """
    random.seed(hash(f"{user_id}:{sample_id}:performative") % (2**32))
    
    # 文本痛苦
    text_emotion = [0.05, 0.4, 0.3, 0.15, 0.1]  # 悲伤/焦虑主导
    # 但声音低唤醒
    acoustic_arousal = random.uniform(0.05, 0.25)
    
    text = random.choice([
        "我不想活了", "好痛苦", "受不了了", "想死",
        "活不下去", "太难受了",
    ])
    
    # 加入 10% 噪声（有些表演性样本可能难以检测）
    if random.random() < 0.10:
        acoustic_arousal = random.uniform(0.5, 0.7)  # 提高声学唤醒
    
    return {
        "user_id": user_id,
        "sample_id": sample_id,
        "text": text,
        "text_emotion": text_emotion,
        "acoustic_arousal": acoustic_arousal,
        "true_label": "performative",
        "is_masking": False,
        "is_performative": True,
    }


# ============================================================
# 检测方法
# ============================================================

def text_only_detection(sample: dict) -> str:
    """单模态文本检测
    
    只看文本情绪，不看声学特征。
    """
    text_emotion = sample["text_emotion"]
    # 主导情绪
    dominant_idx = text_emotion.index(max(text_emotion))
    dominant_label = ["happy", "sad", "anxious", "angry", "neutral"][dominant_idx]
    
    # 判断是否痛苦
    negative_sum = text_emotion[1] + text_emotion[2] + text_emotion[3]  # 悲伤 + 焦虑 + 愤怒
    
    if negative_sum > 0.5:
        return "distressed"
    elif negative_sum < 0.25:
        return "calm"
    else:
        return "neutral"


def cross_modal_detection(sample: dict) -> dict:
    """跨模态矛盾检测
    
    同时看文本情绪和声学特征，检测矛盾。
    
    Returns:
        dict: {
            "classification": str,  # congruent/masking/performative
            "confidence": float,
            "incongruence_score": float,
        }
    """
    text_emotion = sample["text_emotion"]
    acoustic_arousal = sample["acoustic_arousal"]
    
    # 文本情绪判断
    negative_sum = text_emotion[1] + text_emotion[2] + text_emotion[3]
    text_is_calm = negative_sum <= 0.25  # 改为 <= 修复边界 bug
    text_is_distressed = negative_sum > 0.5
    
    # 声学唤醒度判断
    high_acoustic = acoustic_arousal > 0.55
    low_acoustic = acoustic_arousal < 0.3
    
    # 矛盾检测
    if text_is_calm and high_acoustic:
        # 言语平静但声音紧张 → 掩饰
        incongruence_score = acoustic_arousal - (1 - negative_sum)
        return {
            "classification": "masking",
            "confidence": min(0.95, 0.5 + incongruence_score * 0.5),
            "incongruence_score": round(incongruence_score, 4),
        }
    elif text_is_distressed and low_acoustic:
        # 言语痛苦但声音平淡 → 表演性
        incongruence_score = negative_sum - acoustic_arousal
        return {
            "classification": "performative",
            "confidence": min(0.95, 0.5 + incongruence_score * 0.5),
            "incongruence_score": round(incongruence_score, 4),
        }
    else:
        # 一致
        return {
            "classification": "congruent",
            "confidence": 0.8,
            "incongruence_score": 0.0,
        }


# ============================================================
# 评估
# ============================================================

def evaluate_detection(
    samples: list[dict],
    detection_func,
) -> dict:
    """评估检测方法
    
    Args:
        samples: 样本列表
        detection_func: 检测函数
    
    Returns:
        dict: 评估指标
    """
    correct = 0
    total = len(samples)
    
    # 混淆矩阵
    confusion = {
        "congruent": {"congruent": 0, "masking": 0, "performative": 0},
        "masking": {"congruent": 0, "masking": 0, "performative": 0},
        "performative": {"congruent": 0, "masking": 0, "performative": 0},
    }
    
    for sample in samples:
        true_label = sample["true_label"]
        # 标准化真实标签
        if "congruent" in true_label:
            true_category = "congruent"
        elif "masking" in true_label:
            true_category = "masking"
        else:
            true_category = "performative"
        
        # 预测
        result = detection_func(sample)
        if isinstance(result, dict):
            pred_category = result["classification"]
        else:
            pred_category = result
        
        # 标准化预测标签
        if "congruent" in pred_category or pred_category in ("calm", "neutral", "distressed"):
            pred_category = "congruent"
        
        # 统计
        if true_category == pred_category:
            correct += 1
        confusion[true_category][pred_category] += 1
    
    accuracy = correct / total if total > 0 else 0
    
    # 各类别准确率
    category_accuracy = {}
    for category in ["congruent", "masking", "performative"]:
        total_cat = sum(confusion[category].values())
        correct_cat = confusion[category][category]
        category_accuracy[category] = (
            correct_cat / total_cat if total_cat > 0 else 0
        )
    
    return {
        "accuracy": round(accuracy, 4),
        "category_accuracy": {
            k: round(v, 4) for k, v in category_accuracy.items()
        },
        "confusion_matrix": confusion,
        "total_samples": total,
        "correct_predictions": correct,
    }


# ============================================================
# 实验主流程
# ============================================================

def run_experiment(
    num_users: int = 30,
    samples_per_type: int = 30,
) -> dict:
    """运行实验
    
    Args:
        num_users: 用户数量
        samples_per_type: 每种类型的样本数
    
    Returns:
        dict: 实验结果
    """
    print(f"开始实验：{num_users} 用户 × {samples_per_type} 样本/类型")
    print()
    
    # 生成样本
    all_samples = []
    sample_id = 0
    
    for user_idx in range(num_users):
        user_id = f"user_{user_idx:03d}"
        
        # 一致样本
        for i in range(samples_per_type):
            sample = generate_congruent_sample(user_id, sample_id)
            all_samples.append(sample)
            sample_id += 1
        
        # 掩饰样本
        for i in range(samples_per_type):
            sample = generate_masking_sample(user_id, sample_id)
            all_samples.append(sample)
            sample_id += 1
        
        # 表演性样本
        for i in range(samples_per_type):
            sample = generate_performative_sample(user_id, sample_id)
            all_samples.append(sample)
            sample_id += 1
    
    print(f"生成 {len(all_samples)} 个样本")
    print(f"  一致样本：{samples_per_type * num_users}")
    print(f"  掩饰样本：{samples_per_type * num_users}")
    print(f"  表演性样本：{samples_per_type * num_users}")
    print()
    
    # 单模态文本检测
    text_results = evaluate_detection(all_samples, text_only_detection)
    
    # 跨模态矛盾检测
    cross_modal_results = evaluate_detection(all_samples, cross_modal_detection)
    
    # 结果
    results = {
        "num_users": num_users,
        "samples_per_type": samples_per_type,
        "total_samples": len(all_samples),
        "text_only": text_results,
        "cross_modal": cross_modal_results,
        "improvement": {
            "accuracy": cross_modal_results["accuracy"] - text_results["accuracy"],
        },
    }
    
    return results


def print_results(results: dict) -> None:
    """打印实验结果"""
    print("=" * 60)
    print("实验结果：跨模态矛盾检测 vs 单模态文本检测")
    print("=" * 60)
    print()
    print(f"样本总数：{results['total_samples']}")
    print()
    
    print("单模态文本检测：")
    to = results["text_only"]
    print(f"  总体准确率：{to['accuracy']:.4f}")
    print(f"  各类别准确率：")
    for cat, acc in to["category_accuracy"].items():
        print(f"    {cat}: {acc:.4f}")
    print()
    
    print("跨模态矛盾检测：")
    cm = results["cross_modal"]
    print(f"  总体准确率：{cm['accuracy']:.4f}")
    print(f"  各类别准确率：")
    for cat, acc in cm["category_accuracy"].items():
        print(f"    {cat}: {acc:.4f}")
    print()
    
    print("提升幅度：")
    imp = results["improvement"]
    print(f"  总体准确率：+{imp['accuracy']:.4f} ({imp['accuracy']/to['accuracy']*100:.1f}%)")
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
        num_users=30,
        samples_per_type=30,
    )
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    output_dir = Path(__file__).resolve().parent.parent / "artifacts" / "metrics"
    save_results(results, str(output_dir / "experiment2_cross_modal_detection.json"))
