"""
训练脚本 —— 完整的训练/验证/评估流程

包含：数据加载 → tokenizer → 模型定义 → 训练 → 验证 → 保存 → 评估报告

使用方式：
    # 使用假数据跑通流程
    python -m perception.text_emotion.train --fake

    # 使用真实数据
    python -m perception.text_emotion.train --data path/to/train.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from transformers import set_seed

from perception.text_emotion.config import (
    BATCH_SIZE,
    CONFUSION_MATRIX_PATH,
    DEVICE,
    EPOCHS,
    ID2LABEL,
    LEARNING_RATE,
    METRICS_SAVE_PATH,
    MODEL_NAME,
    MODEL_SAVE_PATH,
    NUM_LABELS,
    SEED,
    VALIDATION_SPLIT,
    WEIGHT_DECAY,
)
from perception.text_emotion.dataset import (
    EmotionDataset,
    compute_class_weights,
    generate_fake_data,
    load_tokenizer,
)
from perception.text_emotion.model import TextEmotionClassifier


# ============================================================
# 模型保存/加载工具（自定义 nn.Module）
# ============================================================

def _save_model(model: nn.Module, save_dir: str) -> None:
    """保存自定义模型（state_dict + 配置 JSON）"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    config = {
        "num_labels": NUM_LABELS,
        "model_name": MODEL_NAME,
    }
    with open(os.path.join(save_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ============================================================
# 固定随机种子
# ============================================================

def fix_seed(seed: int = SEED) -> None:
    """固定所有随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ============================================================
# 训练循环
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """训练一个 epoch

    Returns:
        平均训练损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================
# 验证循环
# ============================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, list[int], list[int]]:
    """验证模型

    Returns:
        (平均验证损失, 所有预测标签, 所有真实标签)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, all_preds, all_labels


# ============================================================
# 评估可视化
# ============================================================

def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    save_path: str,
) -> None:
    """保存混淆矩阵图片"""
    labels = list(ID2LABEL.keys())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="真实标签",
        xlabel="预测标签",
        title="混淆矩阵",
    )

    # 在格子中标注数字
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] 混淆矩阵已保存至: {save_path}")


# ============================================================
# 主流程
# ============================================================

def main(data_path: str | None = None, use_fake: bool = False) -> None:
    """训练主流程

    Args:
        data_path: 训练数据 JSON 路径
        use_fake: 是否使用假数据
    """
    fix_seed()

    # 1. 准备数据
    if use_fake or data_path is None:
        print("[INFO] 使用假数据跑通训练流程...")
        data_path = generate_fake_data(num_per_class=40)
        print(f"[INFO] 假数据已生成: {data_path}")

    print(f"[INFO] 加载数据: {data_path}")
    tokenizer = load_tokenizer()
    full_dataset = EmotionDataset(data_path=data_path, tokenizer=tokenizer)
    print(f"[INFO] 数据集大小: {len(full_dataset)}")

    # 2. 划分训练集/验证集
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"[INFO] 训练集: {train_size} | 验证集: {val_size}")

    # 3. 计算类别权重（处理不均衡）
    train_indices = train_dataset.indices
    train_labels = [full_dataset.labels[i] for i in train_indices]

    from collections import Counter
    label_counts = Counter(train_labels)
    total = len(train_labels)
    weights_list = [total / (NUM_LABELS * label_counts.get(i, 1)) for i in range(NUM_LABELS)]
    class_weights = torch.tensor(weights_list, dtype=torch.float)
    print(f"[INFO] 类别权重: {dict(zip(ID2LABEL.values(), [f'{w:.3f}' for w in weights_list]))}")

    # 4. DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 模型 & 优化器 & 损失函数
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[INFO] 使用设备: {device}")

    model = TextEmotionClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 6. 训练循环
    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存自定义模型（state_dict + 配置）
            _save_model(model, MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            print(f"[INFO] 保存最优模型 → {MODEL_SAVE_PATH}")

    # 7. 最终评估
    print("\n" + "=" * 60)
    print("最终验证集评估报告")
    print("=" * 60)

    _, final_preds, final_labels = validate(model, val_loader, criterion, device)
    target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(
        final_labels, final_preds,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    # 8. 保存评估指标
    metrics = {
        "classification_report": classification_report(
            final_labels, final_preds,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "best_val_loss": best_val_loss,
        "num_train": train_size,
        "num_val": val_size,
    }
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 评估指标已保存至: {METRICS_SAVE_PATH}")

    # 9. 混淆矩阵
    save_confusion_matrix(final_labels, final_preds, CONFUSION_MATRIX_PATH)

    print("\n[INFO] 训练完成！")


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文文本情感分类 - 训练脚本")
    parser.add_argument("--data", type=str, default=None, help="训练数据 JSON 路径")
    parser.add_argument("--fake", action="store_true", help="使用假数据跑通流程")
    args = parser.parse_args()

    main(data_path=args.data, use_fake=args.fake)
