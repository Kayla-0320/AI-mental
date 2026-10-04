"""
对抗去偏见训练模块 —— 消除文本情感模型对敏感属性的依赖

基于 Adversarial Debiasing (Zhang et al., 2018) 思想：
- 主分类器：预测情感标签（主任务）
- 对抗分类器：从主分类器特征中预测敏感属性（如性别/方言/年龄组）
- 训练目标：主分类器最小化情感损失 + 最大化对抗分类器的损失（GRL）
- 最终效果：主分类器学到的特征不包含敏感属性信息

使用方式：
    from perception.text_emotion.adversarial_debias import (
        AdversarialEmotionModel,
        GradientReversalLayer,
        train_with_adversarial_debiasing,
    )
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from shared.dataclasses import EmotionResult


# ============================================================
# 梯度反转层 (Gradient Reversal Layer, GRL)
# ============================================================

class GradientReversalFunction(torch.autograd.Function):
    """梯度反转函数

    前向传播：恒等映射
    反向传播：梯度乘以 -lambda
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """梯度反转层

    Args:
        lambda_val: 梯度反转系数，控制对抗强度
    """

    def __init__(self, lambda_val: float = 1.0) -> None:
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)

    def set_lambda(self, lambda_val: float) -> None:
        """动态调整 lambda（训练过程中可逐步增大）"""
        self.lambda_val = lambda_val


# ============================================================
# 对抗情感模型
# ============================================================

@dataclass
class AdversarialConfig:
    """对抗训练配置"""
    num_emotion_labels: int = 5
    num_sensitive_classes: int = 3  # 敏感属性类别数（如性别=2, 方言=3）
    sensitive_attr_name: str = "gender"
    adversarial_lambda: float = 1.0
    adversarial_weight: float = 0.3  # 对抗损失权重 alpha
    hidden_dim: int = 256
    dropout: float = 0.1
    seed: int = 42


class AdversarialEmotionModel(nn.Module):
    """带对抗去偏见的文本情感模型

    结构：
    - 共享编码器（RoBERTa）→ 特征表示 h
    - 主分类头：h → 情感标签（正常 CE loss）
    - 对抗分类头：GRL(h) → 敏感属性（adversarial CE loss）

    训练目标：
    min_theta max_adv { L_emotion - alpha * L_adversarial }

    Args:
        backbone: 预训练 backbone（RoBERTa 等）
        config: 对抗训练配置
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: AdversarialConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AdversarialConfig()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, 'hidden_size', 768)

        # 主分类头
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_size, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_emotion_labels),
        )

        # 对抗分类头（通过 GRL）
        self.grl = GradientReversalLayer(self.config.adversarial_lambda)
        self.adversary_classifier = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_size, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_sensitive_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """前向传播

        Returns:
            dict with keys:
                - emotion_logits: (batch, num_emotion_labels)
                - adversary_logits: (batch, num_sensitive_classes)
                - features: (batch, hidden_size) 共享特征
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        emotion_logits = self.emotion_classifier(cls_output)
        adversary_input = self.grl(cls_output)
        adversary_logits = self.adversary_classifier(adversary_input)

        return {
            "emotion_logits": emotion_logits,
            "adversary_logits": adversary_logits,
            "features": cls_output,
        }

    def predict_emotion(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """仅预测情感（推理时使用）"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.emotion_classifier(cls_output)


# ============================================================
# 对抗训练循环
# ============================================================

@dataclass
class AdversarialTrainResult:
    """对抗训练结果"""
    final_emotion_loss: float = 0.0
    final_adversary_loss: float = 0.0
    final_emotion_acc: float = 0.0
    final_adversary_acc: float = 0.0  # 越低越好（理想接近随机水平）
    num_epochs: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


def _compute_disparity(
    predictions: list[int],
    sensitive_labels: list[int],
    num_classes: int,
) -> dict[str, float]:
    """计算预测在不同敏感属性组间的差异（DPD 指标）

    Returns:
        各组预测分布的差异度量
    """
    groups: dict[int, list[int]] = {}
    for pred, sens in zip(predictions, sensitive_labels):
        groups.setdefault(sens, []).append(pred)

    # 各组预测为正类（非 neutral）的比例
    group_positive_rates: dict[int, float] = {}
    for group_id, preds in groups.items():
        if len(preds) > 0:
            group_positive_rates[group_id] = sum(1 for p in preds if p != 0) / len(preds)
        else:
            group_positive_rates[group_id] = 0.0

    rates = list(group_positive_rates.values())
    if len(rates) >= 2:
        max_rate = max(rates)
        min_rate = min(rates)
        disparity = max_rate - min_rate
    else:
        disparity = 0.0

    return {
        "disparity": disparity,
        "group_rates": {str(k): v for k, v in group_positive_rates.items()},
        "group_sizes": {str(k): len(v) for k, v in groups.items()},
    }


def train_with_adversarial_debiasing(
    model: AdversarialEmotionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    lr_main: float = 2e-5,
    lr_adv: float = 1e-4,
    weight_decay: float = 0.01,
    device: str = "cpu",
    alpha: float = 0.3,
    seed: int = 42,
) -> AdversarialTrainResult:
    """对抗去偏见训练

    Args:
        model: 对抗情感模型
        train_loader: 训练数据（需包含 sensitive_label 字段）
        val_loader: 验证数据
        num_epochs: 训练轮数
        lr_main: 主分类器学习率
        lr_adv: 对抗分类器学习率
        weight_decay: L2 正则化
        device: 计算设备
        alpha: 对抗损失权重
        seed: 随机种子

    Returns:
        AdversarialTrainResult
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = model.to(device)

    # 分离参数组：backbone + emotion head vs adversary head
    backbone_params = list(model.backbone.parameters())
    emotion_params = list(model.emotion_classifier.parameters())
    adversary_params = list(model.adversary_classifier.parameters())

    optimizer_main = torch.optim.AdamW(
        [{"params": backbone_params}, {"params": emotion_params}],
        lr=lr_main,
        weight_decay=weight_decay,
    )
    optimizer_adv = torch.optim.Adam(adversary_params, lr=lr_adv)

    criterion = nn.CrossEntropyLoss()
    result = AdversarialTrainResult(num_epochs=num_epochs)

    for epoch in range(1, num_epochs + 1):
        # 动态调整 lambda：随训练进度逐渐增大
        progress = epoch / num_epochs
        current_lambda = model.config.adversarial_lambda * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
        model.grl.set_lambda(current_lambda)

        model.train()
        total_emotion_loss = 0.0
        total_adv_loss = 0.0
        total_combined_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["label"].to(device)
            sensitive_labels = batch["sensitive_label"].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            emotion_logits = outputs["emotion_logits"]
            adversary_logits = outputs["adversary_logits"]

            # 主损失：情感分类
            emotion_loss = criterion(emotion_logits, emotion_labels)

            # 对抗损失：预测敏感属性
            adv_loss = criterion(adversary_logits, sensitive_labels)

            # 组合损失：min emotion_loss - alpha * adv_loss
            # 注意：GRL 已经处理了梯度反转，所以这里直接减
            combined_loss = emotion_loss - alpha * adv_loss

            # 更新主网络
            optimizer_main.zero_grad()
            optimizer_adv.zero_grad()
            combined_loss.backward(retain_graph=True)
            optimizer_main.step()

            # 单独更新对抗网络（最大化对抗损失）
            adv_loss_only = criterion(adversary_logits, sensitive_labels)
            optimizer_adv.zero_grad()
            optimizer_main.zero_grad()
            adv_loss_only.backward()
            optimizer_adv.step()

            total_emotion_loss += emotion_loss.item()
            total_adv_loss += adv_loss.item()
            total_combined_loss += combined_loss.item()
            num_batches += 1

        # 验证
        model.eval()
        val_emotion_correct = 0
        val_adv_correct = 0
        val_total = 0

        all_preds = []
        all_sensitive = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                emotion_labels = batch["label"].to(device)
                sensitive_labels = batch["sensitive_label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                emotion_preds = outputs["emotion_logits"].argmax(dim=-1)
                adv_preds = outputs["adversary_logits"].argmax(dim=-1)

                val_emotion_correct += (emotion_preds == emotion_labels).sum().item()
                val_adv_correct += (adv_preds == sensitive_labels).sum().item()
                val_total += emotion_labels.size(0)

                all_preds.extend(emotion_preds.cpu().tolist())
                all_sensitive.extend(sensitive_labels.cpu().tolist())

        emotion_acc = val_emotion_correct / max(val_total, 1)
        adv_acc = val_adv_correct / max(val_total, 1)
        avg_emotion_loss = total_emotion_loss / max(num_batches, 1)
        avg_adv_loss = total_adv_loss / max(num_batches, 1)

        # 计算公平性差异
        disparity = _compute_disparity(all_preds, all_sensitive, model.config.num_emotion_labels)

        epoch_log = {
            "epoch": epoch,
            "emotion_loss": avg_emotion_loss,
            "adv_loss": avg_adv_loss,
            "emotion_acc": emotion_acc,
            "adversary_acc": adv_acc,
            "grl_lambda": current_lambda,
            "disparity": disparity["disparity"],
        }
        result.history.append(epoch_log)

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"emotion_loss={avg_emotion_loss:.4f} adv_loss={avg_adv_loss:.4f} "
            f"emotion_acc={emotion_acc:.4f} adv_acc={adv_acc:.4f} "
            f"lambda={current_lambda:.3f} disparity={disparity['disparity']:.4f}"
        )

    result.final_emotion_loss = avg_emotion_loss
    result.final_adversary_loss = avg_adv_loss
    result.final_emotion_acc = emotion_acc
    result.final_adversary_acc = adv_acc

    return result


# ============================================================
# 公平性评估
# ============================================================

@dataclass
class FairnessReport:
    """公平性评估报告"""
    overall_accuracy: float = 0.0
    group_accuracies: dict[str, float] = field(default_factory=dict)
    group_sizes: dict[str, int] = field(default_factory=dict)
    max_group_disparity: float = 0.0
    equalized_odds_diff: float = 0.0
    demographic_parity_diff: float = 0.0
    recommendation: str = ""


@torch.no_grad()
def evaluate_fairness(
    model: AdversarialEmotionModel,
    dataloader: DataLoader,
    device: str = "cpu",
) -> FairnessReport:
    """评估模型的公平性

    Args:
        model: 对抗情感模型
        dataloader: 测试数据（含 sensitive_label）
        device: 计算设备

    Returns:
        FairnessReport
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_sensitive: list[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        emotion_labels = batch["label"].to(device)
        sensitive_labels = batch["sensitive_label"].to(device)

        logits = model.predict_emotion(input_ids=input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(emotion_labels.cpu().tolist())
        all_sensitive.extend(sensitive_labels.cpu().tolist())

    # 整体准确率
    overall_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / max(len(all_labels), 1)

    # 分组准确率
    groups: dict[int, tuple[int, int]] = {}  # group_id -> (correct, total)
    group_positive: dict[int, tuple[int, int]] = {}  # group_id -> (positive_preds, total)

    for pred, label, sens in zip(all_preds, all_labels, all_sensitive):
        if sens not in groups:
            groups[sens] = (0, 0)
            group_positive[sens] = (0, 0)

        correct, total = groups[sens]
        if pred == label:
            correct += 1
        groups[sens] = (correct, total + 1)

        pos_preds, pos_total = group_positive[sens]
        if pred != 0:  # 非 neutral 视为正类
            pos_preds += 1
        group_positive[sens] = (pos_preds, pos_total + 1)

    group_accuracies: dict[str, float] = {}
    for gid, (correct, total) in groups.items():
        group_accuracies[f"group_{gid}"] = correct / max(total, 1)

    group_sizes: dict[str, int] = {f"group_{gid}": total for gid, (_, total) in groups.items()}

    # Demographic Parity Difference
    pos_rates = []
    for gid, (pos, total) in group_positive.items():
        pos_rates.append(pos / max(total, 1))
    dpd = max(pos_rates) - min(pos_rates) if len(pos_rates) >= 2 else 0.0

    # Equalized Odds Difference
    # 对每个敏感组，计算 TPR 和 FPR 的差异
    group_tpr: dict[int, float] = {}
    group_fpr: dict[int, float] = {}

    for sens in set(all_sensitive):
        tp = fp = tn = fn = 0
        for pred, label, s in zip(all_preds, all_labels, all_sensitive):
            if s != sens:
                continue
            is_positive = label != 0
            pred_positive = pred != 0
            if is_positive and pred_positive:
                tp += 1
            elif is_positive and not pred_positive:
                fn += 1
            elif not is_positive and pred_positive:
                fp += 1
            else:
                tn += 1
        group_tpr[sens] = tp / max(tp + fn, 1)
        group_fpr[sens] = fp / max(fp + tn, 1)

    tprs = list(group_tpr.values())
    fprs = list(group_fpr.values())
    eod = max(
        max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0,
        max(fprs) - min(fprs) if len(fprs) >= 2 else 0.0,
    )

    # 最大组间准确率差异
    accs = list(group_accuracies.values())
    max_acc_disp = max(accs) - min(accs) if len(accs) >= 2 else 0.0

    # 生成建议
    if dpd < 0.05 and eod < 0.05:
        recommendation = "模型公平性良好，各组间差异在可接受范围内。"
    elif dpd < 0.10 and eod < 0.10:
        recommendation = "模型存在轻微公平性差异，建议继续监控。"
    else:
        recommendation = (
            "模型公平性存在显著差异，建议：(1) 增加对抗训练轮数；"
            "(2) 增大 adversarial_weight；(3) 检查训练数据分布。"
        )

    return FairnessReport(
        overall_accuracy=overall_acc,
        group_accuracies=group_accuracies,
        group_sizes=group_sizes,
        max_group_disparity=max_acc_disp,
        equalized_odds_diff=eod,
        demographic_parity_diff=dpd,
        recommendation=recommendation,
    )


# ============================================================
# 带敏感属性的数据集
# ============================================================

class BiasedEmotionDataset(Dataset):
    """带敏感属性标签的情感数据集

    用于模拟和测试对抗去偏见训练。
    在 fake 模式下，会人为制造敏感属性与情感标签之间的相关性。

    Args:
        texts: 文本列表
        labels: 情感标签列表
        sensitive_labels: 敏感属性标签列表
        tokenizer: HuggingFace tokenizer
        max_length: 最大序列长度
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        sensitive_labels: list[int],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.sensitive_labels = sensitive_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "sensitive_label": torch.tensor(self.sensitive_labels[idx], dtype=torch.long),
        }


def generate_biased_fake_data(
    num_per_class: int = 30,
    num_sensitive_groups: int = 3,
    bias_strength: float = 0.7,
    seed: int = 42,
) -> tuple[list[str], list[int], list[int]]:
    """生成带偏见的假数据

    人为制造敏感属性与情感之间的相关性，用于测试去偏见效果。

    Args:
        num_per_class: 每个情感类别的样本数
        num_sensitive_groups: 敏感属性组数
        bias_strength: 偏见强度（0=无偏见, 1=完全偏见）
        seed: 随机种子

    Returns:
        (texts, labels, sensitive_labels)
    """
    random.seed(seed)
    np.random.seed(seed)

    emotion_templates = {
        0: ["今天天气不错", "心情还好", "一般般吧", "没什么特别的", "还行"],
        1: ["有点开心", "感觉还不错", "挺高兴的", "蛮愉快的", "心情不错"],
        2: ["有点难过", "不太开心", "感觉不太好", "有些失落", "心情低落"],
        3: ["很伤心", "非常难过", "太痛苦了", "绝望了", "崩溃了"],
        4: ["好焦虑", "紧张不安", "压力很大", "烦躁得很", "坐立难安"],
    }

    # 敏感属性修饰词（模拟方言/表达风格差异）
    sensitive_prefixes = {
        0: ["嗯", "啊", "哦"],
        1: ["呐", "呢", "呀"],
        2: ["嘛", "呗", "喽"],
    }

    texts: list[str] = []
    labels: list[int] = []
    sensitive_labels: list[int] = []

    for emotion_id in range(5):
        for _ in range(num_per_class):
            template = random.choice(emotion_templates[emotion_id])

            # 根据偏见强度决定敏感属性
            if random.random() < bias_strength:
                # 偏见模式：不同情感倾向分配给不同组
                # 组 0 倾向正面，组 2 倾向负面
                if emotion_id <= 1:
                    sens_group = 0 if random.random() < 0.7 else random.randint(0, num_sensitive_groups - 1)
                elif emotion_id >= 3:
                    sens_group = num_sensitive_groups - 1 if random.random() < 0.7 else random.randint(0, num_sensitive_groups - 1)
                else:
                    sens_group = random.randint(0, num_sensitive_groups - 1)
            else:
                sens_group = random.randint(0, num_sensitive_groups - 1)

            prefix = random.choice(sensitive_prefixes.get(sens_group, [""]))
            text = f"{prefix}，{template}"

            texts.append(text)
            labels.append(emotion_id)
            sensitive_labels.append(sens_group)

    return texts, labels, sensitive_labels
