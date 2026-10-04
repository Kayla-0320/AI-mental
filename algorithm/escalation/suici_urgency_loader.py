"""
Suici-Urgency 数据集加载器 —— 将 Suici-Urgency JSON 格式转换为项目统一接口

参考数据：
    Suici-Urgency (finexsf/Suici-Urgency)
    许可证：CC BY-NC 4.0（仅限学术研究，禁止商用训练）
    格式：JSON 对话，含 6 级风险标注 (1-6)

⚠️ 使用限制：
    1. 仅限学术研究使用，不得用于商业模型训练
    2. 使用前须确认数据集最新许可条款
    3. 不得重新分发原始数据
    4. 引用须注明：Suici-Urgency (finexsf/Suici-Urgency)

数据字段：
    - text: 用户发言文本
    - severity: 风险等级 (1-6)，对应 CrisisSeverity 枚举
    - time_clue: 时间线索（可选，如"最近"、"昨天"）
    - line_index: 行索引（数据定位用）

核心接口：
    - load_suici_urgency(path) -> list[CrisisSample]
    - load_from_jsonl(path) -> list[CrisisSample]
    - split_dataset(samples) -> (train, val, test)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .crisis_severity import CrisisSeverity, classify_severity


# ============================================================
# 数据样本结构
# ============================================================

@dataclass
class CrisisSample:
    """单条危机标注样本

    Attributes:
        text: 用户发言文本
        severity: 标注的危机严重程度 (1-6)
        time_clue: 时间线索（可选）
        source: 数据来源标识
        line_index: 原始数据行号
    """
    text: str = ""
    severity: int = 1
    time_clue: str = ""
    source: str = "suici_urgency"
    line_index: int = 0


@dataclass
class CrisisDataset:
    """危机数据集（含统计信息）

    Attributes:
        samples: 样本列表
        source: 数据集名称
        total: 样本总数
        distribution: 各级别分布
    """
    samples: list[CrisisSample] = field(default_factory=list)
    source: str = ""
    total: int = 0
    distribution: dict[int, int] = field(default_factory=dict)


# ============================================================
# 数据加载
# ============================================================

def load_suici_urgency(path: str | Path) -> CrisisDataset:
    """从 Suici-Urgency JSON 文件加载数据集

    支持的格式：
    1. JSON 数组: [{"text": "...", "severity": 3, ...}, ...]
    2. JSON 对象: {"data": [...]}
    3. JSONL 文件: 每行一个 JSON 对象

    Args:
        path: 数据文件路径

    Returns:
        CrisisDataset: 加载后的数据集

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式错误
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    content = path.read_text(encoding="utf-8")
    samples = []

    # 优先尝试完整 JSON 解析（数组或对象）
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for i, item in enumerate(data):
                samples.append(_parse_sample(item, i))
        elif isinstance(data, dict) and "data" in data:
            for i, item in enumerate(data["data"]):
                samples.append(_parse_sample(item, i))
        elif isinstance(data, dict):
            # 单个样本对象
            samples.append(_parse_sample(data, 0))
    except json.JSONDecodeError:
        # JSON 解析失败，尝试 JSONL（每行一个 JSON 对象）
        for i, line in enumerate(content.strip().split("\n")):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append(_parse_sample(item, i))

    # 统计分布
    distribution: dict[int, int] = {}
    for s in samples:
        distribution[s.severity] = distribution.get(s.severity, 0) + 1

    return CrisisDataset(
        samples=samples,
        source="suici_urgency",
        total=len(samples),
        distribution=distribution,
    )


def load_from_jsonl(path: str | Path) -> CrisisDataset:
    """从 JSONL 文件加载（每行一个 JSON 对象）

    Args:
        path: JSONL 文件路径

    Returns:
        CrisisDataset
    """
    return load_suici_urgency(path)


def _parse_sample(item: dict, index: int) -> CrisisSample:
    """解析单条样本

    支持的字段名映射：
    - text / content / message / 文本
    - severity / level / label / risk_level / 等级
    - time_clue / time / 时间
    """
    text = (
        item.get("text") or item.get("content") or
        item.get("message") or item.get("文本") or ""
    )
    severity = (
        item.get("severity") or item.get("level") or
        item.get("label") or item.get("risk_level") or
        item.get("等级") or 1
    )
    time_clue = (
        item.get("time_clue") or item.get("time") or
        item.get("时间") or ""
    )

    severity = max(1, min(6, int(severity)))

    return CrisisSample(
        text=str(text).strip(),
        severity=severity,
        time_clue=str(time_clue).strip(),
        source="suici_urgency",
        line_index=index,
    )


# ============================================================
# 数据集分割
# ============================================================

def split_dataset(
    samples: list[CrisisSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[CrisisSample], list[CrisisSample], list[CrisisSample]]:
    """分层分割数据集（保持各级别比例）

    Args:
        samples: 样本列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        (train, val, test): 三个子集
    """
    import random
    rng = random.Random(seed)

    # 按级别分组
    by_level: dict[int, list[CrisisSample]] = {}
    for s in samples:
        by_level.setdefault(s.severity, []).append(s)

    train, val, test = [], [], []

    for level in sorted(by_level.keys()):
        group = by_level[level]
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0
        n_test = n - n_train - n_val

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    return train, val, test


# ============================================================
# 评估辅助
# ============================================================

def evaluate_classifier(
    samples: list[CrisisSample],
) -> dict:
    """评估分级器在标注数据上的表现

    使用 classify_severity 对每条样本重新分级，
    与标注等级对比计算准确率和各级别召回率。

    Args:
        samples: 带标注的样本列表

    Returns:
        dict: 评估指标
    """
    total = len(samples)
    if total == 0:
        return {"accuracy": 0.0, "total": 0}

    correct = 0
    by_level_total: dict[int, int] = {}
    by_level_correct: dict[int, int] = {}

    for sample in samples:
        result = classify_severity(sample.text)
        predicted = result.severity.value
        actual = sample.severity

        by_level_total[actual] = by_level_total.get(actual, 0) + 1

        # 允许 ±1 级容差（规则匹配不可能完美复现人工标注）
        if abs(predicted - actual) <= 1:
            correct += 1
            by_level_correct[actual] = by_level_correct.get(actual, 0) + 1

    accuracy = correct / total

    recall_by_level = {}
    for level in sorted(by_level_total.keys()):
        n_total = by_level_total[level]
        n_correct = by_level_correct.get(level, 0)
        recall_by_level[level] = round(n_correct / n_total, 4) if n_total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "recall_by_level": recall_by_level,
        "level_distribution": by_level_total,
    }


# ============================================================
# 合成演示数据（当无法获取真实数据时使用）
# ============================================================

def generate_demo_samples(n_per_level: int = 5, seed: int = 42) -> CrisisDataset:
    """生成模拟标注数据（用于测试管线）

    ⚠️ 仅用于功能验证，不具临床效度。
    真实评估须使用 Suici-Urgency 原始数据。

    Args:
        n_per_level: 每个等级生成的样本数
        seed: 随机种子

    Returns:
        CrisisDataset
    """
    import random
    rng = random.Random(seed)

    demo_texts = {
        1: [
            "今天天气不错，心情还可以",
            "作业好多啊，得赶紧写",
            "和朋友出去玩了一下午",
            "明天有个考试，有点紧张",
            "中午吃了好吃的",
        ],
        2: [
            "最近好累啊，什么都不想做",
            "感觉压力好大，喘不过气",
            "和朋友吵架了，心情很差",
            "失眠好几天了，睡不着",
            "觉得自己什么都做不好",
        ],
        3: [
            "活着好累，感觉没有任何意义",
            "我是废物，什么都做不好",
            "消失就好了，不想再面对了",
            "撑不住了，真的撑不下去了",
            "看不到希望，一切都是黑暗的",
        ],
        4: [
            "我不想活了，想结束这一切",
            "想死，活着太痛苦了",
            "我查了怎么自杀",
            "活不下去了，死了算了",
            "想从楼上跳下去",
        ],
        5: [
            "遗书已经写好了",
            "把我的东西都送给朋友了",
            "对不起大家，我不值得你们关心",
            "药已经买好了，准备好了",
            "这是我最后一次跟你们说话了",
        ],
        6: [
            "我已经站在楼顶了",
            "药已经吃了，再见了",
            "绳子已经准备好了，马上就",
            "永别了，来世再见",
            "正在割，终于要解脱了",
        ],
    }

    samples = []
    for level in range(1, 7):
        texts = demo_texts.get(level, [])
        for i in range(min(n_per_level, len(texts))):
            samples.append(CrisisSample(
                text=texts[i],
                severity=level,
                source="demo",
                line_index=i,
            ))

    distribution: dict[int, int] = {}
    for s in samples:
        distribution[s.severity] = distribution.get(s.severity, 0) + 1

    return CrisisDataset(
        samples=samples,
        source="demo",
        total=len(samples),
        distribution=distribution,
    )
