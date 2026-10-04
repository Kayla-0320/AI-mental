"""
训练配置 —— 集中管理所有超参数和路径

随机种子固定为 42，确保结果可复现。
"""
from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# 随机种子
# ============================================================
SEED = 42

# ============================================================
# 模型与分词器
# ============================================================
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_LENGTH = 128            # 文本最大 token 长度

# ============================================================
# 类别定义（5 类情感）
# ============================================================
LABEL2ID = {
    "anxiety": 0,           # 焦虑
    "depression": 1,        # 抑郁
    "anger": 2,             # 愤怒
    "neutral": 3,           # 中性
    "positive": 4,          # 积极
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ============================================================
# 训练超参数
# ============================================================
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
VALIDATION_SPLIT = 0.2      # 验证集比例

# ============================================================
# 路径配置
# ============================================================
_MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = _MODULE_DIR / "data"
OUTPUT_DIR = _MODULE_DIR / "outputs"

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 训练产物路径
MODEL_SAVE_PATH = str(OUTPUT_DIR / "model")
METRICS_SAVE_PATH = str(OUTPUT_DIR / "metrics.json")
CONFUSION_MATRIX_PATH = str(OUTPUT_DIR / "confusion_matrix.png")

# ============================================================
# 设备
# ============================================================
DEVICE = "cuda" if os.environ.get("USE_CPU") != "1" else "cpu"
