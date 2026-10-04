"""
多任务评估模型配置
"""
from pathlib import Path

# 随机种子
SEED = 42

# 数据参数
FEATURE_DIM = 20          # 感知层输出特征维度
N_SAMPLES = 1000          # 模拟数据样本数（扩大以提升训练稳定性）
TASK_NAMES = ["depression", "anxiety", "sleep"]

# 模型参数
HIDDEN_DIM = 32           # 共享编码器隐藏层维度
TASK_HIDDEN_DIM = 16      # 任务头隐藏层维度
LEARNING_RATE = 0.005
EPOCHS = 200
BATCH_SIZE = 32
VAL_RATIO = 0.2

# 非对称损失权重
# β=6.0：优先保障高风险样本召回（来源：临床筛查中假阴性代价远大于假阳性）
BETA = 6.0
# 任务权重（可调整各任务重要性）
TASK_WEIGHTS = {
    "depression": 1.0,
    "anxiety": 1.0,
    "sleep": 1.0,
}

# 路径
_MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = _MODULE_DIR / "data"
OUTPUT_DIR = _MODULE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = str(OUTPUT_DIR / "multi_task_model.joblib")
SINGLE_TASK_SAVE_DIR = OUTPUT_DIR / "single_task"
SINGLE_TASK_SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = str(OUTPUT_DIR / "comparison_report.md")
