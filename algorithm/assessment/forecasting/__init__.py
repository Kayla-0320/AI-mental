"""短期本人情绪自评预测研究，独立于现有风险与升级逻辑。"""
from assessment.forecasting.predictor import ForecastPredictor, history_features, validate_history

__all__ = ["ForecastPredictor", "history_features", "validate_history"]
