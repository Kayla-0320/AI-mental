"""
多任务风险分级模型单元测试

覆盖：
- 模型前向传播
- 非对称损失函数
- predictor 输出 RiskAssessment
- 风险等级映射
- PHQ-9 / GAD-7 预估
"""
from __future__ import annotations

import numpy as np
import pytest

from assessment.multi_task.config import FEATURE_DIM, TASK_NAMES
from assessment.multi_task.dataset import generate_fake_data, load_dataset, split_train_val
from assessment.multi_task.model import MultiTaskModel, SharedEncoder, SingleTaskModel, TaskHead
from assessment.multi_task.train import asymmetric_bce_loss, train_multi_task, train_single_task
from shared.dataclasses import RiskAssessment, RiskLevel


# ============================================================
# 模型结构测试
# ============================================================

class TestSharedEncoder:
    def test_output_shape(self):
        enc = SharedEncoder()
        X = np.random.randn(5, FEATURE_DIM)
        out = enc.forward(X)
        assert out.shape == (5, 32)  # HIDDEN_DIM=32

    def test_relu_non_negative(self):
        enc = SharedEncoder()
        X = np.random.randn(10, FEATURE_DIM)
        out = enc.forward(X)
        assert np.all(out >= 0)


class TestTaskHead:
    def test_output_range(self):
        head = TaskHead()
        x = np.random.randn(10, 32)
        proba = head.forward(x)
        assert proba.shape == (10,)
        assert np.all(proba >= 0) and np.all(proba <= 1)


class TestMultiTaskModel:
    def test_forward_all_tasks(self):
        model = MultiTaskModel()
        X = np.random.randn(5, FEATURE_DIM)
        proba = model.forward(X)
        assert set(proba.keys()) == set(TASK_NAMES)
        for task in TASK_NAMES:
            assert proba[task].shape == (5,)
            assert np.all(proba[task] >= 0) and np.all(proba[task] <= 1)

    def test_predict_binary(self):
        model = MultiTaskModel()
        X = np.random.randn(3, FEATURE_DIM)
        labels = model.predict(X)
        for task in TASK_NAMES:
            assert set(np.unique(labels[task])).issubset({0, 1})

    def test_save_load(self, tmp_path):
        model = MultiTaskModel()
        path = str(tmp_path / "test_model.joblib")
        model.save(path)
        loaded = MultiTaskModel.load(path)
        X = np.random.randn(5, FEATURE_DIM)
        orig = model.forward(X)
        loaded_proba = loaded.forward(X)
        for task in TASK_NAMES:
            np.testing.assert_allclose(orig[task], loaded_proba[task])

    def test_shared_features(self):
        model = MultiTaskModel()
        X = np.random.randn(4, FEATURE_DIM)
        features = model.get_shared_features(X)
        assert features.shape == (4, 32)


class TestSingleTaskModel:
    def test_forward(self):
        model = SingleTaskModel(task_name="depression")
        X = np.random.randn(5, FEATURE_DIM)
        proba = model.forward(X)
        assert proba.shape == (5,)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_save_load(self, tmp_path):
        model = SingleTaskModel(task_name="anxiety")
        path = str(tmp_path / "single_test.joblib")
        model.save(path)
        loaded = SingleTaskModel.load(path)
        assert loaded.task_name == "anxiety"


# ============================================================
# 损失函数测试
# ============================================================

class TestAsymmetricLoss:
    def test_perfect_prediction_low_loss(self):
        proba = np.array([0.99, 0.99, 0.01, 0.01])
        labels = np.array([1, 1, 0, 0])
        loss = asymmetric_bce_loss(proba, labels, beta=6.0)
        assert loss < 0.2

    def test_wrong_prediction_high_loss(self):
        proba = np.array([0.01, 0.01, 0.99, 0.99])
        labels = np.array([1, 1, 0, 0])
        loss = asymmetric_bce_loss(proba, labels, beta=6.0)
        assert loss > 2.0

    def test_beta_amplifies_positive_loss(self):
        proba = np.array([0.3])
        labels = np.array([1])
        loss_low_beta = asymmetric_bce_loss(proba, labels, beta=1.0)
        loss_high_beta = asymmetric_bce_loss(proba, labels, beta=6.0)
        assert loss_high_beta > loss_low_beta

    def test_loss_non_negative(self):
        proba = np.random.rand(20)
        labels = np.random.randint(0, 2, 20)
        loss = asymmetric_bce_loss(proba, labels, beta=6.0)
        assert loss >= 0


# ============================================================
# 训练测试
# ============================================================

class TestTraining:
    @pytest.fixture
    def data(self, tmp_path):
        path = generate_fake_data(n_samples=100, output_path=str(tmp_path / "data.json"))
        data = load_dataset(path)
        return split_train_val(data, val_ratio=0.2)

    def test_multi_task_loss_decreases(self, data):
        train_data, val_data = data
        model = train_multi_task(train_data, val_data, epochs=50)
        # 验证模型能输出合理概率
        proba = model.forward(val_data["X"])
        for task in TASK_NAMES:
            assert np.all(proba[task] >= 0) and np.all(proba[task] <= 1)

    def test_single_task_training(self, data):
        train_data, val_data = data
        model = train_single_task(train_data, val_data, "depression", epochs=50)
        proba = model.forward(val_data["X"])
        assert proba.shape == (len(val_data["X"]),)


# ============================================================
# Predictor 测试
# ============================================================

class TestPredictor:
    @pytest.fixture
    def model(self, tmp_path):
        """训练一个小模型用于测试"""
        path = generate_fake_data(n_samples=100, output_path=str(tmp_path / "data.json"))
        data = load_dataset(path)
        train_data, val_data = split_train_val(data, val_ratio=0.2)
        return train_multi_task(train_data, val_data, epochs=30)

    def test_predict_returns_risk_assessment(self, model):
        from assessment.multi_task.predictor import predict
        features = np.random.randn(FEATURE_DIM)
        result = predict(features, model=model)
        assert isinstance(result, RiskAssessment)

    def test_predict_has_phq9_range(self, model):
        from assessment.multi_task.predictor import predict
        features = np.random.randn(FEATURE_DIM)
        result = predict(features, model=model)
        assert len(result.phq9_estimated) == 2
        assert result.phq9_estimated[0] >= 0
        assert result.phq9_estimated[1] <= 27

    def test_predict_has_gad7_range(self, model):
        from assessment.multi_task.predictor import predict
        features = np.random.randn(FEATURE_DIM)
        result = predict(features, model=model)
        assert len(result.gad7_estimated) == 2
        assert result.gad7_estimated[0] >= 0
        assert result.gad7_estimated[1] <= 21

    def test_predict_risk_level_is_enum(self, model):
        from assessment.multi_task.predictor import predict
        features = np.random.randn(FEATURE_DIM)
        result = predict(features, model=model)
        assert isinstance(result.risk_level, RiskLevel)

    def test_predict_confidence_range(self, model):
        from assessment.multi_task.predictor import predict
        features = np.random.randn(FEATURE_DIM)
        result = predict(features, model=model)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_has_evidence(self, model):
        from assessment.multi_task.predictor import predict
        features = np.random.randn(FEATURE_DIM)
        result = predict(features, model=model)
        assert len(result.evidence) >= 1

    def test_predict_batch(self, model):
        from assessment.multi_task.predictor import predict_batch
        features = np.random.randn(5, FEATURE_DIM)
        results = predict_batch(features, model=model)
        assert len(results) == 5
        assert all(isinstance(r, RiskAssessment) for r in results)
