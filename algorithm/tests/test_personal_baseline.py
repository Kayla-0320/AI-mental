"""
个人基线存储与偏离计算单元测试

覆盖：
- 数据结构序列化/反序列化
- EMA 算法（与前端 usePersonalBaseline.ts 一致）
- 基线同步（完整同步 + 增量更新）
- 偏离计算（Z-score）
- 显著偏离标记（|z| > 2）
- 数据缺失降级
- JSON 文件持久化
- FastAPI 路由集成
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from assessment.personal_baseline import (
    BaselineMetric,
    PersonalBaseline,
    BaselineDeviation,
    BaselineStore,
    _update_metric_ema,
    _update_emotion_ema,
    _calc_z_score,
    sync_baseline,
    compute_deviation,
    get_baseline,
    reset_store,
    load_baseline_config,
)


# ============================================================
# Fixture
# ============================================================

@pytest.fixture(autouse=True)
def clean_store():
    """每个测试前重置全局存储并清理持久化文件"""
    from assessment.personal_baseline import _get_store, _OUTPUT_DIR
    baselines_file = _OUTPUT_DIR / "baselines.json"
    # 清空现有存储的数据
    store = _get_store()
    store._data.clear()
    # 删除持久化文件
    if baselines_file.exists():
        baselines_file.unlink()
    yield
    # 测试后清理
    store._data.clear()
    if baselines_file.exists():
        baselines_file.unlink()


@pytest.fixture
def temp_storage_path():
    """临时存储路径"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_baselines.json"


# ============================================================
# 数据结构测试
# ============================================================

class TestBaselineMetric:
    def test_default_creation(self):
        """默认创建"""
        m = BaselineMetric()
        assert m.mean == 0.0
        assert m.std == 0.0
        assert m.n == 0

    def test_with_values(self):
        """带值创建"""
        m = BaselineMetric(mean=72.5, std=3.2, n=15)
        assert m.mean == 72.5
        assert m.std == 3.2
        assert m.n == 15


class TestPersonalBaseline:
    def test_default_creation(self):
        """默认创建"""
        bl = PersonalBaseline(user_id="user_001")
        assert bl.user_id == "user_001"
        assert bl.metrics == {}
        assert bl.emotion_baseline == [0.2, 0.2, 0.2, 0.2, 0.2]
        assert bl.sample_count == 0

    def test_to_dict(self):
        """序列化为字典"""
        bl = PersonalBaseline(
            user_id="user_001",
            metrics={"heartRate": BaselineMetric(mean=72.0, std=3.0, n=10)},
            emotion_baseline=[0.3, 0.2, 0.2, 0.1, 0.2],
            created_at=1000.0,
            last_updated=2000.0,
            sample_count=10,
        )
        d = bl.to_dict()
        assert d["user_id"] == "user_001"
        assert d["metrics"]["heartRate"]["mean"] == 72.0
        assert d["metrics"]["heartRate"]["std"] == 3.0
        assert d["metrics"]["heartRate"]["n"] == 10
        assert d["emotion_baseline"] == [0.3, 0.2, 0.2, 0.1, 0.2]
        assert d["sample_count"] == 10

    def test_from_dict(self):
        """从字典反序列化"""
        data = {
            "user_id": "user_002",
            "metrics": {
                "heartRate": {"mean": 68.0, "std": 2.5, "n": 20},
                "breathingRate": {"mean": 15.0, "std": 1.2, "n": 20},
            },
            "emotion_baseline": [0.25, 0.25, 0.2, 0.15, 0.15],
            "created_at": 1000.0,
            "last_updated": 3000.0,
            "sample_count": 20,
        }
        bl = PersonalBaseline.from_dict(data)
        assert bl.user_id == "user_002"
        assert bl.metrics["heartRate"].mean == 68.0
        assert bl.metrics["breathingRate"].std == 1.2
        assert bl.emotion_baseline[0] == 0.25
        assert bl.sample_count == 20

    def test_roundtrip(self):
        """序列化-反序列化往返"""
        original = PersonalBaseline(
            user_id="user_rt",
            metrics={
                "heartRate": BaselineMetric(mean=75.0, std=4.0, n=30),
                "typingSpeed": BaselineMetric(mean=120.0, std=15.0, n=30),
            },
            emotion_baseline=[0.1, 0.3, 0.3, 0.1, 0.2],
            created_at=100.0,
            last_updated=200.0,
            sample_count=30,
        )
        restored = PersonalBaseline.from_dict(original.to_dict())
        assert restored.user_id == original.user_id
        assert restored.metrics["heartRate"].mean == original.metrics["heartRate"].mean
        assert restored.metrics["typingSpeed"].std == original.metrics["typingSpeed"].std
        assert restored.emotion_baseline == original.emotion_baseline
        assert restored.sample_count == original.sample_count


class TestBaselineDeviation:
    def test_default_creation(self):
        """默认创建"""
        dev = BaselineDeviation(user_id="user_001")
        assert dev.user_id == "user_001"
        assert dev.modality_z_scores == {}
        assert dev.emotion_z_scores == []
        assert dev.significant_deviations == []
        assert dev.confidence == 0.0
        assert dev.calibrated is False

    def test_to_dict(self):
        """序列化为字典"""
        dev = BaselineDeviation(
            user_id="user_001",
            computed_at=1234.0,
            modality_z_scores={"heartRate": 2.5, "breathingRate": -1.2},
            emotion_z_scores=[0.5, -0.3, 2.1, 0.0, -0.8],
            significant_deviations=["heartRate"],
            confidence=0.75,
            calibrated=True,
        )
        d = dev.to_dict()
        assert d["user_id"] == "user_001"
        assert d["modality_z_scores"]["heartRate"] == 2.5
        assert d["significant_deviations"] == ["heartRate"]
        assert d["calibrated"] is True


# ============================================================
# EMA 算法测试
# ============================================================

class TestEMAAlgorithm:
    def test_first_sample_initialization(self):
        """第一个样本直接初始化"""
        m = BaselineMetric()
        updated = _update_metric_ema(m, 72.0, alpha=0.05)
        assert updated.mean == 72.0
        assert updated.std == 0.0
        assert updated.n == 1

    def test_ema_mean_update(self):
        """EMA 均值更新"""
        m = BaselineMetric(mean=70.0, std=2.0, n=10)
        updated = _update_metric_ema(m, 80.0, alpha=0.05)
        # new_mean = 0.05 * 80 + 0.95 * 70 = 4 + 66.5 = 70.5
        expected_mean = 0.05 * 80.0 + 0.95 * 70.0
        assert abs(updated.mean - round(expected_mean, 4)) < 0.01
        assert updated.n == 11

    def test_ema_variance_update(self):
        """EMA 方差更新"""
        m = BaselineMetric(mean=70.0, std=2.0, n=10)
        updated = _update_metric_ema(m, 80.0, alpha=0.05)
        # new_variance = 0.05 * (80-70)^2 + 0.95 * 4 = 0.05*100 + 3.8 = 8.8
        expected_var = 0.05 * (80.0 - 70.0) ** 2 + 0.95 * (2.0 ** 2)
        expected_std = math.sqrt(expected_var)
        assert abs(updated.std - round(expected_std, 4)) < 0.01

    def test_ema_stability(self):
        """EMA 稳定性：多次更新后均值趋向观测值"""
        m = BaselineMetric()
        for i in range(100):
            m = _update_metric_ema(m, 75.0, alpha=0.05)
        # 100 次相同值后，均值应接近 75
        assert abs(m.mean - 75.0) < 0.5

    def test_emotion_ema_update(self):
        """情绪 EMA 更新"""
        baseline = [0.2, 0.2, 0.2, 0.2, 0.2]
        probs = [0.1, 0.3, 0.4, 0.1, 0.1]
        updated = _update_emotion_ema(baseline, probs, alpha=0.05)
        assert len(updated) == 5
        # 第一个维度：0.05 * 0.1 + 0.95 * 0.2 = 0.005 + 0.19 = 0.195
        expected_0 = 0.05 * 0.1 + 0.95 * 0.2
        assert abs(updated[0] - round(expected_0, 4)) < 0.001

    def test_emotion_ema_invalid_length(self):
        """情绪 EMA 无效长度不更新"""
        baseline = [0.2, 0.2, 0.2, 0.2, 0.2]
        result = _update_emotion_ema(baseline, [0.5, 0.5], alpha=0.05)
        assert result == baseline


class TestZScore:
    def test_insufficient_samples(self):
        """样本不足时返回 0"""
        m = BaselineMetric(mean=70.0, std=2.0, n=1)
        assert _calc_z_score(m, 80.0) == 0.0

    def test_zero_std(self):
        """标准差为 0 时返回 0"""
        m = BaselineMetric(mean=70.0, std=0.0, n=10)
        assert _calc_z_score(m, 80.0) == 0.0

    def test_normal_z_score(self):
        """正常 Z-score 计算"""
        m = BaselineMetric(mean=70.0, std=2.0, n=10)
        z = _calc_z_score(m, 74.0)
        assert abs(z - 2.0) < 0.01

    def test_negative_z_score(self):
        """负 Z-score"""
        m = BaselineMetric(mean=70.0, std=2.0, n=10)
        z = _calc_z_score(m, 66.0)
        assert abs(z - (-2.0)) < 0.01


# ============================================================
# 存储层测试
# ============================================================

class TestBaselineStore:
    def test_put_and_get(self, temp_storage_path):
        """存储和获取"""
        store = BaselineStore(storage_path=temp_storage_path)
        bl = PersonalBaseline(
            user_id="user_001",
            metrics={"heartRate": BaselineMetric(mean=72.0, std=3.0, n=10)},
            sample_count=10,
        )
        store.put(bl)
        retrieved = store.get("user_001")
        assert retrieved is not None
        assert retrieved.user_id == "user_001"
        assert retrieved.metrics["heartRate"].mean == 72.0

    def test_get_nonexistent(self, temp_storage_path):
        """获取不存在的用户"""
        store = BaselineStore(storage_path=temp_storage_path)
        assert store.get("nonexistent") is None

    def test_delete(self, temp_storage_path):
        """删除用户基线"""
        store = BaselineStore(storage_path=temp_storage_path)
        bl = PersonalBaseline(user_id="user_001")
        store.put(bl)
        assert store.delete("user_001") is True
        assert store.get("user_001") is None

    def test_delete_nonexistent(self, temp_storage_path):
        """删除不存在的用户"""
        store = BaselineStore(storage_path=temp_storage_path)
        assert store.delete("nonexistent") is False

    def test_list_users(self, temp_storage_path):
        """列出所有用户"""
        store = BaselineStore(storage_path=temp_storage_path)
        store.put(PersonalBaseline(user_id="user_001"))
        store.put(PersonalBaseline(user_id="user_002"))
        store.put(PersonalBaseline(user_id="user_003"))
        users = store.list_users()
        assert len(users) == 3
        assert "user_001" in users
        assert "user_002" in users

    def test_json_persistence(self, temp_storage_path):
        """JSON 文件持久化"""
        store = BaselineStore(storage_path=temp_storage_path)
        bl = PersonalBaseline(
            user_id="user_persist",
            metrics={"heartRate": BaselineMetric(mean=68.0, std=2.5, n=15)},
            emotion_baseline=[0.3, 0.2, 0.2, 0.15, 0.15],
            sample_count=15,
        )
        store.put(bl)

        # 验证文件存在
        assert temp_storage_path.exists()

        # 从文件重新加载
        store2 = BaselineStore(storage_path=temp_storage_path)
        retrieved = store2.get("user_persist")
        assert retrieved is not None
        assert retrieved.metrics["heartRate"].mean == 68.0
        assert retrieved.emotion_baseline[0] == 0.3


# ============================================================
# 核心接口测试
# ============================================================

class TestSyncBaseline:
    def test_incremental_update(self):
        """增量更新模式"""
        result = sync_baseline("user_001", {
            "heartRate": 72.0,
            "breathingRate": 16.0,
        })
        assert result.user_id == "user_001"
        assert "heartRate" in result.metrics
        assert result.metrics["heartRate"].mean == 72.0
        assert result.sample_count == 1

    def test_multiple_incremental_updates(self):
        """多次增量更新"""
        for _ in range(5):
            sync_baseline("user_002", {"heartRate": 70.0})
        baseline = get_baseline("user_002")
        assert baseline is not None
        assert baseline.sample_count == 5
        assert baseline.metrics["heartRate"].n == 5

    def test_full_sync(self):
        """完整同步模式"""
        full_data = {
            "metrics": {
                "heartRate": {"mean": 68.0, "std": 2.5, "n": 20},
                "breathingRate": {"mean": 15.0, "std": 1.0, "n": 20},
            },
            "emotion_baseline": [0.25, 0.25, 0.2, 0.15, 0.15],
            "sample_count": 20,
        }
        result = sync_baseline("user_003", full_data)
        assert result.metrics["heartRate"].mean == 68.0
        assert result.emotion_baseline[0] == 0.25
        assert result.sample_count == 20

    def test_emotion_probs_update(self):
        """情绪概率增量更新"""
        sync_baseline("user_004", {
            "heartRate": 72.0,
            "emotionProbs": [0.1, 0.3, 0.4, 0.1, 0.1],
        })
        baseline = get_baseline("user_004")
        assert baseline is not None
        # 情绪基线应该被更新
        assert baseline.emotion_baseline[2] > 0.2  # 焦虑维度增加

    def test_invalid_values_skipped(self):
        """无效值被跳过"""
        result = sync_baseline("user_005", {
            "heartRate": 72.0,
            "invalidKey": -5.0,  # 负值应被跳过
            "zeroValue": 0.0,    # 零值应被跳过
        })
        assert "heartRate" in result.metrics
        assert "invalidKey" not in result.metrics
        assert "zeroValue" not in result.metrics


class TestComputeDeviation:
    def test_no_baseline(self):
        """无基线时返回空结果"""
        deviation = compute_deviation("nonexistent", {"heartRate": 72.0})
        assert deviation.user_id == "nonexistent"
        assert deviation.modality_z_scores == {}
        assert deviation.calibrated is False

    def test_normal_deviation(self):
        """正常偏离计算"""
        # 先建立基线
        for _ in range(25):
            sync_baseline("user_dev", {"heartRate": 70.0})

        # 计算偏离（当前值 70，与基线均值接近）
        deviation = compute_deviation("user_dev", {"heartRate": 70.0})
        assert "heartRate" in deviation.modality_z_scores
        # Z-score 应该接近 0
        assert abs(deviation.modality_z_scores["heartRate"]) < 1.0

    def test_significant_deviation(self):
        """显著偏离标记"""
        # 建立有波动的心率基线（模拟真实心率波动）
        import random
        random.seed(42)
        for _ in range(30):
            hr = 65.0 + random.gauss(0, 3)  # 均值 65，标准差约 3
            sync_baseline("user_sig", {"heartRate": hr})

        # 当前心率显著偏高（85 远超基线均值 65±3）
        deviation = compute_deviation("user_sig", {"heartRate": 85.0})
        # 应该有显著偏离
        assert deviation.confidence == 1.0  # 30 样本已满置信度
        # Z-score 应该较大（85 - 65 ≈ 20，std ≈ 3，z ≈ 6+）
        z = deviation.modality_z_scores.get("heartRate", 0)
        assert abs(z) > 2.0, f"Expected |z| > 2.0, got z={z}"

    def test_emotion_deviation(self):
        """情绪偏离计算"""
        # 建立基线
        for _ in range(25):
            sync_baseline("user_emo", {
                "heartRate": 70.0,
                "emotionProbs": [0.3, 0.2, 0.1, 0.1, 0.3],
            })

        # 计算情绪偏离（高焦虑）
        deviation = compute_deviation("user_emo", {
            "emotionProbs": [0.05, 0.15, 0.6, 0.1, 0.1],
        })
        assert len(deviation.emotion_z_scores) == 5
        # 焦虑维度（索引2）应该有正 Z-score
        assert deviation.emotion_z_scores[2] > 0

    def test_missing_modality(self):
        """缺失模态降级"""
        # 建立基线（只有心率）
        sync_baseline("user_miss", {"heartRate": 70.0})

        # 请求计算不存在的模态偏离
        deviation = compute_deviation("user_miss", {
            "heartRate": 70.0,
            "breathingRate": 16.0,  # 基线中没有此模态
        })
        # 心率应该有结果
        assert "heartRate" in deviation.modality_z_scores
        # 呼吸率不在基线中，不应出现在结果中
        assert "breathingRate" not in deviation.modality_z_scores

    def test_confidence_based_on_samples(self):
        """置信度基于样本数"""
        # 5 个样本 → 置信度 = 5/20 = 0.25
        for _ in range(5):
            sync_baseline("user_conf", {"heartRate": 70.0})
        deviation = compute_deviation("user_conf", {"heartRate": 70.0})
        assert abs(deviation.confidence - 0.25) < 0.01
        assert deviation.calibrated is False

        # 20 个样本 → 置信度 = 1.0
        for _ in range(15):
            sync_baseline("user_conf", {"heartRate": 70.0})
        deviation = compute_deviation("user_conf", {"heartRate": 70.0})
        assert deviation.confidence == 1.0
        assert deviation.calibrated is True


class TestGetBaseline:
    def test_get_existing(self):
        """获取已有基线"""
        sync_baseline("user_get", {"heartRate": 72.0})
        baseline = get_baseline("user_get")
        assert baseline is not None
        assert baseline.user_id == "user_get"

    def test_get_nonexistent(self):
        """获取不存在的基线"""
        baseline = get_baseline("nonexistent_user")
        assert baseline is None


# ============================================================
# 配置测试
# ============================================================

class TestConfig:
    def test_load_config(self):
        """加载配置"""
        config = load_baseline_config()
        assert "ema" in config
        assert "alpha" in config["ema"]
        assert config["ema"]["alpha"] == 0.05

    def test_config_thresholds(self):
        """配置阈值"""
        config = load_baseline_config()
        assert config["deviation"]["significant_threshold"] == 2.0
        assert config["calibration"]["min_samples"] == 20


# ============================================================
# FastAPI 路由测试
# ============================================================

class TestAPIRoutes:
    """FastAPI 路由集成测试"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        from fastapi.testclient import TestClient
        from main import app
        reset_store()
        with TestClient(app) as c:
            yield c

    def test_sync_baseline(self, client):
        """POST /assessment/baseline/sync"""
        response = client.post("/api/v1/assessment/baseline/sync", json={
            "user_id": "api_user_001",
            "baseline_data": {"heartRate": 72.0, "breathingRate": 16.0},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "api_user_001"
        assert "heartRate" in data["metrics"]
        assert data["sample_count"] == 1

    def test_get_baseline(self, client):
        """GET /assessment/baseline/{user_id}"""
        # 先同步
        client.post("/api/v1/assessment/baseline/sync", json={
            "user_id": "api_user_002",
            "baseline_data": {"heartRate": 70.0},
        })
        # 获取
        response = client.get("/api/v1/assessment/baseline/api_user_002")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "api_user_002"

    def test_get_baseline_not_found(self, client):
        """GET /assessment/baseline/{user_id} 404"""
        response = client.get("/api/v1/assessment/baseline/nonexistent")
        assert response.status_code == 404

    def test_compute_deviation(self, client):
        """POST /assessment/baseline/deviation"""
        # 先建立基线
        for _ in range(25):
            client.post("/api/v1/assessment/baseline/sync", json={
                "user_id": "api_user_003",
                "baseline_data": {"heartRate": 70.0},
            })
        # 计算偏离
        response = client.post("/api/v1/assessment/baseline/deviation", json={
            "user_id": "api_user_003",
            "current_features": {"heartRate": 70.0},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "api_user_003"
        assert "heartRate" in data["modality_z_scores"]
        assert data["calibrated"] is True

    def test_full_sync_via_api(self, client):
        """完整同步通过 API"""
        response = client.post("/api/v1/assessment/baseline/sync", json={
            "user_id": "api_user_full",
            "baseline_data": {
                "metrics": {
                    "heartRate": {"mean": 68.0, "std": 2.5, "n": 20},
                },
                "emotion_baseline": [0.3, 0.2, 0.2, 0.15, 0.15],
                "sample_count": 20,
            },
        })
        assert response.status_code == 200
        data = response.json()
        assert data["calibrated"] is True
        assert data["sample_count"] == 20
