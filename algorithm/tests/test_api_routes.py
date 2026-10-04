"""
API 路由端点单元测试 —— audit / federated / privacy

覆盖：
- 正常输入 → 200
- 缺失字段 → 400/422
- 异常场景 → 404/409/500
"""
from __future__ import annotations

import base64
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================
# 辅助数据
# ============================================================

def _demo_demographics(n=100):
    """构造人口统计学数据"""
    return {
        "ages": [15 + (i % 4) for i in range(n)],
        "genders": ["male" if i % 2 == 0 else "female" for i in range(n)],
        "regions": [["urban", "town", "rural"][i % 3] for i in range(n)],
    }


# ============================================================
# audit/fairness 端点测试
# ============================================================

class TestAuditFairnessAPI:
    """POST /api/v1/audit/fairness"""

    def test_normal_input_returns_200(self, client):
        n = 60
        predictions = [i % 2 for i in range(n)]
        labels = [(i + 1) % 2 for i in range(n)]
        demos = _demo_demographics(n)

        resp = client.post("/api/v1/audit/fairness", json={
            "predictions": predictions,
            "labels": labels,
            "demographics": demos,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "report_id" in data
        assert "demographic_parity_diff" in data
        assert "group_metrics" in data
        assert len(data["group_metrics"]) > 0

    def test_mismatched_lengths_returns_400(self, client):
        resp = client.post("/api/v1/audit/fairness", json={
            "predictions": [0, 1, 0],
            "labels": [0, 1],  # 长度不同
            "demographics": {
                "ages": [15, 16, 17],
                "genders": ["male", "female", "male"],
                "regions": ["urban", "town", "rural"],
            },
        })
        assert resp.status_code == 400

    def test_missing_demographics_field_returns_400(self, client):
        resp = client.post("/api/v1/audit/fairness", json={
            "predictions": [0, 1],
            "labels": [0, 1],
            "demographics": {"ages": [15, 16]},  # 缺少 genders, regions
        })
        assert resp.status_code == 400

    def test_missing_body_returns_422(self, client):
        resp = client.post("/api/v1/audit/fairness", json={})
        assert resp.status_code == 422


class TestAuditReportAPI:
    """GET /api/v1/audit/report/{report_id}"""

    def test_existing_report_returns_200(self, client):
        # 先创建报告
        n = 60
        resp = client.post("/api/v1/audit/fairness", json={
            "predictions": [i % 2 for i in range(n)],
            "labels": [(i + 1) % 2 for i in range(n)],
            "demographics": _demo_demographics(n),
        })
        report_id = resp.json()["report_id"]

        resp2 = client.get(f"/api/v1/audit/report/{report_id}")
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["report_id"] == report_id
        assert "content" in data
        assert "# 公平性审计报告" in data["content"]

    def test_nonexistent_report_returns_404(self, client):
        resp = client.get("/api/v1/audit/report/nonexistent")
        assert resp.status_code == 404


class TestAuditChartAPI:
    """GET /api/v1/audit/chart/{report_id}"""

    def test_existing_chart_returns_200(self, client):
        n = 60
        resp = client.post("/api/v1/audit/fairness", json={
            "predictions": [i % 2 for i in range(n)],
            "labels": [(i + 1) % 2 for i in range(n)],
            "demographics": _demo_demographics(n),
        })
        report_id = resp.json()["report_id"]

        resp2 = client.get(f"/api/v1/audit/chart/{report_id}")
        assert resp2.status_code == 200
        assert resp2.headers["content-type"] == "image/png"

    def test_nonexistent_chart_returns_404(self, client):
        resp = client.get("/api/v1/audit/chart/nonexistent")
        assert resp.status_code == 404


# ============================================================
# federated 端点测试
# ============================================================

class TestFederatedStatusAPI:
    """GET /api/v1/federated/status"""

    def test_status_returns_200(self, client):
        resp = client.get("/api/v1/federated/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "initialized" in data
        assert "num_clients" in data
        assert "client_ids" in data
        assert "has_global_model" in data


class TestFederatedAggregateAPI:
    """POST /api/v1/federated/aggregate"""

    def test_aggregate_returns_200(self, client):
        resp = client.post("/api/v1/federated/aggregate", json={
            "num_rounds": 2,
            "n_samples_per_client": 100,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["num_rounds"] == 2
        assert "final_accuracy" in data
        assert "final_f1" in data
        assert len(data["round_results"]) == 2

    def test_aggregate_default_params(self, client):
        resp = client.post("/api/v1/federated/aggregate", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_rounds"] == 5


class TestFederatedRegisterAPI:
    """POST /api/v1/federated/register"""

    def test_register_returns_200(self, client):
        resp = client.post("/api/v1/federated/register", json={
            "client_id": "test_client_001",
            "n_samples": 200,
            "n_features": 8,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["client_id"] == "test_client_001"
        assert data["status"] == "registered"
        assert data["n_samples"] == 200

    def test_duplicate_register_returns_409(self, client):
        client.post("/api/v1/federated/register", json={
            "client_id": "dup_client",
        })
        resp = client.post("/api/v1/federated/register", json={
            "client_id": "dup_client",
        })
        assert resp.status_code == 409


# ============================================================
# privacy 端点测试
# ============================================================

class TestPrivacyAnonymizeAPI:
    """POST /api/v1/privacy/anonymize"""

    def test_normal_input_returns_200(self, client):
        records = [
            {"user_id": f"u{i}", "age": 14 + (i % 5), "gender": "male" if i % 2 == 0 else "female",
             "region": ["urban", "town", "rural"][i % 3], "risk_score": 30 + i}
            for i in range(20)
        ]
        resp = client.post("/api/v1/privacy/anonymize", json={
            "records": records,
            "k_anonymity": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["original_count"] == 20
        assert data["anonymized_count"] > 0

    def test_empty_records_returns_400(self, client):
        resp = client.post("/api/v1/privacy/anonymize", json={
            "records": [],
        })
        assert resp.status_code == 400

    def test_missing_records_returns_422(self, client):
        resp = client.post("/api/v1/privacy/anonymize", json={})
        assert resp.status_code == 422


class TestPrivacyConsentAPI:
    """POST /api/v1/privacy/consent"""

    def test_create_consent_returns_200(self, client):
        resp = client.post("/api/v1/privacy/consent", json={
            "user_id": "user_consent_test",
            "items": [
                {"data_type": "text", "purpose": "emotion_analysis", "granted": True, "description": "文本情感分析"},
                {"data_type": "voice", "purpose": "risk_assessment", "granted": False, "description": "语音风险评估"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "user_consent_test"
        assert data["status"] == "active"
        assert len(data["items"]) == 2

    def test_empty_items_returns_400(self, client):
        resp = client.post("/api/v1/privacy/consent", json={
            "user_id": "user_test",
            "items": [],
        })
        assert resp.status_code == 400

    def test_invalid_enum_returns_400(self, client):
        resp = client.post("/api/v1/privacy/consent", json={
            "user_id": "user_test",
            "items": [{"data_type": "invalid_type", "purpose": "emotion_analysis", "granted": True}],
        })
        assert resp.status_code == 400


class TestPrivacyConsentSummaryAPI:
    """GET /api/v1/privacy/consent/{user_id}"""

    def test_summary_returns_200(self, client):
        # 先创建同意记录
        client.post("/api/v1/privacy/consent", json={
            "user_id": "summary_user",
            "items": [
                {"data_type": "text", "purpose": "emotion_analysis", "granted": True},
            ],
        })
        resp = client.get("/api/v1/privacy/consent/summary_user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "summary_user"
        assert data["active_records"] >= 1

    def test_nonexistent_user_returns_200_empty(self, client):
        resp = client.get("/api/v1/privacy/consent/no_such_user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == 0


class TestPrivacyEncryptAPI:
    """POST /api/v1/privacy/encrypt"""

    def test_encrypt_auto_key_returns_200(self, client):
        resp = client.post("/api/v1/privacy/encrypt", json={
            "plaintext": "Hello, this is sensitive data.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["ciphertext"] != ""
        assert data["algorithm"] == "AES-256-GCM"
        assert data["generated_key"] is not None

    def test_encrypt_with_provided_key_returns_200(self, client):
        key_b64 = base64.b64encode(b"\x00" * 32).decode()
        resp = client.post("/api/v1/privacy/encrypt", json={
            "plaintext": "test data",
            "key": key_b64,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["generated_key"] is None

    def test_empty_plaintext_returns_400(self, client):
        resp = client.post("/api/v1/privacy/encrypt", json={
            "plaintext": "",
        })
        assert resp.status_code == 400

    def test_missing_plaintext_returns_422(self, client):
        resp = client.post("/api/v1/privacy/encrypt", json={})
        assert resp.status_code == 422

    def test_invalid_key_returns_400(self, client):
        resp = client.post("/api/v1/privacy/encrypt", json={
            "plaintext": "test",
            "key": "not-valid-base64!!!",
        })
        assert resp.status_code == 400
