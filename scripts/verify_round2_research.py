"""用真实训练工件核对第二轮研究 HTTP 接口；仅在进程内启用测试令牌。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm"))

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.research import router


def run(data_dir: Path, output: Path) -> None:
    """验证实际权重预测、全缺失弃权、任务/指纹拒绝和两种预测跨度。"""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)
    os.environ["ROUND2_RESEARCH_TOKEN"] = "in-process-verification-only"
    headers = {"Authorization": "Bearer in-process-verification-only"}
    meta = json.loads((ROOT / "algorithm/artifacts/round2_fusion/manifest.json").read_text(encoding="utf-8"))
    names = list(meta["dimensions"])
    with np.load(data_dir / "test.npz", allow_pickle=False) as archive:
        body = {"sample_ids": archive["sample_ids"][:3].tolist(),
                "modalities": {name: archive[name][:3].tolist() for name in names},
                "observed": {name: [True] * 3 for name in names},
                "quality": {name: [1.0] * 3 for name in names},
                "task": meta["task"], "class_names": meta["class_names"],
                "encoder_fingerprints": meta["encoder_fingerprints"]}
    response = client.post("/api/v1/research/fusion", json=body, headers=headers)
    assert response.status_code == 200, response.text
    probabilities = np.array(response.json()["probabilities"])
    assert probabilities.shape == (3, 3) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1)
    valid_result = response.json()
    for name in names:
        body["observed"][name][2] = False
        body["quality"][name][2] = 0.0
    result = client.post("/api/v1/research/fusion", json=body, headers=headers)
    assert result.status_code == 200 and result.json()["abstained"][2]
    assert result.json()["probabilities"][2] == []
    wrong_task = client.post("/api/v1/research/fusion", json={**body, "task": "patient_emotions_5"}, headers=headers)
    assert wrong_task.status_code == 422
    original = body["encoder_fingerprints"][names[0]]
    body["encoder_fingerprints"][names[0]] = "mismatched-encoder"
    wrong_encoder = client.post("/api/v1/research/fusion", json=body, headers=headers)
    assert wrong_encoder.status_code == 422
    body["encoder_fingerprints"][names[0]] = original
    history = [{"step": i, "positive": 60 + i, "negative": 30 - i} for i in range(1, 9)]
    forecasts = {}
    for horizon in (1, 3):
        response = client.post("/api/v1/research/forecast", json={"history": history, "horizon": horizon}, headers=headers)
        assert response.status_code == 200, response.text
        forecasts[str(horizon)] = response.json()
    unauthorized = client.post("/api/v1/research/fusion", json=body)
    assert unauthorized.status_code == 401
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "passed", "real_trained_weights": True,
                                  "transport": "FastAPI in-process TestClient; not external deployment",
                                  "fusion_prediction": valid_result, "forecasts": forecasts,
                                  "all_missing_abstained": True, "wrong_task_rejected": True,
                                  "wrong_encoder_rejected": True, "missing_token_rejected": True},
                                 ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("real research artifact HTTP verification passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.output)
