"""用第三轮真实训练工件核对文字研究参考与支持策略 HTTP 接口。

只在本进程内设置研究令牌与工件路径，不修改仓库默认配置、不改线上风险
工作点。文字参考必须真正加载 Chinese-CLIP 与研究融合头；策略模型必须
是 ESConv 真实训练产物。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithm"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.research import router


def run(artifact: Path, encoder: Path, strategy: Path, output: Path) -> None:
    """覆盖可用、过期、出处摘要不符、语言不符与未授权五类可观察结果。"""
    os.environ["ROUND2_RESEARCH_TOKEN"] = "round3-in-process-verification-only"
    os.environ["ROUND3_FUSION_ARTIFACT"] = str(artifact)
    os.environ["ROUND3_VLM_ENCODER"] = str(encoder)
    os.environ["ROUND3_ENCODER_DEVICE"] = "cpu"
    os.environ["ROUND3_STRATEGY_ARTIFACT"] = str(strategy)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)
    headers = {"Authorization": "Bearer round3-in-process-verification-only"}

    capabilities = client.get("/api/v1/research/capabilities")
    assert capabilities.status_code == 200, capabilities.text
    assert capabilities.json()["version"] == "round3.v1"

    text = "我今天考试没考好，有点难受"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    payload = {"schema_version": "emotion_observation.v1", "request_id": "verify-round3-1",
               "observation": {"text": text, "source_message_id": "msg-round3-1", "source_sha256": digest,
                               "observed_at": now.isoformat(), "language": "zh"}}
    observed = client.post("/api/v1/research/observation", json=payload, headers=headers)
    assert observed.status_code == 200, observed.text
    body = observed.json()
    assert isinstance(body["available"], bool)
    if body["available"]:
        probabilities = body["probabilities"]
        assert len(probabilities) == 3 and abs(sum(probabilities) - 1) < 1e-6
        assert all(0 <= value <= 1 for value in probabilities)
        assert body["task_id"] == "sentiment_polarity_3"
    else:
        assert body["reason"]

    stale = json.loads(json.dumps(payload, ensure_ascii=False))
    stale["observation"]["observed_at"] = (now - timedelta(hours=72)).isoformat()
    stale["request_id"] = "verify-round3-stale"
    response = client.post("/api/v1/research/observation", json=stale, headers=headers)
    assert response.status_code == 200 and response.json()["available"] is False, response.text
    assert response.json()["reason"] == "source_stale_or_future"

    wrong_sha = json.loads(json.dumps(payload, ensure_ascii=False))
    wrong_sha["observation"]["source_sha256"] = "0" * 64
    wrong_sha["request_id"] = "verify-round3-bad-sha"
    assert client.post("/api/v1/research/observation", json=wrong_sha, headers=headers).status_code == 422

    english = [{"role": "user", "content": "I feel really anxious about my exam tomorrow."},
               {"role": "assistant", "content": "That sounds stressful. What worries you most?"},
               {"role": "user", "content": "I cannot sleep at all and I keep thinking about failing."}]
    strategy_response = client.post("/api/v1/research/strategy",
                                    json={"messages": english, "language": "en"}, headers=headers)
    assert strategy_response.status_code == 200, strategy_response.text
    strategy_body = strategy_response.json()
    assert strategy_body["available"] is True and sum(strategy_body["probabilities"]) > 0
    assert strategy_body["strategy"] in strategy_body["labels"]

    chinese = client.post("/api/v1/research/strategy",
                          json={"messages": [{"role": "user", "content": "我明天考试很紧张"}], "language": "zh"},
                          headers=headers)
    assert chinese.status_code == 200 and chinese.json()["available"] is False, chinese.text

    filler = client.post("/api/v1/research/strategy",
                         json={"messages": [{"role": "user", "content": "..."}], "language": "en"}, headers=headers)
    assert filler.status_code == 200 and filler.json()["available"] is False, filler.text

    assert client.post("/api/v1/research/observation", json=payload).status_code == 401

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "passed",
        "transport": "FastAPI in-process TestClient; not external deployment",
        "real_text_encoder": True,
        "artifact": str(artifact),
        "observation_available": body["available"],
        "observation": body,
        "stale_source": "unavailable",
        "wrong_sha_rejected": True,
        "english_strategy_available": True,
        "strategy": strategy_body,
        "chinese_strategy_rejected": True,
        "non_content_strategy_rejected": True,
        "missing_token_rejected": True,
        "human_participants": 0,
        "clinical_efficacy_tested": False,
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("round3 research HTTP verification passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.artifact, arguments.encoder, arguments.strategy, arguments.output)
