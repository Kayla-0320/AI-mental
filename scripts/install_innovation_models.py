"""安装已训练的稀疏图文知识专家到业务目录，保留原始摘要和审核状态。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]


def install(source: Path, destination: Path) -> dict:
    """复制明确的模型工件，核对拷贝摘要；不修改原始权重或知识背书。"""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("architecture") != "sparse_moe" or manifest.get("knowledge_review_mode") != "user_attested_only":
        raise ValueError("必须提供有原始背书的稀疏知识专家工件")
    names = ["manifest.json", "weights.pt", "knowledge.json", "knowledge_vectors.npz"]
    digests = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    if digests["weights.pt"] != manifest["checkpoint_sha256"]:
        raise ValueError("源权重摘要无效")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(source / name, destination / name)
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digests[name]:
            raise ValueError("部署复制摘要不一致")
    report = {"schema": "innovation_deployment.v1", "files": digests, "source_model": "round3_sparse_knowledge",
              "knowledge_status": "user_attested_not_independent_expert_review", "purpose": "auxiliary_reference",
              "clinical_validation": False, "audio_enabled": False,
              "selection_note": "integration candidate; knowledge gain not demonstrated; no clinical threshold changes"}
    (destination / "deployment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path, default=REPO / "algorithm/artifacts/innovation_fusion")
    args = parser.parse_args()
    source = args.source
    if source is None:
        candidates = [path for parent in REPO.parents for path in parent.glob("_*/round3/model/ch_sims_v2_ablation/sparse_knowledge")]
        if not candidates:
            parser.error("未找到原始工件，请通过--source指定sparse_knowledge目录")
        source = candidates[0]
    print(json.dumps(install(source, args.destination), ensure_ascii=False, indent=2))
