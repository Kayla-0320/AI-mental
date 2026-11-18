"""保存第十轮代码快照、实际环境和关键工件指纹，不覆盖旧实验数字。"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))
from experiments.round10_fusion import OUTPUT, ROOT, save


def sha256(path: Path) -> str:
    """避免把大模型整体读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """复制当前代码并记录与结果一一对应的锁定文件和预测指纹。"""
    repo = Path(__file__).resolve().parents[1]
    destination = OUTPUT / "source" / "current"
    files = [*repo.glob("algorithm/experiments/round10_*.py"), *repo.glob("scripts/round10_*.py"), *repo.glob("scripts/round10_*.ps1"),
             repo / "algorithm/experiments/round9_moe_baselines.py", repo / "algorithm/experiments/round9_counterfactual.py",
             repo / "scripts/round9_build_strong_bundle.py", repo / "algorithm/tests/test_round10_protocol.py"]
    source_rows = []
    for source in files:
        relative = source.relative_to(repo)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_rows.append({"path": relative.as_posix(), "sha256": sha256(target)})
    diff = subprocess.check_output(["git", "diff", "--", "algorithm/experiments/round9_moe_baselines.py",
        "algorithm/experiments/round9_counterfactual.py", "scripts/round9_build_strong_bundle.py"], cwd=repo)
    (destination / "tracked_fixes.patch").write_bytes(diff)
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "bitsandbytes", "accelerate",
                 "onnx", "onnxruntime", "catboost", "av", "safetensors", "huggingface-hub", "Pillow", "pytest", "matplotlib", "peft"):
        versions[name] = metadata.version(name)
    import torch
    import cv2
    reconstruction = OUTPUT / "locked_rebuild_check.json"
    training_runtime = json.loads(reconstruction.read_text("utf-8")).get("training_numeric_runtime") if reconstruction.exists() else None
    save(OUTPUT / "execution_environment.json", {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": versions, "cv2_module_version": cv2.__version__,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "verified_training_entry_numeric_runtime": training_runtime,
        "git_head_before_changes": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()})
    artifacts = []
    for pattern in ("*/selection_locked.json", "*/preregister.json", "*/results.json", "*/historical_predictions.npz",
                    "*/model_manifest.json", "*/model.pkl", "*/audit.json", "*_features/manifest.json", "figures/*.png",
                    "sequence/*model.pkl", "sequence/model_archive_check.json", "sequence/*predictions.npz", "sequence/*selection_locked.json"):
        for path in sorted(OUTPUT.glob(pattern)):
            artifacts.append({"path": path.relative_to(OUTPUT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    for path in (OUTPUT / "official_meta.csv", OUTPUT / "regression_junit.xml", OUTPUT / "final_audit.json",
                 OUTPUT / "locked_rebuild_check.json", OUTPUT / "counterfactual_soft_smoke.json", OUTPUT / "REVIEW-PACKAGE.md",
                 ROOT / "第十轮多模态情感模型优化报告-20261116.md"):
        if path.exists():
            artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    archived = [{"path": str(path), "sha256": sha256(path)} for path in sorted((OUTPUT / "source").glob("*.py"))]
    for run in ("reproduction_locked_v1", "reproduction_locked_v2", "reproduction_locked_v3"):
        for relative in ("rebuild_check.json", "rebuild_protocol.json", "refined_final/results.json"):
            path = OUTPUT / run / relative
            if path.exists():
                artifacts.append({"path": path.relative_to(OUTPUT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    save(OUTPUT / "artifact_manifest.json", {"source_snapshot": str(destination), "sources": source_rows, "archived_sources": archived,
        "artifacts": artifacts, "original_data_root": str(ROOT / "round5"), "repo": str(repo),
        "scope": "Research artifacts, repository and original media required for full reproduction; no service deployment"})
    print(json.dumps({"source_files": len(source_rows), "key_artifacts": len(artifacts), "manifest": str(OUTPUT / "artifact_manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
