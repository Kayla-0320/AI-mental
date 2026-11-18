"""第十四轮离线研究的路径、证据写入和文件指纹。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(r"D:\Develop\AIC\_国赛优化")
OUT = ROOT / "round14"
DATA = Path(r"D:\Develop\AIC\data\public\round14_sources")
SEED = 42


def save(path: Path, value: Any) -> None:
    """禁止非有限数字，保留可复算的 UTF-8 证据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    """流式计算文件摘要，避免把大型权重读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(path: Path, value: Any) -> None:
    """预注册相同内容可重入，拒绝事后覆盖实验定义。"""
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"冻结协议冲突: {path}")
    save(path, value)
