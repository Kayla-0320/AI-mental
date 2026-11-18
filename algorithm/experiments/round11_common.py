"""第十一轮独立输出路径与可审计工件，保持第十轮文件只读。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(r"D:\Develop\AIC")
PREVIOUS = ROOT / "_国赛优化" / "round10"
OUTPUT = ROOT / "_国赛优化" / "round11"


def digest(path: Path) -> str:
    """分块读取原始工件，不修改来源文件。"""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: Any, immutable: bool = True) -> None:
    """锁定文件拒绝覆盖，滚动进度须显式指定可更新。"""
    if path.exists() and immutable:
        raise ValueError(f"拒绝覆盖既有第十一轮工件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    temporary.replace(path)
