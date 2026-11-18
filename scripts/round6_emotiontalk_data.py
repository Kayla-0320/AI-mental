"""读取已授权的 EmotionTalk 固定版本；范围下载标注，不下载整包媒体。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
from typing import Any

from huggingface_hub import HfFileSystem, hf_hub_download

REPO = "BAAI/Emotiontalk"
REVISION = "adbc17fc944e8cf2873643906160c6ca0259ab61"


def digest(path: Path) -> str:
    """流式计算本地文件摘要。"""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: Any) -> None:
    """以 UTF-8 保存可审计的元数据。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_local_annotations(path: Path) -> dict[str, dict[str, Any]]:
    """仅读取普通 JSON 成员，拒绝重复 ID 与非预期路径。"""
    result: dict[str, dict[str, Any]] = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name)
            if ".." in name.parts or name.is_absolute() or name.suffix != ".json":
                raise ValueError(f"非预期标注成员: {member.name}")
            if name.stem in result:
                raise ValueError(f"重复标注 ID: {name.stem}")
            result[name.stem] = json.loads(archive.extractfile(member).read())
    return result


def acquire(source: Path, output: Path) -> dict[str, Any]:
    """使用本机已保存的认证读取固定版本，不记录令牌或签名 URL。"""
    source.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "acquisition.json"
    if completed.exists():
        existing = json.loads(completed.read_text(encoding="utf-8"))
        if existing["revision"] != REVISION:
            raise ValueError("已有输出的版本不符")
        for name, expected in existing["annotation_sha256"].items():
            if digest(output / name) != expected:
                raise ValueError("已有标注文件摘要不符")
        return existing
    files: dict[str, Any] = {}
    for name in ["README.md", "Text.tar", "Video.tar"]:
        path = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=source))
        files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    text = read_local_annotations(source / "Text.tar")
    video = read_local_annotations(source / "Video.tar")
    for name, values in [("text_annotations.jsonl", text), ("video_annotations.jsonl", video)]:
        with (output / name).open("w", encoding="utf-8") as handle:
            for key in sorted(values):
                handle.write(json.dumps({"id": key, "annotation": values[key]}, ensure_ascii=False) + "\n")
    fs = HfFileSystem()
    remote = f"datasets/{REPO}@{REVISION}/Multimodal.tar"
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    started = time.monotonic()
    partial = output / "multimodal_annotations.jsonl.partial"
    with fs.open(remote, "rb", block_size=1024 * 1024) as stream, partial.open("w", encoding="utf-8") as handle:
        # 该主包是未压缩 tar：seek 跳过媒体 ZIP，HTTP Range 只读头部和标注。
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = PurePosixPath(member.name)
                if ".." in name.parts or name.is_absolute():
                    raise ValueError("不安全的归档路径")
                inventory.append({"name": member.name, "size": member.size, "offset_data": member.offset_data})
                if name.suffix == ".json":
                    if name.stem in seen:
                        raise ValueError("多模态标注 ID 重复")
                    value = json.loads(archive.extractfile(member).read())
                    handle.write(json.dumps({"id": name.stem, "annotation": value}, ensure_ascii=False) + "\n")
                    seen.add(name.stem)
                    if len(seen) % 5000 == 0:
                        print(json.dumps({"multimodal_annotations": len(seen), "seconds": round(time.monotonic() - started, 1)}), flush=True)
                    if len(seen) == len(text):
                        # 标注集中存放，收齐即停，避免为索引每个视频而取回大量媒体块。
                        break
                else:
                    print(json.dumps({"skipped_media_member": member.name, "bytes": member.size}), flush=True)
    if set(text) != set(video) or set(text) != seen:
        raise ValueError("三个模态标注的样本集合不一致")
    partial.replace(output / "multimodal_annotations.jsonl")
    save(output / "archive_inventory.json", inventory)
    names = ["text_annotations.jsonl", "video_annotations.jsonl", "multimodal_annotations.jsonl", "archive_inventory.json"]
    report = {"repo": REPO, "revision": REVISION, "license": "CC-BY-NC-SA-4.0",
              "acquisition": "authenticated HTTP byte-range tar traversal; no full media archive download",
              "inventory_scope": "archive prefix through all 19250 multimodal annotations; media inventory is incomplete",
              "full_media_archive_downloaded": False, "samples": len(seen), "small_archives": files,
              "annotation_sha256": {name: digest(output / name) for name in names},
              "text_emotions": dict(Counter(row["emotion_result"] for row in text.values())),
              "media_archives": [row for row in inventory if not row["name"].endswith(".json")],
              "seconds": time.monotonic() - started}
    save(completed, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.source, args.output)
