"""固定版本 EmotionTalk 的媒体索引与可续传范围下载，不输出认证信息。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
from typing import Any
import zipfile

from huggingface_hub import HfApi, HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))
from experiments.round11_common import ROOT, OUTPUT, digest, save

REPO = "BAAI/Emotiontalk"
REVISION = "adbc17fc944e8cf2873643906160c6ca0259ab61"
REMOTE = f"datasets/{REPO}@{REVISION}/Multimodal.tar"
DESTINATION = ROOT / "data/public/emotiontalk_media"
RECORDS = OUTPUT / "emotiontalk_acquisition"


def inventory() -> list[dict[str, Any]]:
    """遍历未压缩tar头部，seek越过压缩媒体，不读取情感测试标签。"""
    target = RECORDS / "media_inventory.json"
    if target.exists():
        report = json.loads(target.read_text("utf-8"))
        if report["revision"] != REVISION:
            raise ValueError("索引版本不符")
        return report["media"]
    fs = HfFileSystem()
    media, annotation_count = [], 0
    with fs.open(REMOTE, "rb", block_size=1024*1024) as stream:
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = PurePosixPath(member.name)
                if ".." in name.parts or name.is_absolute():
                    raise ValueError("非法归档路径")
                if name.suffix == ".zip":
                    row = {"name": member.name, "size": member.size, "offset_data": member.offset_data}
                    media.append(row)
                    print(json.dumps({"zip": name.name, "bytes": member.size}), flush=True)
                elif name.suffix == ".json":
                    annotation_count += 1
                else:
                    raise ValueError(f"未识别归档成员类型: {name.suffix}")
    report = {"repo": REPO, "revision": REVISION, "media": media,
              "annotation_members": annotation_count, "total_zip_bytes": sum(row["size"] for row in media),
              "scope": "full tar headers; label values not read", "code_sha256": digest(Path(__file__))}
    if annotation_count != 19250:
        raise ValueError("官方完整标注计数不符")
    save(target, report)
    return media


def expected_ids() -> set[str]:
    """只读标注ID构建媒体白名单，不访问任何情感标注值。"""
    source = ROOT / "_国赛优化/round6/data/text_annotations.jsonl"
    return {json.loads(line)["id"] for line in source.read_text("utf-8").splitlines()}


def download_one(row: dict[str, Any]) -> dict[str, Any]:
    """下载一个tar内ZIP的连续范围，核对长度与逐媒体CRC后安全展开。"""
    name = PurePosixPath(row["name"]).name
    receipt_path = RECORDS / "receipts" / f"{name}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text("utf-8"))
        for sample in receipt["samples"]:
            if (DESTINATION / "clips" / f"{sample['id']}.mp4").stat().st_size != sample["bytes"]:
                raise ValueError("已完成媒体缺失或长度不符")
        return receipt
    archive = DESTINATION / "archives" / name
    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(".zip.partial")
    if not archive.exists():
        offset = part.stat().st_size if part.exists() else 0
        if offset > row["size"]:
            raise ValueError("断点超过预期ZIP长度")
        fs = HfFileSystem()
        with fs.open(REMOTE, "rb", block_size=16*1024*1024, cache_type="readahead") as stream, part.open("ab") as out:
            stream.seek(row["offset_data"]+offset)
            last_progress = time.monotonic()
            while offset < row["size"]:
                block = stream.read(min(4*1024*1024, row["size"]-offset))
                if not block:
                    raise ValueError("远端媒体提前结束")
                out.write(block)
                offset += len(block)
                if time.monotonic()-last_progress > 20:
                    print(json.dumps({"downloading": name, "bytes": offset, "total": row["size"]}), flush=True)
                    last_progress = time.monotonic()
        part.replace(archive)
    if archive.stat().st_size != row["size"]:
        raise ValueError("ZIP长度不符")
    allowed = expected_ids()
    clips = DESTINATION / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    samples = []
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".mp4" or path.stem not in allowed:
                raise ValueError("ZIP含不在媒体白名单的文件")
            target = clips / (path.stem+".mp4")
            if target.exists():
                raise ValueError("重复媒体ID或遗留未完成提取，需要先核验")
            temporary = target.with_suffix(".mp4.partial")
            # ZipExtFile读到EOF会核验该成员的CRC。
            with zipped.open(member) as source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out, length=1024*1024)
            if temporary.stat().st_size != member.file_size:
                raise ValueError("解压成员长度不符")
            temporary.replace(target)
            samples.append({"id": path.stem, "bytes": member.file_size, "crc32": f"{member.CRC:08x}"})
    receipt = {"revision": REVISION, "archive_member": row, "zip_sha256": digest(archive),
               "samples": samples, "crc_verified": True}
    save(receipt_path, receipt)
    print(json.dumps({"downloaded_and_verified": name, "clips": len(samples)}), flush=True)
    return receipt


def main() -> None:
    """完整下载与验证官方混合tar，避免把其中直接存放的MP4漏掉。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    RECORDS.mkdir(parents=True, exist_ok=True)
    if not args.download:
        print(json.dumps({"action": "use --download for fixed revision mixed tar acquisition",
                          "archive_bytes": 21294498304}), flush=True)
        return
    acquire_full(args.workers)


def acquire_full(workers: int = 4) -> None:
    """完整归档哈希加逐片哈希；复用已完成媒体时再次核验，支持提取中断。"""
    archives = DESTINATION / "archives"
    clips = DESTINATION / "clips"
    archives.mkdir(parents=True, exist_ok=True)
    clips.mkdir(parents=True, exist_ok=True)
    archive = archives / "Multimodal.tar"
    metadata = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    entry = next(row for row in metadata.siblings if row.rfilename == "Multimodal.tar")
    expected_hash = entry.lfs.sha256
    expected_bytes = entry.size
    source_lock = RECORDS / "source_lock.json"
    locked = {"repo": REPO, "revision": REVISION, "bytes": expected_bytes,
              "sha256": expected_hash, "format": "uncompressed tar mixing ZIP and direct MP4", "label_values_read": False}
    if source_lock.exists():
        if json.loads(source_lock.read_text("utf-8")) != locked:
            raise ValueError("来源锁不一致")
    else:
        save(source_lock, locked)
    part = archive.with_suffix(".tar.partial")
    if not archive.exists():
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_bytes:
            raise ValueError("下载断点超出官方大小")
        if shutil.disk_usage(ROOT).free < expected_bytes-offset + expected_bytes + 20*1024**3:
            raise ValueError("下载和展开将侵占20GiB保留空间")
        if shutil.disk_usage(ROOT).free < 3*(expected_bytes-offset)+20*1024**3:
            raise ValueError("并行范围暂存空间不足")
        from round11_ranges import download
        download(REMOTE,part,expected_bytes,expected_hash,workers)
        part.replace(archive)
    if archive.stat().st_size != expected_bytes or digest(archive) != expected_hash:
        raise ValueError("官方媒体归档大小或LFS摘要不符")
    save(RECORDS / "archive_verified.json", {**locked, "local_path": str(archive)}, immutable=False)
    allowed = expected_ids()
    previous_path = RECORDS / "media_records.jsonl"
    previous = {row["id"]: row for row in map(json.loads, previous_path.read_text("utf-8").splitlines())} if previous_path.exists() else {}
    samples, members, seen = [], [], set()
    seen_rows: dict[str, Any] = {}
    duplicates: list[dict[str, Any]] = []

    def extract_one(name: str, size: int, source: Any, handle: Any) -> None:
        """流式复制单片，ID白名单、长度与SHA都核对。"""
        path = PurePosixPath(name)
        identifier = path.stem
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".mp4" or identifier not in allowed:
            raise ValueError("媒体成员不在白名单或ID重复")
        if identifier in seen:
            import hashlib
            hashed = hashlib.sha256()
            length = 0
            for block in iter(lambda: source.read(1024*1024), b""):
                hashed.update(block)
                length += len(block)
            original = seen_rows[identifier]
            if size != length or size != original["bytes"] or hashed.hexdigest() != original["sha256"]:
                raise ValueError("重复ID对应不同媒体，禁止静默选取")
            duplicates.append({"id": identifier, "member": name, "sha256": original["sha256"],
                               "identical_bytes_verified": True})
            return
        seen.add(identifier)
        target = clips / f"{identifier}.mp4"
        if identifier in previous:
            row = previous[identifier]
            if target.stat().st_size != size or digest(target) != row["sha256"]:
                raise ValueError("已提取媒体校验失败")
        else:
            import hashlib
            temporary = target.with_suffix(".mp4.partial")
            hashed = hashlib.sha256()
            with temporary.open("wb") as out:
                for block in iter(lambda: source.read(1024*1024), b""):
                    out.write(block)
                    hashed.update(block)
            if temporary.stat().st_size != size:
                raise ValueError("提取媒体长度不符")
            temporary.replace(target)
            row = {"id": identifier, "bytes": size, "sha256": hashed.hexdigest()}
            handle.write(json.dumps(row)+"\n")
            handle.flush()
        samples.append(row)
        seen_rows[identifier] = row
        if len(samples) % 500 == 0:
            print(json.dumps({"media_verified": len(samples), "total": len(allowed)}), flush=True)

    with previous_path.open("a", encoding="utf-8") as handle, tarfile.open(archive, "r:") as tar:
        for member in tar:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("非法tar路径")
            members.append({"name": member.name, "size": member.size, "offset_data": member.offset_data})
            if path.suffix.lower() == ".mp4":
                with tar.extractfile(member) as source:
                    extract_one(member.name, member.size, source, handle)
            elif path.suffix.lower() == ".zip":
                zip_path = archives / path.name
                if not zip_path.exists():
                    with tar.extractfile(member) as source, zip_path.open("wb") as out:
                        shutil.copyfileobj(source, out, 1024*1024)
                with zipfile.ZipFile(zip_path) as zipped:
                    for inner in zipped.infolist():
                        if inner.is_dir():
                            continue
                        with zipped.open(inner) as source:
                            extract_one(inner.filename, inner.file_size, source, handle)
            elif path.suffix.lower() != ".json":
                raise ValueError("未知归档成员")
    if seen != allowed:
        raise ValueError("媒体未完整覆盖标注ID")
    save(RECORDS / "media_inventory.json", {**locked, "members": members})
    save(RECORDS / "duplicate_members_verified.json", {"duplicate_count": len(duplicates),
         "duplicates": duplicates, "rule": "same ID accepted once only after byte-level SHA256 equality"})
    save(RECORDS / "complete.json", {**locked, "clips": len(samples),
        "media_records_sha256": digest(previous_path), "all_members_hashed": True,
        "coverage": 1., "archive_lfs_hash_verified": True, "identical_duplicate_members": len(duplicates)})
    print(json.dumps({"acquisition_complete": True, "clips": len(samples)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # 上游网络异常可能带临时签名URL；终端只输出异常类型。
        print(json.dumps({"acquisition_failed": type(error).__name__, "details_redacted": True}), flush=True)
        raise SystemExit(1)
