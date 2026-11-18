"""下载作者公开 CH-SIMS 特征并核对作者发布的摘要，不接受受限数据协议。"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from pathlib import Path

import requests

FILE_ID = "1l_Nb9h3BRa3S-N76YcSQ3ixakr0gn6Qr"
EXPECTED_SHA256 = "c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f"
AUTHOR_README = "https://raw.githubusercontent.com/thuiar/MMSA/master/README.md"


def sha256_file(path: Path) -> str:
    """分块计算文件摘要，避免将大文件整体读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output_dir: Path) -> Path:
    """仅下载公开作者文件；完整性校验成功后才发布最终路径。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "unaligned_39.pkl"
    session = requests.Session()
    readme = session.get(AUTHOR_README, timeout=30)
    readme.raise_for_status()
    if EXPECTED_SHA256 not in readme.text:
        raise RuntimeError("作者摘要已变化，停止下载，需核查来源")
    (output_dir / "MMSA_author_README.md").write_text(readme.text, encoding="utf-8")
    if destination.exists() and sha256_file(destination) == EXPECTED_SHA256:
        return destination
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    response = session.get(url, stream=True, timeout=(20, 60))
    response.raise_for_status()
    if "text/html" in response.headers.get("Content-Type", ""):
        page = response.text
        response.close()
        if "Google Drive can't scan this file for viruses" not in page:
            raise RuntimeError("文件返回的不是公开大文件下载确认页，拒绝继续")
        fields = dict(re.findall(r'name="([^\"]+)" value="([^\"]*)"', page))
        if fields.get("id") != FILE_ID or fields.get("confirm") != "t":
            raise RuntimeError("公开下载确认字段不匹配")
        response = session.get(
            "https://drive.usercontent.google.com/download",
            params={key: html.unescape(value) for key, value in fields.items()},
            stream=True,
            timeout=(20, 90),
        )
        response.raise_for_status()
    if "text/html" in response.headers.get("Content-Type", ""):
        raise RuntimeError("未取得数据二进制文件，拒绝把网页保存为特征")
    partial = destination.with_suffix(".pkl.partial")
    start = time.monotonic()
    size = 0
    digest = hashlib.sha256()
    with response, partial.open("wb") as target:
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            if size // (100 * 1024 * 1024) != (size - len(chunk)) // (100 * 1024 * 1024):
                print(f"downloaded_mb={size / 1024 / 1024:.0f}", flush=True)
    actual = digest.hexdigest()
    evidence = {
        "dataset": "CH-SIMS v1", "feature_file": "SIMS/Processed/unaligned_39.pkl",
        "author_readme": AUTHOR_README, "drive_file_id": FILE_ID,
        "downloaded_bytes": size, "expected_sha256": EXPECTED_SHA256,
        "actual_sha256": actual, "sha256_matches_author": actual == EXPECTED_SHA256,
        "seconds": time.monotonic() - start, "authenticated": False,
        "authorization_agreement_accepted": False,
        "scope": "academic research; preserve upstream conditions; no clinical claims",
    }
    (output_dir / "download_receipt.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    if actual != EXPECTED_SHA256:
        raise RuntimeError("作者 SHA256 不匹配，保留 partial 但不发布可用特征")
    partial.replace(destination)
    print(json.dumps(evidence, ensure_ascii=False), flush=True)
    return destination


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    download(args.output_dir)


if __name__ == "__main__":
    main()
