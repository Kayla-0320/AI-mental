"""下载维护者发布的 StudentLife 转换版，只提取 EMA 自评表。"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zipfile

import requests

from experiments.round14_common import DATA, OUT, save, sha256

URLS = [
    "https://zenodo.org/api/records/3529253/files/dataset_rds.zip/content",
    "https://zenodo.org/records/3529253/files/dataset_rds.zip?download=1",
]
MD5 = "353ae79f157097a57bff1c004634ba1b"
SIZE = 230469489


def range_download(url: str, temporary: Path) -> None:
    """续传被中断的归档，显式检查 Range 响应，最终仍校验完整 MD5。"""
    start = temporary.stat().st_size if temporary.exists() else 0
    if start > SIZE: raise ValueError("临时文件大小超出正式归档")
    chunks = [(offset, min(offset + 4 * 2**20 - 1, SIZE - 1)) for offset in range(start, SIZE, 4 * 2**20)]

    def fetch(bounds: tuple[int, int]) -> bytes:
        """独立分段仅返回与请求区间严格相符的字节。"""
        first, last = bounds
        response = requests.get(url, headers={"Range": f"bytes={first}-{last}"}, timeout=(20, 90))
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {first}-{last}/{SIZE}":
            raise ValueError("服务器未遵守分段边界")
        if len(response.content) != last - first + 1: raise ValueError("分段字节数错误")
        return response.content

    with ThreadPoolExecutor(max_workers=8) as pool, temporary.open("ab") as stream:
        for bounds, content in zip(chunks, pool.map(fetch, chunks), strict=True):
            stream.write(content); stream.flush()
            print(json.dumps({"stage": "download", "bytes": bounds[1] + 1, "total": SIZE}), flush=True)


def download() -> Path:
    """校验正式发布摘要，下载失败保留日志，不以测试夹具替代研究数据。"""
    DATA.mkdir(parents=True, exist_ok=True)
    destination = DATA / "dataset_rds.zip"
    attempts: list[dict] = []
    if not destination.exists():
        for url in URLS:
            try:
                response = requests.get(url, stream=True, headers={"Range": "bytes=0-0"}, timeout=(20, 30))
                attempts.append({"url": url, "status": response.status_code})
                response.raise_for_status()
                response.close()
                temporary = destination.with_suffix(".part")
                range_download(url, temporary)
                digest = hashlib.md5(temporary.read_bytes()).hexdigest()
                if digest != MD5:
                    raise ValueError("StudentLife 发布 MD5 不一致")
                temporary.replace(destination)
                break
            except (requests.RequestException, ValueError) as error:
                attempts.append({"error": str(error)})
                save(OUT / "data" / "download_attempts.json", attempts)
    if not destination.exists():
        raise RuntimeError("StudentLife 下载失败，详见 download_attempts.json")
    if hashlib.md5(destination.read_bytes()).hexdigest() != MD5:
        raise ValueError("已下载文件与发布摘要不一致")
    with zipfile.ZipFile(destination) as archive:
        members = archive.namelist()
        selected = [name for name in members if "EMA" in name and name.lower().endswith((".rds", ".rdata"))]
        for name in selected:
            target = DATA / "ema_rds" / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    for filename, url in {
        "SL_tbl_trans.R": "https://raw.githubusercontent.com/frycast/studentlife/master/R/SL_tbl_trans.R",
        "studentlife_DESCRIPTION": "https://raw.githubusercontent.com/frycast/studentlife/master/DESCRIPTION",
        "GPL-3.0.txt": "https://raw.githubusercontent.com/spdx/license-list-data/main/text/GPL-3.0-only.txt",
    }.items():
        target = DATA / filename
        if not target.exists():
            response = requests.get(url, timeout=(20, 45)); response.raise_for_status()
            target.write_bytes(response.content)
    save(OUT / "data" / "studentlife_provenance.json", {
        "record": "https://zenodo.org/records/3529253", "original": "https://studentlife.cs.dartmouth.edu/dataset.html",
        "converter": "https://github.com/frycast/studentlife", "published_md5": MD5,
        "sha256": sha256(destination), "bytes": destination.stat().st_size,
        "license": "CC BY 4.0 (publisher metadata checked in earlier source audit)",
        "earlier_project_use": "catalogued only; local notebook is not the EMA dataset; no StudentLife training scripts found",
        "archive_members": members, "ema_members": selected, "attempts": attempts,
        "scope": "university-student self-report; no clinical labels, treatment outcomes, or adolescent transfer claim",
    })
    print(json.dumps({"downloaded": str(destination), "ema_members": selected}), flush=True)
    return destination


if __name__ == "__main__":
    download()
