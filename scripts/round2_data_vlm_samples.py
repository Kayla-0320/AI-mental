"""从公开 CH-SIMS v2 归档流读取少量真实原片，供编码器前向验证。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import requests

BASE = "https://huggingface.co/datasets/THUIAR/MMLA-Datasets/resolve/main/CH-SIMSv2.0/"


def collect(output: Path, maximum: int = 4) -> None:
    """保存有限原片和标注出处；不把少量前向验证样例混入全量训练。"""
    if not 1 <= maximum <= 10:
        raise ValueError("验证样例数量必须在1至10之间")
    output.mkdir(parents=True, exist_ok=True)
    annotation: dict[str, dict[str, str]] = {}
    for split in ("train", "dev", "test"):
        response = requests.get(BASE + split + ".tsv?download=true", timeout=30)
        response.raise_for_status()
        (output / f"{split}.tsv").write_text(response.text, encoding="utf-8")
        for row in csv.DictReader(io.StringIO(response.text), delimiter="\t"):
            row["split"] = split
            annotation[row["id"]] = row
    url = BASE + "Ch-simsv2_video.tar.gz?download=true"
    found: list[dict[str, object]] = []
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        if response.status_code != 200 or "text/html" in response.headers.get("content-type", ""):
            raise RuntimeError("未取得完整归档流响应")
        response.raw.decode_content = True
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".mp4"):
                    continue
                if member.size > 50 * 1024 * 1024:
                    continue
                # 按归档中的父目录及文件名匹配公开标注，不使用任意归档路径写文件。
                parts = member.name.split("/")
                clip = Path(parts[-1]).stem
                parent = parts[-2] if len(parts) > 1 else ""
                candidates = [f"{parent}_{clip}", f"{parent}$_${clip}", clip, clip.removeprefix("Ch-sims_"), f"{parent}/{clip}"]
                key = next((key for key in candidates if key in annotation), None)
                if key is None:
                    if len(found) == 0:
                        print(f"unmatched_member={member.name};example_annotation={next(iter(annotation))}", flush=True)
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                raw = stream.read()
                filename = f"sample_{len(found):02d}.mp4"
                (output / filename).write_bytes(raw)
                found.append({"id": key, "path": str(output / filename), "archive_member": member.name,
                              "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                              "annotation": annotation[key]})
                print(f"saved={filename} id={key} bytes={len(raw)}", flush=True)
                if len(found) >= maximum:
                    break
    receipt = {"dataset": "CH-SIMS-v2", "purpose": "real raw clip frozen VLM forward smoke check only; separate from CH-SIMS-v1 training",
               "source": url, "full_archive_downloaded": False, "samples": found,
               "license_note": "retain THUIAR and upstream research conditions; not a new clinical dataset"}
    (output / "samples.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(found) < maximum:
        raise RuntimeError(f"只取到{len(found)}个匹配原片")


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum", type=int, default=4)
    args = parser.parse_args()
    collect(args.output_dir, args.maximum)


if __name__ == "__main__":
    main()
