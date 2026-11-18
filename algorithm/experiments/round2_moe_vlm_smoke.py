"""公开真实原片的冻结 Chinese-CLIP 前向；不是情感性能实验。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.knowledge_moe.features import FrozenCLIPEncoder, save_json, sha256_file


def download_chinese_clip(destination: Path) -> dict[str, Any]:
    """按固定公开 revision 下载权重，核对 LFS SHA；不使用账号或访问令牌。"""
    import requests

    repo = "OFA-Sys/chinese-clip-vit-base-patch16"
    revision = "36e679e65c2a2fead755ae21162091293ad37834"
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    listing = session.get(f"https://huggingface.co/api/models/{repo}/tree/{revision}", timeout=30)
    listing.raise_for_status()
    items = {x["path"]: x for x in listing.json()}
    provenance = {"model": repo, "revision": revision, "authenticated": False, "files": {}}
    for name in ["config.json", "preprocessor_config.json", "vocab.txt", "pytorch_model.bin"]:
        expected = items[name].get("lfs", {}).get("oid")
        path = destination / name
        if not path.exists() or (expected and sha256_file(path) != expected):
            response = session.get(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=(20, 120), stream=True)
            response.raise_for_status()
            temp = path.with_suffix(path.suffix + ".part")
            digest = hashlib.sha256()
            received = 0
            with temp.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
            if expected and digest.hexdigest() != expected:
                raise ValueError("公开模型权重 SHA 与发布者 LFS 不匹配")
            temp.replace(path)
            print(json.dumps({"file": name, "bytes": received, "sha256": digest.hexdigest()}), flush=True)
        provenance["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path), "publisher_lfs_sha256": expected}
    save_json(destination / "download_provenance.json", provenance)
    return provenance


def smoke(model_dir: Path, sample_manifest: Path, output: Path) -> dict[str, Any]:
    """真实原片取中间帧，对原始中文文本做联合预训练编码前向。"""
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    torch.manual_seed(42)
    torch.set_num_threads(1)
    samples_data = json.loads(sample_manifest.read_text(encoding="utf-8"))
    samples = samples_data if isinstance(samples_data, list) else samples_data["samples"]
    images, texts, records = [], [], []
    for row in samples:
        annotation = row.get("annotation", row)
        path = Path(row["path"])
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"无法打开原片: {path.name}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError("无法读取视频中间帧")
        images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        texts.append(annotation["text"])
        records.append({"sample_id": row.get("sample_id", row.get("id")), "source_sha256": sha256_file(path),
                        "frame_index": count // 2, "text": annotation["text"], "split": annotation.get("split")})
    start = time.perf_counter()
    encoder = FrozenCLIPEncoder(model_dir, device="cpu")
    load_seconds = time.perf_counter() - start
    timings = []
    features = None
    for _ in range(4):
        start = time.perf_counter()
        features = encoder.encode(images, texts)
        timings.append(time.perf_counter() - start)
    if features is None or not all(np.isfinite(x).all() for x in features.values()):
        raise ValueError("冻结编码器未返回有限特征")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "vlm_features.npz", **features, sample_ids=np.array([str(r["sample_id"]) for r in records]))
    result = {"schema_version": "vlm_encoder_smoke.v1", "source_dataset": "CH-SIMS-v2_public_sample",
              "model": "OFA-Sys/chinese-clip-vit-base-patch16", "device": "cpu", "threads": 1,
              "encoder_fingerprint": encoder.fingerprint, "samples": records,
              "feature_shapes": {k: list(v.shape) for k, v in features.items()}, "finite": True,
              "requires_grad_parameter_count": sum(p.numel() for p in encoder.model.parameters() if p.requires_grad),
              "load_seconds": load_seconds, "batch_forward_seconds_including_warmup": timings,
              "warm_p50_batch_seconds": float(np.median(timings[1:])),
              "limitations": ["仅4段公开原片的一帧及原文前向，不是完整时序视频理解",
                              "未训练情感分类，未证明准确率或融合增益", "v2样例不混入v1消融实验",
                              "不含音频编码；语音韵律需独立音频分支"]}
    save_json(output / "vlm_smoke.json", result)
    print(json.dumps({"finite": True, "shapes": result["feature_shapes"], "warm_p50_batch_seconds": result["warm_p50_batch_seconds"]}), flush=True)
    return result


def main() -> None:
    """运行显式下载和前向验证，不触发线上模型替换。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download_chinese_clip(args.model_dir)
    smoke(args.model_dir, args.samples, args.output)


if __name__ == "__main__":
    main()
