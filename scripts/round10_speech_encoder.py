"""冻结语音情感编码器替代单纯MFCC证据，保留实际缺失和输入截断记录。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))

import av
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from experiments.round10_fusion import ROOT, OUTPUT, load_npz, save
from perception.knowledge_moe.features import sha256_file

REPO = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
REVISION = "6eba34a2485ea31cb03600241787c3a5edab8626"
MODEL = Path(r"D:\ai\models\encoders\wav2vec2-emotion-msp")
MEDIA = ROOT.parent / "data" / "public" / "ch_sims_v2" / "supervised"


def decode(identifier: str) -> tuple[np.ndarray, float, bool]:
    """原片解码至16kHz单声道，长片仅取居中的十秒并如实记录。"""
    values = []
    try:
        with av.open(str(MEDIA / f"{identifier}.mp4")) as container:
            stream = next(iter(container.streams.audio))
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    values.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):
                values.append(out.to_ndarray().reshape(-1))
    except (OSError, ValueError, StopIteration, av.error.FFmpegError):
        return np.zeros(0, np.float32), 0., False
    waveform = np.concatenate(values).astype(np.float32) if values else np.zeros(0, np.float32)
    duration = len(waveform)/16000
    valid = len(waveform) >= 1600 and np.isfinite(waveform).all()
    truncated = len(waveform) > 160000
    if truncated:
        start = (len(waveform)-160000)//2
        waveform = waveform[start:start+160000]
    return np.clip(waveform, -1, 1), duration, valid


def main() -> None:
    """编码器仅用于离线研究，英语监督来源的中文迁移由目标来源OOF评估。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "selection"])
    args = parser.parse_args()
    snapshot_download(REPO, revision=REVISION, local_dir=MODEL,
        allow_patterns=["model.safetensors", "config.json", "preprocessor_config.json", "README.md", "LICENSE"])
    torch.set_num_threads(1)
    torch.manual_seed(42)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(MODEL, local_files_only=True, attn_implementation="sdpa").cuda().eval()
    with safe_open(MODEL / "model.safetensors", framework="pt") as weights:
        head = {name: weights.get_tensor("classifier."+name).cuda() for name in
                ("dense.weight", "dense.bias", "out_proj.weight", "out_proj.bias")}
    digest = sha256_file(MODEL / "model.safetensors")
    output = OUTPUT / "speech_features"
    output.mkdir(parents=True, exist_ok=True)
    previous = output / "manifest.json"
    summaries = json.loads(previous.read_text("utf-8")).get("selection_summary", {}) if previous.exists() else {}
    for split in args.splits:
        if (output / f"{split}.npz").exists():
            raise ValueError("拒绝覆盖语音表征")
        data = load_npz(ROOT / "round5" / "text_features" / f"{split}.npz")
        n = len(data["labels"])
        original_present = data["modality_mask"][:, 1] & (data["quality"][:, 1] > 0)
        hidden = np.zeros((n, model.config.hidden_size*2), np.float32)
        emotions = np.zeros((n, 3), np.float32)
        available = np.zeros(n, bool)
        durations = np.zeros(n, np.float32)
        indices = np.flatnonzero(original_present)
        with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
            for block in range(0, len(indices), 128):
                ix = indices[block:block+128]
                decoded = list(pool.map(decode, data["sample_ids"][ix].astype(str).tolist()))
                batch_order = sorted([j for j, row in enumerate(decoded) if row[2]], key=lambda j: len(decoded[j][0]))
                for start in range(0, len(batch_order), 8):
                    positions = batch_order[start:start+8]
                    source = [decoded[j][0] for j in positions]
                    row = extractor(source, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt").to("cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        encoded = model(**row).last_hidden_state.float()
                    mask = model._get_feature_vector_attention_mask(encoded.shape[1], row["attention_mask"]).unsqueeze(-1)
                    mean = (encoded*mask).sum(1)/mask.sum(1).clamp_min(1)
                    variance = ((encoded-mean[:, None]).square()*mask).sum(1)/mask.sum(1).clamp_min(1)
                    dim = torch.nn.functional.linear(torch.tanh(torch.nn.functional.linear(mean, head["dense.weight"], head["dense.bias"])),
                                                    head["out_proj.weight"], head["out_proj.bias"])
                    target = ix[positions]
                    hidden[target] = torch.cat([mean, variance.clamp_min(0).sqrt()], 1).cpu().numpy()
                    emotions[target] = dim.cpu().numpy()
                    available[target] = True
                for j, index in enumerate(ix):
                    durations[index] = decoded[j][1]
                print(json.dumps({"stage": "speech_features", "split": split, "processed": min(len(indices), block+128), "total": len(indices)}), flush=True)
        np.savez_compressed(output / f"{split}.npz", speech=hidden, speech_emotion=emotions,
                            available=available, original_duration=durations, sample_ids=data["sample_ids"])
        summaries[split] = {"n": n, "available": int(available.sum()), "decode_failed_or_short": int((original_present & ~available).sum()),
                            "truncated_over10s": int((durations > 10).sum())}
    save(output / "manifest.json", {"repo": REPO, "revision": REVISION, "sha256": digest, "license": "CC-BY-NC-SA-4.0, research only",
         "upstream": "English MSP-Podcast dimensional emotion; no own target-label fine-tuning", "target_language": "Chinese; transfer not assumed",
         "features": "masked mean and std of1024D speech states plus arousal/dominance/valence",
         "max_contiguous_seconds": 10, "selection_summary": summaries})


if __name__ == "__main__":
    main()
