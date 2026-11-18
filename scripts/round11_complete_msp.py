"""补齐旧mask跳过的真实MSP语音编码，构成观测条件相同的语音对照。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from experiments.round11_common import OUTPUT, PREVIOUS, digest, save
from round11_sensevoice import decode

MODEL = Path(r"D:\ai\models\encoders\wav2vec2-emotion-msp")
MEDIA = Path(r"D:\Develop\AIC\data\public\ch_sims_v2\supervised")


def main() -> None:
    """保持已编码行逐位不变，仅补充此前未执行的编码。"""
    output = OUTPUT / "msp_complete_features"
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(MODEL, local_files_only=True, attn_implementation="sdpa").cuda().eval()
    with safe_open(MODEL/"model.safetensors", framework="pt") as weights:
        head = {name: weights.get_tensor("classifier."+name).cuda() for name in
                ("dense.weight", "dense.bias", "out_proj.weight", "out_proj.bias")}
    reports = {}
    for split in ("train", "selection"):
        destination = output / f"{split}.npz"
        if destination.exists():
            raise ValueError("拒绝覆盖补齐后的对照表征")
        original = PREVIOUS / "speech_features" / f"{split}.npz"
        with np.load(original, allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        present = values["available"].copy()
        indices = np.flatnonzero(~present)
        with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
            for start in range(0, len(indices), 8):
                ix = indices[start:start+8]
                decoded = list(pool.map(decode, [MEDIA/f"{values['sample_ids'][i]}.mp4" for i in ix]))
                positions = [j for j, row in enumerate(decoded) if row[2]]
                if positions:
                    batch = extractor([decoded[j][0] for j in positions], sampling_rate=16000,
                        padding=True, return_attention_mask=True, return_tensors="pt").to("cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        encoded = model(**batch).last_hidden_state.float()
                    mask = model._get_feature_vector_attention_mask(encoded.shape[1], batch["attention_mask"]).unsqueeze(-1)
                    mean = (encoded*mask).sum(1)/mask.sum(1).clamp_min(1)
                    variance = ((encoded-mean[:,None]).square()*mask).sum(1)/mask.sum(1).clamp_min(1)
                    dim = torch.nn.functional.linear(torch.tanh(torch.nn.functional.linear(mean,head["dense.weight"],head["dense.bias"])),
                        head["out_proj.weight"],head["out_proj.bias"])
                    target = ix[positions]
                    values["speech"][target] = torch.cat([mean, variance.clamp_min(0).sqrt()],1).cpu().numpy()
                    values["speech_emotion"][target] = dim.cpu().numpy()
                    values["available"][target] = True
                for j, i in enumerate(ix):
                    values["original_duration"][i] = decoded[j][1]
        with np.load(original, allow_pickle=False) as data:
            for key in ("speech", "speech_emotion", "available", "original_duration"):
                if not np.array_equal(values[key][present], data[key][present]):
                    raise ValueError("既有有效表征被改变")
        np.savez_compressed(destination, **values)
        reports[split] = {"n": len(present), "newly_encoded": int((values["available"] & ~present).sum()),
                          "original_present_unchanged": True, "source_sha256": digest(original),
                          "destination_sha256": digest(destination)}
    save(output/"manifest.json", {"splits": reports, "mask_independent_encoding": True,
        "source_model_sha256": digest(MODEL/"model.safetensors"), "new_row_precision": "bf16 encoder/fp32 pooling same as original",
        "code_sha256": digest(Path(__file__)), "test_or_calibration_read": False})
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
