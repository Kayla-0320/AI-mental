"""公开监督情感编码器的冻结特征；二类输出仅作特征，不伪造中性标签。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from experiments.round10_fusion import ROOT, OUTPUT, load_npz, save
from perception.knowledge_moe.features import sha256_file

REPO = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
REVISION = "7c257c5cde3225d0789acfa8d67eb043289b0295"
MODEL = Path(r"D:\ai\models\encoders\erlangshen-roberta-110m-sentiment")


def main() -> None:
    """固定公开版本与许可证，显式记录上游训练数据重合的不确定性。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "selection"])
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    repo = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment" if args.large else REPO
    revision = "7005a53278f557cb27010fb08bb56a88b92173a9" if args.large else REVISION
    model_path = MODEL.with_name("erlangshen-roberta-330m-sentiment") if args.large else MODEL
    weight_file = "model.safetensors" if args.large else "pytorch_model.bin"
    snapshot_download(repo, revision=revision, local_dir=model_path,
                      allow_patterns=["config.json", weight_file, "vocab.txt", "README.md"])
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).cuda().eval()
    output = OUTPUT / ("external_large_features" if args.large else "external_features")
    output.mkdir(parents=True, exist_ok=True)
    identity = sha256_file(model_path / weight_file)
    for split in args.splits:
        if (output / f"{split}.npz").exists():
            raise ValueError("拒绝覆盖公开编码工件")
        data = load_npz(ROOT / "round5" / "text_features" / f"{split}.npz")
        texts = data["raw_text"].astype(str).tolist()
        means, cls, scores = [], [], []
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                row = tokenizer(texts[start:start+64], truncation=True, max_length=128, padding=True, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    encoded = model.bert(**row)
                    binary_logits = model.classifier(encoded.pooler_output)
                hidden = encoded.last_hidden_state.float()
                mask = row["attention_mask"].unsqueeze(-1)
                cls.append(hidden[:, 0].cpu().numpy())
                means.append(((hidden*mask).sum(1)/mask.sum(1)).cpu().numpy())
                scores.append(binary_logits.float().cpu().numpy())
        np.savez_compressed(output / f"{split}.npz", sent_mean=np.concatenate(means), sent_cls=np.concatenate(cls),
                            sent_scores=np.concatenate(scores), sample_ids=data["sample_ids"])
        print(json.dumps({"encoded": split, "n": len(texts)}), flush=True)
    save(output / "manifest.json", {"repo": repo, "revision": revision, "weights_sha256": identity,
         "license": "Apache-2.0", "upstream_supervision": "model card: eight sentiment datasets, 227347 samples, binary sentiment",
         "own_target_label_training": False, "upstream_target_overlap_verified": False,
         "limitation": "Upstream training collection not fully enumerated; no claim of absence of CH-SIMS/pretraining contamination",
         "source": f"https://huggingface.co/{repo}"})


if __name__ == "__main__":
    main()
