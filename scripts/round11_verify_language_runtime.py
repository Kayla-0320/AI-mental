"""以原冻结批次重放核验省略未用输出头后的数值，不读取任何目标。"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
import torch
from experiments.round11_common import ROOT,OUTPUT,PREVIOUS,digest,save
from round11_language_runtime import build,encode


def main() -> None:
    """同原batch8/24、原提示与PAD；只比开发输入已有冻结表示。"""
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("7b","3b"));args=parser.parse_args()
    torch.manual_seed(42);torch.set_num_threads(1)
    batch=8 if args.mode=="7b" else 24
    with np.load(ROOT/"_国赛优化/round5/text_features/train.npz",allow_pickle=False) as z:ids=z["sample_ids"][:batch];texts=z["raw_text"][:batch].astype(str).tolist()
    model,tokenizer,head,metadata=build(args.mode);actual=encode(model,tokenizer,head,texts)
    folder="llm_large_features" if args.mode=="7b" else "llm_features"
    with np.load(PREVIOUS/folder/"train.npz",allow_pickle=False) as z:
        if not np.array_equal(ids,z["sample_ids"][:batch]):raise ValueError("冻结语言批次ID不一致")
        errors={name:float(np.abs(value-z[name][:batch]).max()) for name,value in actual.items()}
    if max(errors.values())>1e-5:raise ValueError(f"解码器分支重放与原冻结表示不一致:{errors}")
    save(OUTPUT/f"validation/language_runtime_{args.mode}_equivalence.json",{"n":batch,"metadata":metadata,"max_abs_errors":errors,
        "source_code_sha256":digest(Path(__file__).with_name("round11_language_runtime.py")),"target_labels_used":False},immutable=False)
    print(errors,flush=True)


if __name__=="__main__":main()
