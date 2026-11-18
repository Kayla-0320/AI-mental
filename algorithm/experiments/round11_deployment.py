"""固定获胜融合配方下对照7B、3B与无生成模型的部署成本。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"

import numpy as np
from experiments import round11_evidence as evidence
from experiments.round11_features import load_data as original_load
from experiments.round11_common import OUTPUT,PREVIOUS,digest,save


def load_data(quality: str,mode: str) -> dict[str,np.ndarray]:
    """同一开发来源只替换文字表征，不使用703或821。"""
    data=original_load(quality)
    if mode=="3b":
        pieces=[]
        for split in ("train","selection"):
            with np.load(PREVIOUS/"llm_features"/f"{split}.npz",allow_pickle=False) as source:
                pieces.append({key:source[key] for key in source.files})
        if not np.array_equal(data["sample_ids"],np.concatenate([part["sample_ids"] for part in pieces])):
            raise ValueError("3B语义表征ID未对齐")
        for field in ("llm_last","llm_mean","llm_scores"):
            data[field]=np.concatenate([part[field] for part in pieces])
    elif mode!="light":
        raise ValueError("未知部署对照")
    return data


def main() -> None:
    """只执行事先固定的family10，作为精度/资源Pareto候选。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("3b","light"))
    args=parser.parse_args()
    folder=OUTPUT/f"deployment_{args.mode}"
    plan={"mode":args.mode,"fixed_family_index":10,"quality":"pts","encoder":"both","classes":3,
        "delegated_meta_candidates":"same depth4/5 and reliability false/true;300 fixed iterations",
        "text_fields":["llm_last","aux_mean","bge_cls"] if args.mode=="3b" else ["aux_mean","bge_cls"],
        "purpose":"quality/actual raw-media inference resource Pareto;no cached-head speedup claim",
        "selection_data":"same development2876 whole-source4outer/3inner;no703 or821",
        "source_sha256":{"runner":digest(Path(__file__)),"delegated_evidence":digest(Path(evidence.__file__)),
                         "features":digest(Path(__file__).with_name("round11_features.py"))},
        "third_party_feature_manifest_sha256":digest(PREVIOUS/"llm_features/manifest.json") if args.mode=="3b" else None}
    path=folder/"wrapper_preregister.json"
    if path.exists():
        if json.loads(path.read_text("utf-8"))!=plan:
            raise ValueError("部署对照计划不一致")
    else:
        save(path,plan)
    evidence.OUTPUT=folder
    evidence.load_data=lambda quality:load_data(quality,args.mode)
    if args.mode=="light":
        evidence.TEXT=("aux_mean","bge_cls")
    sys.argv=["round11_evidence.py","--family","10"]
    evidence.main()


if __name__=="__main__":
    main()
