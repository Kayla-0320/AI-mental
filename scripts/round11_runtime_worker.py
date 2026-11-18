"""当前请求的单个隔离阶段，进程结束即释放模型上下文。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
from experiments.round11_common import OUTPUT,save


def language_features(text: str,mode: str) -> dict[str,np.ndarray]:
    """兼容的单句特征接口，使用已核验的等价解码器分支。"""
    from round11_language_runtime import encode_one
    return encode_one(text,mode)[0]


def main() -> None:
    """读取本次请求全新中间值，禁止读取历史离线编码充作实时输入。"""
    parser=argparse.ArgumentParser();parser.add_argument("kind",choices=("base","llm","fusion"));parser.add_argument("--folder",type=Path,required=True)
    parser.add_argument("--mode",choices=("7b","3b"),required=True);args=parser.parse_args();folder=args.folder
    job=json.loads((folder/"request.json").read_text("utf-8"));metrics={}
    if args.kind=="fusion":
        import pickle
        from experiments.round11_final import predict
        with np.load(folder/"complete.npz",allow_pickle=False) as z:data={k:z[k] for k in z.files}
        with (OUTPUT/f"final_models/{args.mode}/model.pkl").open("rb") as handle:bundle=pickle.load(handle)
        np.savez_compressed(folder/"prediction.npz",probabilities=predict(bundle,data),sample_ids=data["sample_ids"])
    else:
        import torch
        torch.manual_seed(42);torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
        if args.kind=="base":
            from round11_runtime import RawRuntime
            runtime=RawRuntime();_,timings,data=runtime.infer(Path(job["clip"]),job["text"])
            np.savez_compressed(folder/"base.npz",**data);metrics["inference_components"]=timings
        else:
            from round11_language_runtime import encode_one
            features,language_metadata=encode_one(job["text"],args.mode)
            metrics["language_metadata"]=language_metadata
            with np.load(folder/"base.npz",allow_pickle=False) as z:data={k:z[k] for k in z.files}
            data.update(features);np.savez_compressed(folder/"complete.npz",**data)
        torch.cuda.synchronize();metrics.update(cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,cuda_peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2)
    save(folder/f"{args.kind}_metrics.json",metrics,immutable=False)


if __name__=="__main__":main()
