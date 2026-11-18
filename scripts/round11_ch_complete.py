"""补齐最终拟合与一次历史诊断所需纯输入，不把目标传给编码器。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
from experiments.round11_common import ROOT,OUTPUT,PREVIOUS,digest,save


def pts(split: str,workers: int) -> None:
    """保持既定阈值，只替换可解码的时轴测量。"""
    from round11_input_audit import audit_one
    folder=OUTPUT/"input_audit";target=folder/f"{split}_masks.npz"
    if target.exists():return
    with np.load(ROOT/f"_国赛优化/round5/text_features/{split}.npz",allow_pickle=False) as z:
        items=[{"id":str(i),"split":split,"old_mask":m.tolist(),"old_quality":q.tolist()}
               for i,m,q in zip(z["sample_ids"],z["modality_mask"],z["quality"])]
    path=folder/f"{split}_records.jsonl"
    records=[json.loads(line) for line in path.read_text("utf-8").splitlines()] if path.exists() else []
    done={r["id"] for r in records}
    with ThreadPoolExecutor(max_workers=workers) as pool,path.open("a",encoding="utf-8") as handle:
        missing=[r for r in items if r["id"] not in done]
        for start in range(0,len(missing),32):
            for row in pool.map(audit_one,missing[start:start+32]):
                handle.write(json.dumps(row,ensure_ascii=False)+"\n");handle.flush();records.append(row)
            print(json.dumps({"pts_split":split,"audited":len(records),"n":len(items)}),flush=True)
    lookup={r["id"]:r for r in records};ordered=[lookup[r["id"]] for r in items]
    np.savez_compressed(target,sample_ids=np.array([r["id"] for r in ordered]),
        modality_mask=np.array([r["new_mask"] for r in ordered],bool),quality=np.array([r["new_quality"] for r in ordered],np.float32))
    save(folder/f"{split}_complete.json",{"n":len(items),"restored_audio":sum(not r["old_mask"][1] and r["new_mask"][1] for r in ordered),
        "verified":sum(r["measurement_verified"] for r in ordered),"source_sha256":digest(path),"labels_read":False})


def speech(kind: str,splits: list[str]) -> None:
    """同冻结模型和精度，旧MSP已存在的有效行逐位保留。"""
    import torch
    from round11_sensevoice import build_model,acquire_split,decode
    from round11_external_gpu import msp_model,msp_encode
    torch.manual_seed(42);torch.set_num_threads(1)
    if kind=="sense":
        model,kwargs=build_model()
        for split in splits:
            if not (OUTPUT/f"sensevoice_deterministic_features/{split}.json").exists():acquire_split(model,kwargs,split,8)
        return
    model,extractor,head=msp_model()
    for split in splits:
        target=OUTPUT/f"msp_complete_features/{split}.npz"
        if target.exists():continue
        original=PREVIOUS/f"speech_features/{split}.npz"
        with np.load(original,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        present=values["available"].copy();ix=np.flatnonzero(~present)
        for start in range(0,len(ix),8):
            positions=ix[start:start+8];decoded=[decode(ROOT/f"data/public/ch_sims_v2/supervised/{values['sample_ids'][i]}.mp4") for i in positions]
            active=[j for j,r in enumerate(decoded) if r[2]]
            if active:
                p,e=msp_encode(model,extractor,head,[decoded[j][0] for j in active]);dst=positions[active]
                values["speech"][dst]=p;values["speech_emotion"][dst]=e;values["available"][dst]=True
            for j,i in enumerate(positions):values["original_duration"][i]=decoded[j][1]
            gc.collect();torch.cuda.empty_cache()
        with np.load(original,allow_pickle=False) as z:
            if any(not np.array_equal(values[k][present],z[k][present]) for k in values):raise ValueError("旧MSP有效行改变")
        np.savez_compressed(target,**values)
        save(target.with_suffix(".json"),{"n":len(present),"new":int((~present&values["available"]).sum()),
            "old_present_unchanged":True,"sha256":digest(target),"labels_read":False})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("kind",choices=("pts","sense","msp"))
    parser.add_argument("--splits",nargs="+",default=["calibration","test"]);parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.kind=="pts":
        for split in args.splits:pts(split,args.workers)
    else:speech(args.kind,args.splits)
