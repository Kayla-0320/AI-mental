"""检查像素退化的保守离线修复，保留暗光和遮挡中的有效视觉信息。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
from experiments.round14_common import freeze,save,sha256
from experiments.round18_images import OUT,CONDITIONS,altered
from experiments.round17_pixels import frames
from experiments.round17_quality import MEDIA,score


def unusable(pixels: np.ndarray) -> bool:
    """仅无帧或每帧空间完全恒定才降级；不按明暗、类别分歧或标签降级。"""
    if not len(pixels):return True
    if pixels.ndim!=4 or pixels.shape[-1]!=3:raise ValueError("需要实际RGB帧")
    spread=np.ptp(pixels.astype(float),axis=(1,2))
    return bool(np.all(spread==0))


def run() -> None:
    """在负结果出现后冻结修复；同批开发素材的探索结果不冒充独立验证。"""
    freeze(OUT/"conservative_protocol.json",{"rule":"drop only absent or spatially constant RGB in every sampled frame","post_inspection_repair":True,"source_sha256":sha256(Path(__file__)),"fresh_independent_test":False})
    z=np.load(OUT/"reencoded.npz",allow_pickle=False)
    with ThreadPoolExecutor(max_workers=6) as pool:rgb=list(pool.map(lambda i:frames(MEDIA/(str(i)+".mp4")),z["sample_ids"]))
    result={}
    # T头在旧质量规则拒绝视觉时已被保存，所有受控损坏均为同一干净训练模型。
    text=np.load(OUT/"black_predictions.npz",allow_pickle=False)["quality"]
    for c in CONDITIONS:
        old=np.load(OUT/(c+"_predictions.npz"),allow_pickle=False)
        drop=np.array([unusable(altered(x,c)) if len(x) else True for x in rgb])
        selected=old["uniform"].copy();selected[drop]=text[drop]
        np.savez_compressed(OUT/(c+"_guard.npz"),truth=z["labels"],groups=z["groups"],selected=selected,drop=drop)
        result[c]={"selected":score(z["labels"],selected),"drop_count":int(drop.sum())}
    save(OUT/"conservative_results.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":run()
