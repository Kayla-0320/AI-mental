"""资源回收入口与先前原数值提取结果的逐字段核验。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from round11_cpu_signals_gc import ReclaimingExecutor
from round11_external_signals import prepare
import cv2
import numpy as np
import psutil
from experiments.round11_common import OUTPUT,ROOT,digest,save


def main() -> None:
    """固定三个已有开发来源媒体，只验证资源调度不会改变数值。"""
    cv2.setNumThreads(1)
    rows=[json.loads(line) for line in (OUTPUT/"corpus/train.jsonl").read_text("utf-8").splitlines()]
    ids=[rows[i]["id"] for i in (0,500,2500)]
    before=psutil.Process().num_threads()
    with ReclaimingExecutor(max_workers=2) as pool:values=list(pool.map(prepare,[ROOT/f"data/public/emotiontalk_media/clips/{key}.mp4" for key in ids]))
    checks=[]
    for key,value in zip(ids,values):
        paths=[OUTPUT/f"emotiontalk_features/signal_features/{name}/{key}.npz" for name in ("samples_v2_bounded","samples")]
        source=next(p for p in paths if p.exists())
        with np.load(source,allow_pickle=False) as saved:
            fields={k:bool(np.array_equal(v,saved[k])) for k,v in value.items()}
        if not all(fields.values()):raise ValueError("资源回收改变了原特征数值")
        checks.append({"id":key,"fields_bitwise_equal":fields,"reference_cache_sha256":digest(source)})
    save(OUTPUT/"validation/signal_batch_gc_verification.json",{"checks":checks,"threads_before":before,
        "threads_after_batch":psutil.Process().num_threads(),"numeric_extractor_sha256":digest(Path(__file__).with_name("round11_external_signals.py")),
        "resource_wrapper_sha256":digest(Path(__file__).with_name("round11_cpu_signals_gc.py")),"target_labels_used":False},immutable=False)
    print(json.dumps({"verified_media":len(checks),"all_fields_bitwise_equal":True}),flush=True)


if __name__=="__main__":main()
