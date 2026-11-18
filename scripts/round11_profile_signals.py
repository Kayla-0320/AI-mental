"""只读排查媒体任务的线程过量，不改变既定特征算法。"""
from __future__ import annotations
import cProfile
import gc
import json
import os
from pathlib import Path
import pstats
import sys
import time
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import psutil
from threadpoolctl import threadpool_info
from experiments.round11_common import OUTPUT,ROOT
from round11_external_signals import prepare
import cv2
import numpy as np


def main() -> None:
    """打印库线程数量与单条时间剖面，仅用于资源调度。"""
    cv2.setNumThreads(1)
    key=json.loads((OUTPUT/"corpus/train.jsonl").read_text("utf-8").splitlines()[0])["id"]
    print("before_threads",psutil.Process().num_threads(),threadpool_info(),flush=True)
    profiler=cProfile.Profile();start=time.perf_counter();profiler.enable()
    result=prepare(ROOT/f"data/public/emotiontalk_media/clips/{key}.mp4")
    profiler.disable()
    print("after_threads",psutil.Process().num_threads(),"seconds",time.perf_counter()-start,threadpool_info(),flush=True)
    from numba import get_num_threads
    print("numba_threads",get_num_threads(),"cpu_count",os.cpu_count(),flush=True)
    for attempt in range(2):
        gc.collect()
        print("gc_threads",psutil.Process().num_threads(),flush=True)
        start=time.perf_counter();result=prepare(ROOT/f"data/public/emotiontalk_media/clips/{key}.mp4")
        print("warm_seconds",time.perf_counter()-start,"threads",psutil.Process().num_threads(),flush=True)
    for directory in ("samples_v2_bounded","samples"):
        path=OUTPUT/f"emotiontalk_features/signal_features/{directory}/{key}.npz"
        if path.exists():
            with np.load(path,allow_pickle=False) as previous:
                print("cache_equal",{k:bool(np.array_equal(v,previous[k])) for k,v in result.items()},flush=True)
            break
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(18)


if __name__=="__main__":main()
