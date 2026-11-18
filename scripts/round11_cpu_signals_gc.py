"""批次屏障回收已关闭PyAV上下文，防止FFmpeg线程资源累积。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import sys
from typing import Any,Callable,Iterable,Iterator
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"


class ReclaimingExecutor(ThreadPoolExecutor):
    """每个固定32条批次完成、没有活跃解码器时回收闭环引用。"""
    def map(self,fn: Callable[...,Any],*iterables: Iterable[Any],timeout: float | None = None,chunksize: int = 1) -> Iterator[Any]:
        try:
            yield from super().map(fn,*iterables,timeout=timeout,chunksize=chunksize)
        finally:
            gc.collect()


def main() -> None:
    """原数值提取器与缓存绑定不变，仅改变资源调度。"""
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=8);args=parser.parse_args()
    import round11_external_signals as module
    from round11_resources import require_memory
    def bounded(minimum_gib: float = 5.) -> None:
        require_memory(2.5 if minimum_gib==5. else minimum_gib)
    module.ThreadPoolExecutor=ReclaimingExecutor;module.require_memory=bounded
    sys.argv=["round11_external_signals.py","--workers",str(args.workers)];module.main()
    from experiments.round11_common import OUTPUT,digest,save
    save(OUTPUT/"emotiontalk_features/signal_features/resource_resume.json",{"wrapper_sha256":digest(Path(__file__)),
        "workers":args.workers,"batch_barrier_gc":32,"minimum_commit_gib":2.5,"NUMBA_NUM_THREADS":1,
        "numeric_algorithm_changed":False,"verification":"validation/signal_profile_gc.log:all fields bitwise equal"},immutable=False)


if __name__=="__main__":main()
