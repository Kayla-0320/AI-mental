"""单个CPU媒体任务与小CLIP协作的有界资源入口，数值提取器保持原样。"""
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import sys


def main() -> None:
    """四个单线程解码器，保留2.5GiB余量；不完整采样的6GiB重验仍强制。"""
    parser=argparse.ArgumentParser();parser.add_argument("kind",choices=("faces","signals"));parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    from round11_resources import require_memory as original
    module=importlib.import_module(f"round11_external_{args.kind}")
    def bounded(minimum_gib: float = 5.) -> None:
        original(2.5 if minimum_gib==5. else minimum_gib)
    module.require_memory=bounded
    sys.argv=[f"round11_external_{args.kind}.py","--workers",str(args.workers)]
    module.main()
    from experiments.round11_common import OUTPUT,digest,save
    save(OUTPUT/f"emotiontalk_features/{'face_features' if args.kind=='faces' else 'signal_features'}/resource_resume.json",
        {"wrapper_sha256":digest(Path(__file__)),"workers":args.workers,"minimum_commit_gib":2.5,
         "partial_sampling_repeat_minimum_gib":6.,"numeric_algorithm_changed":False},immutable=False)


if __name__=="__main__":main()
