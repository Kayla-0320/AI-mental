"""在全新输出目录重训第十二轮，避免覆盖已锁定研究工件。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    """指定新目录执行原固定实验；输入仍限定于开发来源。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise ValueError("重训输出目录必须尚不存在，避免覆盖锁定结果")
    from experiments import round12_explore as experiment
    experiment.OUTPUT = target
    sys.argv = ["round12_explore"] + (["--smoke"] if args.smoke else [])
    experiment.main()


if __name__ == "__main__":
    main()
