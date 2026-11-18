"""在新目录复现本轮一个语料的固定全流程，不覆盖原封存工件。"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """复用已核验的公开输入与冻结骨干，重新拟合全部当前语料候选。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=("ch","mosi","meld"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    from experiments import round13_data as data
    target = args.output.resolve()
    if target.exists() or target.parent != data.OUTPUT.parent.resolve():
        raise ValueError("输出须为_国赛优化下尚不存在的新目录，确保旧对照路径仍可解析")
    from experiments import round13_prepare as prepare,round13_train as training,round13_neural as neural
    from experiments import round13_evaluate as evaluation,round13_finish_ch as finish,round13_runtime as runtime
    for module in (data,prepare,training,neural,evaluation,finish,runtime):
        module.OUTPUT = target
    # 默认参数在定义时绑定路径，独立进程中显式改为本次新目的地。
    runtime.CHPredictor.__init__.__defaults__ = (target/"final_ch",)
    if args.dataset=="mosi":
        prepare.mosi()
    elif args.dataset=="meld":
        prepare.meld()
    training.train(args.dataset)
    neural.main(args.dataset)
    if args.dataset=="ch":
        finish.main()
    else:
        evaluation.score(args.dataset)


if __name__ == "__main__":
    main()
