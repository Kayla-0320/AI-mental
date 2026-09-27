"""pytest 启动前的环境约束。

⚠️ 本文件必须在任何测试模块 import numpy/torch 之前被加载 —— pytest 会在收集
测试用例前先 import 根目录的 conftest.py，因此这里设置的环境变量能赶在
OpenBLAS 初始化线程池之前生效。

为什么这么做：本机 20 逻辑核，OpenBLAS 默认按核数为每个线程预留一块内存
arena，import torch/numpy 时一次性申请数十 GB 提交内存，即便物理内存充裕也会
抛 "OpenBLAS Error: Memory allocation still failed after 10 retries"。把 BLAS
线程数压到 1 即可根治（测试不靠 BLAS 并行提速）。

只影响测试进程：生产后端由 run-algorithm 守护、不经过 pytest，不会加载本文件，
推理仍可用多核。
"""
import os

# 必须在 import numpy 之前设置，故放在模块最顶部、且用 setdefault 不覆盖显式指定。
for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
