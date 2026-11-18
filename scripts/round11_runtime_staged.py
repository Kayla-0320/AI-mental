"""兼容的3B/7B原媒体基准入口，实际调度独立进程以释放完整上下文。"""
from __future__ import annotations
from round11_runtime_staged_isolated import main


if __name__=="__main__":main()
