"""实测 C2D2 场景索引相关度分布，复现 MIN_SCENE_SCORE 阈值的取值依据。

用途：改动检索阈值（``intervention/scene_retrieval/index.py`` 的
``MIN_SCENE_SCORE``）后，用它回看真实相似度分布，确认"寒暄被挡掉、
真实困扰被保留"这条界线是否还成立。

用法（从 ``algorithm/`` 目录运行）：

    python tools/measure_scene_threshold.py

只读：不修改索引、不写入任何文件；会调用一次 embedding
（默认本地 BGE，未配置本地模型时走 DashScope）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 允许直接以脚本方式从 algorithm/ 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intervention.scene_retrieval import (  # noqa: E402
    MIN_SCENE_SCORE,
    get_default_embedder,
    resolve_index_dir,
)
from intervention.scene_retrieval.index import SceneRetriever  # noqa: E402

# (类别, 话术)。类别只用于阅读输出，不参与计算。
CASES: list[tuple[str, str]] = [
    ("寒暄", "你好"),
    ("寒暄", "你好呀"),
    ("寒暄", "在吗"),
    ("寒暄", "谢谢"),
    ("灰区", "还行吧"),
    ("灰区", "我今天有些伤心"),
    ("灰区", "我有点累"),
    ("灰区", "为什么我会这样"),
    ("真实", "我好难受"),
    ("真实", "我觉得没人理解我"),
    ("真实", "我一想到明天要去学校就心慌，喘不上气"),
    ("真实", "我这次考试又没考好，我觉得自己特别笨，怎么努力都没用"),
    ("真实", "爸妈吵架都是因为我，都是我的错"),
]

THRESHOLDS = (0.65, 0.68, 0.70, 0.72, 0.75, 0.80)


def main() -> int:
    index_dir = resolve_index_dir()
    if index_dir is None:
        print("未找到场景索引，请先运行 tools/build_scene_index.py 或用 SCENE_INDEX_DIR 指定。")
        return 1

    retriever = SceneRetriever.load(index_dir)
    vectors = retriever._vectors
    embedder = retriever._embedder
    print(f"索引目录: {index_dir}")
    print(f"索引规模: {retriever.size} 条想法 / {retriever.scene_count} 场景 / dim={retriever.dim}")
    print(f"编码器:   {type(embedder).__name__}")
    print(f"当前阈值: MIN_SCENE_SCORE = {MIN_SCENE_SCORE}")
    print("=" * 84)

    scores_by_case: list[tuple[str, str, float, float]] = []
    for kind, text in CASES:
        qvec = embedder.encode([text], is_query=True)[0]
        scores = vectors @ qvec
        top1 = float(np.max(scores))
        top8 = float(np.sort(scores)[-8])
        scores_by_case.append((kind, text, top1, top8))
        print(f"{kind}  {text[:32]:34s} top1={top1:.4f}  top8={top8:.4f}")

    print("=" * 84)
    print(f"各阈值下 top-8 内过线条数（当前阈值 {MIN_SCENE_SCORE} 应让寒暄全为 0）")
    header = "  ".join(f"{t:.2f}" for t in THRESHOLDS)
    print(f"{'query':36s}{header}")
    for kind, text, top1, _top8 in scores_by_case:
        qvec = embedder.encode([text], is_query=True)[0]
        scores = vectors @ qvec
        counts = [int((scores >= t).sum()) for t in THRESHOLDS]
        print(f"{text[:34]:36s}" + "  ".join(f"{c:>4d}" for c in counts))

    print("=" * 84)
    print("判定：注入与否只看 build_context 的最高分是否 >= MIN_SCENE_SCORE")
    for kind, text, top1, _top8 in scores_by_case:
        verdict = "注入" if top1 >= MIN_SCENE_SCORE else "跳过"
        print(f"  [{verdict}] {text[:34]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
