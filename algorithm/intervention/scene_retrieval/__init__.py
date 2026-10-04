"""C2D2 场景检索 —— 把真实青少年「情境 → 想法 → 认知扭曲」语料检索进对话 prompt。

集成位置：``api/intervention.py`` 的 ``/smart-chat`` 流程，作为紧邻
「记忆召回（步骤 2.6）」的一个兄弟步骤。检索结果注入 ``_call_llm`` 的
system prompt，把原先两条手写 few-shot 示例替换成按用户真实处境捞出来的
真实语料。

为什么是检索不是分类：见 :mod:`intervention.scene_retrieval.index` 模块 docstring。
一句话——92% 的场景对应多种扭曲标签，场景只决定「措辞贴合度」，
扭曲类型由「想法」决定，两者正交。

安全约束（AGENTS.md 红线）：
    - 检索结果只是 prompt 的参考上下文，**不是诊断结论**，不得直接呈现给用户。
    - 注入文本中已显式要求模型「不要说扭曲类型名称、不要下结论」。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from .embedder import (
    BGE_QUERY_INSTRUCTION,
    DEFAULT_MODEL_PATH,
    DashScopeEmbedder,
    Embedder,
    BGEEmbedder,
    get_default_embedder,
)
from .index import MIN_SCENE_SCORE, NON_DISTORTED_ZH, SceneHit, SceneRetriever

logger = logging.getLogger(__name__)

__all__ = [
    "BGE_QUERY_INSTRUCTION",
    "DEFAULT_MODEL_PATH",
    "DashScopeEmbedder",
    "Embedder",
    "BGEEmbedder",
    "MIN_SCENE_SCORE",
    "NON_DISTORTED_ZH",
    "SceneHit",
    "SceneRetriever",
    "build_scene_context",
    "get_default_embedder",
    "get_scene_retriever",
    "reset_scene_retriever",
    "resolve_index_dir",
]

# 索引是生成产物（由 tools/build_scene_index.py 重建，或 tools/export_model_artifacts.py 导入）。
# 解析顺序：环境变量 → **仓库内 artifacts（随代码分发）** → 开发机默认位置。
# parents[2] 即 algorithm/。
_DEFAULT_INDEX_DIRS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[2] / "artifacts" / "scene_index",
    Path(r"D:\Develop\AIC\data\c2d2\scene_index"),
)

_retriever: Optional[SceneRetriever] = None
_retriever_load_failed = False


def resolve_index_dir() -> Optional[Path]:
    """定位可用的索引目录。

    解析顺序：
        1. 环境变量 ``SCENE_INDEX_DIR``
        2. ``<algorithm>/data/c2d2_scene_index``
        3. ``D:\\Develop\\AIC\\data\\c2d2\\scene_index``（开发机）

    Returns:
        Optional[Path]: 含 ``scene_index.json`` 的目录；都没找到则 None。
    """
    env_dir = os.environ.get("SCENE_INDEX_DIR", "").strip()
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(_DEFAULT_INDEX_DIRS)

    for path in candidates:
        if (path / "scene_index.json").exists() and (path / "scene_index.npz").exists():
            return path
    return None


def get_scene_retriever() -> Optional[SceneRetriever]:
    """获取全局检索引擎（懒加载单例）。

    设计取舍：索引缺失或加载失败时返回 ``None`` 而不是抛异常，
    保证 ``/smart-chat`` 主流程不会因为检索不可用而中断
    （与 ``build_memory_context`` 的容错策略一致）。

    Returns:
        Optional[SceneRetriever]: 检索引擎；不可用时为 None。
    """
    global _retriever, _retriever_load_failed

    if _retriever is not None:
        return _retriever
    if _retriever_load_failed:
        return None

    index_dir = resolve_index_dir()
    if index_dir is None:
        logger.info(
            "场景检索索引未找到，跳过场景检索。"
            "请运行 tools/build_scene_index.py 生成，或用 SCENE_INDEX_DIR 指定目录。"
        )
        _retriever_load_failed = True
        return None

    try:
        _retriever = SceneRetriever.load(index_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("场景检索索引加载失败（%s）: %s", index_dir, e)
        _retriever_load_failed = True
        return None

    logger.info(
        "场景检索索引已加载：%d 条想法 / %d 个场景 / %d 维（%s）",
        _retriever.size, _retriever.scene_count, _retriever.dim, index_dir,
    )
    return _retriever


def reset_scene_retriever() -> None:
    """清空单例缓存（测试与索引重建后使用）。"""
    global _retriever, _retriever_load_failed
    _retriever = None
    _retriever_load_failed = False


def build_scene_context(
    query: str,
    *,
    top_k: int = 8,
    max_per_scene: int = 2,
    min_score: float = MIN_SCENE_SCORE,
    max_scenes: int = 3,
    socratic: bool = True,
    distortion_hint: str = "",
) -> str:
    """安全版上下文构造：任何异常都退化为空串。

    供 ``/smart-chat`` 直接调用。

    与 ``SceneRetriever.build_context`` 的区别只有两点：容错（异常 → 空串）
    与**命中情况日志**。日志是刻意留的：阈值 :data:`MIN_SCENE_SCORE` 是按
    离线相似度实测定的，线上真实分布的偏高/偏低只能靠这些日志回看，
    因此"命中了但不是真的像"和"该命中却没命中"都能在日志里看到分数。
    Args:
        query: 用户当前发言。
        top_k: 检索条数上限。
        max_per_scene: 同一场景最多取几条。
        min_score: 余弦相似度下限，默认 :data:`MIN_SCENE_SCORE`。
        max_scenes: 展示的场景数上限。
        socratic: 是否允许在追问里带苏格拉底式反问。
        distortion_hint: 8 分类模型产出的扭曲倾向；非空时优先于检索投票。

    Returns:
        str: 可注入 system prompt 的上下文块；不可用或相关度不足时为空串。
    """
    preview = (query or "").strip()[:30]
    try:
        retriever = get_scene_retriever()
        if retriever is None:
            return ""
        ctx = retriever.build_context(
            query,
            top_k=top_k,
            max_per_scene=max_per_scene,
            min_score=min_score,
            max_scenes=max_scenes,
            socratic=socratic,
            distortion_hint=distortion_hint,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("场景检索失败，已跳过: %s", e)
        return ""

    if not ctx:
        # 未注入：把最佳相似度记下来，供回看阈值是否偏严
        try:
            hits = retriever.search(query, top_k=1, max_per_scene=1, min_score=0.0)
            best = hits[0].score if hits else 0.0
        except Exception:  # noqa: BLE001
            best = -1.0
        logger.info(
            "[场景检索] 未注入（最佳相似度 %.4f < 阈值 %.2f）query=%r",
            best, min_score, preview,
        )
        return ""

    logger.info(
        "[场景检索] 已注入 %d 个情境 / 阈值 %.2f query=%r",
        ctx.count("- 情境："), min_score, preview,
    )
    return ctx
