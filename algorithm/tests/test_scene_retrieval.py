"""C2D2 场景检索模块测试

用确定性假编码器隔离单元测试，不加载 409MB 的 BGE 权重，
保证测试快速且不依赖本地模型目录。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intervention.scene_retrieval import (  # noqa: E402
    SceneHit,
    SceneRetriever,
    build_scene_context,
    reset_scene_retriever,
    resolve_index_dir,
)
import intervention.scene_retrieval as sr_pkg  # noqa: E402
from intervention.scene_retrieval.index import NON_DISTORTED_ZH  # noqa: E402


class _FakeEmbedder:
    """确定性假编码器：字符袋哈希 → 定长向量。

    相同文本得到相同向量；共享字符多的文本余弦相似度更高。
    足够验证检索排序与聚合逻辑，且完全离线、瞬时完成。

    ⚠️ 仅用于测试，绝不可用于生产（无任何语义能力）。
    """

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts, *, batch_size: int = 32, is_query: bool = False):
        del batch_size, is_query
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for ch in str(text):
                out[i, ord(ch) % self._dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def _entry(row_id, scene_key, scene, text, label, label_zh):
    return {
        "row_id": row_id,
        "scene_key": scene_key,
        "scene": scene,
        "text": text,
        "label": label,
        "label_zh": label_zh,
        "label_en": label_zh,
    }


@pytest.fixture()
def retriever():
    """构造一个包含 3 个场景的小索引。"""
    entries = [
        _entry(1, "s1", "考试没考好", "我就是个废物", 3, "乱贴标签"),
        _entry(2, "s1", "考试没考好", "我什么都做不好", 5, "过度泛化"),
        _entry(3, "s1", "考试没考好", "这次没考好下次努力就好", 7, NON_DISTORTED_ZH),
        _entry(4, "s2", "没人理我", "他们肯定讨厌我", 0, "读心术"),
        _entry(5, "s3", "妈妈骂我", "她根本不爱我", 2, "情绪化推理"),
    ]
    embedder = _FakeEmbedder()
    vectors = embedder.encode([e["text"] for e in entries])
    return SceneRetriever(embedder, entries, vectors)


class TestConstruction:
    """构造与校验"""

    def test_basic_properties(self, retriever):
        assert retriever.size == 5
        assert retriever.dim == 32
        assert retriever.scene_count == 3

    def test_length_mismatch_raises(self):
        embedder = _FakeEmbedder()
        with pytest.raises(ValueError, match="行数不一致"):
            SceneRetriever(embedder, [_entry(1, "s1", "a", "b", 0, "读心术")], np.zeros((2, 32)))


class TestSearch:
    """检索行为"""

    def test_returns_scene_hit_objects(self, retriever):
        hits = retriever.search("我就是个废物")
        assert hits
        assert all(isinstance(h, SceneHit) for h in hits)

    def test_scores_descending(self, retriever):
        hits = retriever.search("考试没考好我就是个废物", top_k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_exact_text_ranks_first(self, retriever):
        hits = retriever.search("我就是个废物", top_k=1)
        assert hits[0].text == "我就是个废物"
        assert hits[0].label_zh == "乱贴标签"

    def test_top_k_limit(self, retriever):
        hits = retriever.search("考试没考好", top_k=2)
        assert len(hits) == 2

    def test_max_per_scene(self, retriever):
        """同一场景最多取 max_per_scene 条，避免大场景刷屏。"""
        hits = retriever.search("考试没考好我什么都做不好", top_k=5, max_per_scene=1)
        scene_keys = [h.scene_key for h in hits]
        assert len(scene_keys) == len(set(scene_keys))

    def test_empty_query_returns_empty(self, retriever):
        assert retriever.search("") == []
        assert retriever.search("   ") == []

    def test_exclude_non_distorted(self, retriever):
        hits = retriever.search(
            "这次没考好下次努力就好", top_k=5, include_non_distorted=False
        )
        assert all(h.label_zh != NON_DISTORTED_ZH for h in hits)

    def test_min_score_filters(self, retriever):
        hits = retriever.search("完全无关的句子xyz", min_score=0.99)
        assert hits == []


class TestVotes:
    """标签投票"""

    def test_votes_exclude_non_distorted(self, retriever):
        hits = retriever.search("考试没考好我什么都做不好", top_k=5, max_per_scene=3)
        votes = retriever.distortion_votes(hits)
        assert NON_DISTORTED_ZH not in votes

    def test_scene_label_distribution(self, retriever):
        dist = retriever.scene_label_distribution("s1")
        assert dist["乱贴标签"] == 1
        assert dist["过度泛化"] == 1
        assert dist[NON_DISTORTED_ZH] == 1
        assert sum(dist.values()) == 3

    def test_unknown_scene_returns_empty(self, retriever):
        assert retriever.scene_label_distribution("nope") == {}


class TestBuildContext:
    """prompt 上下文构造

    ⚠️ 这些用例只验证**措辞与结构**，因此统一传 ``min_score=0.0``。
    假编码器是字符袋哈希，相似度天然偏低（实测 0.3~0.5），会被生产阈值
    ``MIN_SCENE_SCORE``（0.68，按真实 BGE 实测标定）全部挡掉 —— 那属于
    检索门槛，已由 ``TestDefaultThresholdGate`` 单独覆盖。
    """

    def test_contains_all_sections(self, retriever):
        ctx = retriever.build_context(
            "考试没考好我就是个废物", top_k=5, max_scenes=2, min_score=0.0
        )
        assert "[情境检索]" in ctx
        assert "[使用要求]" in ctx
        assert "- 情境：" in ctx

    def test_includes_scene_and_thought(self, retriever):
        ctx = retriever.build_context(
            "我就是个废物", top_k=3, max_scenes=1, min_score=0.0
        )
        assert "考试没考好" in ctx
        assert "我就是个废物" in ctx

    def test_socratic_requires_question(self, retriever):
        """socratic=True（本轮允许追问）时，只约束语料怎么用、不禁追问。

        断言按分支标记串走，不绑死整句话术 —— 2026-09-26 文案改成
        「接 ta 自己说的那句话」后，原先绑句子的断言一度误报失败。
        """
        ctx = retriever.build_context("我就是个废物", socratic=True, min_score=0.0)
        assert "接 ta 自己说的那句话" in ctx
        assert "本轮不要追问" not in ctx

    def test_empathy_forbids_question(self, retriever):
        ctx = retriever.build_context("我就是个废物", socratic=False, min_score=0.0)
        assert "本轮不要追问" in ctx
        assert "接 ta 自己说的那句话" not in ctx

    def test_always_forbids_naming_labels(self, retriever):
        """两种模式都必须禁止说出扭曲类型名称（AGENTS.md 禁止诊断性语言）。"""
        for socratic in (True, False):
            ctx = retriever.build_context(
                "我就是个废物", socratic=socratic, min_score=0.0
            )
            assert "不要说出扭曲类型的名称" in ctx

    def test_max_scenes_limits_output(self, retriever):
        ctx = retriever.build_context(
            "考试没考好他们讨厌我妈妈骂我", top_k=5, max_scenes=1, min_score=0.0
        )
        assert ctx.count("- 情境：") == 1

    def test_no_hits_returns_empty(self, retriever):
        assert retriever.build_context("完全无关xyz", min_score=0.999) == ""

    def test_distortion_hint_overrides_votes(self, retriever):
        """8 分类模型判定优先于检索命中统计。"""
        ctx = retriever.build_context(
            "我就是个废物", distortion_hint="过度泛化（置信度 0.62）", min_score=0.0
        )
        assert "过度泛化（置信度 0.62）" in ctx
        assert "8 分类模型输出" in ctx
        assert "检索统计" not in ctx

    def test_falls_back_to_votes_without_hint(self, retriever):
        """模型不可用时必须退回检索投票，保证功能不缺失。"""
        ctx = retriever.build_context("我就是个废物", min_score=0.0)
        assert "检索统计" in ctx
        assert "8 分类模型输出" not in ctx


class TestDefaultThresholdGate:
    """默认阈值闸门 —— 没有「像」的语料绝不注入 prompt。

    背景：``min_score`` 曾默认为 0.0，导致「你好」「嗯嗯」这类零信息输入
    也必然命中 8 条无关语料，而 ``[使用要求]`` 又要求模型借用这些情境细节
    来回应 —— 模型只能对用户没说过的处境表达理解（编造共情）。
    """

    def test_default_threshold_is_not_zero(self):
        from intervention.scene_retrieval.index import MIN_SCENE_SCORE

        assert MIN_SCENE_SCORE > 0.0

    def test_default_threshold_rejects_literal_lookalikes(self):
        """阈值必须高到挡掉"字面像但处境无关"的语料。

        实测：0.68 时「我好难受」(0.710) 会注入「你最近怀孕了」「因背痛醒来」
        「疫情被感染」；0.72 才挡得住。这是真实调用暴露的问题，别再调低。
        """
        from intervention.scene_retrieval.index import MIN_SCENE_SCORE

        assert MIN_SCENE_SCORE >= 0.72

    def test_below_threshold_returns_empty(self, retriever):
        """最佳命中低于阈值时必须返回空串，而不是"凑够几条"。

        用与索引几乎零字符重叠的查询，保证最佳相似度落在阈值之下。
        """
        assert retriever.build_context("完全无关xyz") == ""

    def test_passed_threshold_still_injects(self, retriever):
        """过线时正常注入，且只保留真正过线的条目。"""
        ctx = retriever.build_context("我就是个废物")
        assert ctx
        assert "[情境检索]" in ctx
        assert "[使用要求]" in ctx


class TestPersistence:
    """保存与加载"""

    def test_roundtrip(self, retriever, tmp_path):
        retriever.save(tmp_path, source_csv=None)
        assert (tmp_path / "scene_index.npz").exists()
        assert (tmp_path / "scene_index.json").exists()
        assert (tmp_path / "manifest.json").exists()

        loaded = SceneRetriever.load(tmp_path, embedder=_FakeEmbedder())
        assert loaded.size == retriever.size
        assert loaded.scene_count == retriever.scene_count

        a = retriever.search("我就是个废物", top_k=3)
        b = loaded.search("我就是个废物", top_k=3)
        assert [h.text for h in a] == [h.text for h in b]

    def test_manifest_records_provenance(self, retriever, tmp_path):
        import json

        retriever.save(tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["entries"] == 5
        assert manifest["scenes"] == 3
        assert manifest["dim"] == 32
        assert manifest["seed"] == 42

    def test_load_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="索引不完整"):
            SceneRetriever.load(tmp_path, embedder=_FakeEmbedder())

    def test_dim_mismatch_raises(self, retriever, tmp_path):
        retriever.save(tmp_path)
        with pytest.raises(ValueError, match="维度"):
            SceneRetriever.load(tmp_path, embedder=_FakeEmbedder(dim=16))


class TestSafeWrapper:
    """build_scene_context 的容错行为（/smart-chat 依赖它不抛异常）"""

    def test_returns_empty_when_index_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sr_pkg, "_DEFAULT_INDEX_DIRS", ())
        monkeypatch.setenv("SCENE_INDEX_DIR", str(tmp_path / "does_not_exist"))
        reset_scene_retriever()

        assert resolve_index_dir() is None
        assert build_scene_context("我很失败") == ""

        reset_scene_retriever()

    def test_returns_empty_on_empty_query(self):
        reset_scene_retriever()
        assert build_scene_context("") == ""
        reset_scene_retriever()


class TestResolveIndexDir:
    """索引目录解析顺序"""

    def test_repo_artifacts_path_is_preferred(self):
        """仓库内 artifacts 必须排在开发机路径之前，保证随代码分发可用。"""
        first = sr_pkg._DEFAULT_INDEX_DIRS[0]
        assert first.parts[-2:] == ("artifacts", "scene_index")
        assert first.is_absolute()

    @pytest.mark.skipif(
        not (sr_pkg._DEFAULT_INDEX_DIRS[0] / "scene_index.json").exists(),
        reason="仓库内尚未导出产物（先跑 tools/export_model_artifacts.py）",
    )
    def test_resolves_to_repo_artifacts_when_present(self, monkeypatch):
        monkeypatch.delenv("SCENE_INDEX_DIR", raising=False)
        assert resolve_index_dir() == sr_pkg._DEFAULT_INDEX_DIRS[0]

    def test_env_var_wins(self, tmp_path, monkeypatch):
        (tmp_path / "scene_index.json").write_text("[]", encoding="utf-8")
        (tmp_path / "scene_index.npz").write_bytes(b"")
        monkeypatch.setenv("SCENE_INDEX_DIR", str(tmp_path))
        assert resolve_index_dir() == tmp_path

    def test_env_var_without_index_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sr_pkg, "_DEFAULT_INDEX_DIRS", ())
        monkeypatch.setenv("SCENE_INDEX_DIR", str(tmp_path))
        assert resolve_index_dir() is None
