"""C2D2 场景检索索引 —— 把「情境 → 自动思维 → 认知扭曲」三元组库变成可检索的 few-shot 来源。

为什么是检索而不是分类
----------------------
本地 C2D2 数据实测：276 个场景中**只有 22 个（8.0%）只含单一扭曲标签**，
即 92% 的场景对应多种扭曲。同一处境下不同的人产生不同想法，
**扭曲类型由「想法」决定，不由「处境」决定**。两根轴正交：

    - 场景（情境）= 发生了什么  → 决定提问贴合什么具体处境（措辞）
    - 扭曲（标签）= 怎么解读的  → 决定提问往哪个方向走（苏格拉底策略）

同时 276 类里 18 个场景只出现 1 次、109 个 ≤10 次，分类器不可训。
因此本模块在**想法级别**建索引（7404 条），查询时聚合回场景，
既拿到贴近用户措辞的例句，又拿到该处境的扭曲分布。

设计约束：
    - 索引是**生成产物**，由 ``tools/build_scene_index.py`` 从清洗后的
      ``c2d2_clean.csv`` 重建，不入库。
    - 检索结果只作为 prompt 的**参考上下文**，不是诊断结论。
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .embedder import Embedder, get_default_embedder

# 非扭曲类别名（用于 prompt 里区分「健康反例」与「扭曲倾向」）
NON_DISTORTED_ZH = "非扭曲"

_INDEX_FILE = "scene_index.npz"
_ENTRIES_FILE = "scene_index.json"
_MANIFEST_FILE = "manifest.json"

# 场景相关度门槛（query 与索引条目的余弦相似度）。
#
# 为什么不能再用 0.0：min_score=0 时**任何非空输入都必然命中**，
# 「你好」「嗯嗯」这种零信息输入也会捞回 8 条与自身处境无关的语料，
# 而 [使用要求] 又要求模型「借用上述情境的具体细节来回应」——
# 结果是模型被迫对用户没说过的处境表达理解，读起来像在编造共情。
#
# 阈值取值依据（本地 BGE 编码器 + 7404 条 C2D2 索引实测，
# 复现脚本：`tools/measure_scene_threshold.py`）：
#
#     query                         top1    >=0.72 的条数
#     你好                          0.581    0
#     你好呀                        0.624    0
#     在吗                          0.631    0
#     谢谢                          0.564    0
#     还行吧                        0.581    0
#     为什么我会这样                 0.672    0
#     我一想到明天要去学校就心慌      0.670    0
#     我今天有些伤心                 0.652    0
#     我好难受                      0.710    0
#     我觉得没人理解我               0.750   20
#     我这次考试又没考好…            0.795   59
#     爸妈吵架都是因为我…            0.805   33
#
# 0.72 的依据是一次真实调用暴露的问题：0.68 能挡住寒暄，却挡不住
# **字面相似但处境无关**的语料 —— 「我好难受」在 0.710 命中并注入了
# 「你最近怀孕了，感觉很难受」「每周因背痛醒来」「疫情被感染」。
# 实测这些噪声没有泄漏进回复（模型忽略了它们），但也没有任何增益，
# 白烧 token；而 0.75 以上的命中（被试责、被否定、考试失败）才是
# 真正处境相近、值得作为 few-shot 的语料。
#
# 取舍：宁可偏严。少注入一次只是少一次措辞贴合，
# 误注入一次就是让模型对用户没说过的处境表达理解（编造共情）。
MIN_SCENE_SCORE = 0.72


@dataclass(frozen=True)
class SceneHit:
    """一条检索命中（想法级别）。

    Attributes:
        row_id: 在清洗后 CSV 中的原始 ``num`` 字段。
        score: 余弦相似度。
        scene_key: 场景分组键（``scene_<sha1前12位>``）。
        scene: 场景原文（情境描述）。
        text: 想法原文（自动思维）。
        label: 扭曲标签 id。
        label_zh: 扭曲标签中文名。
        label_en: 扭曲标签英文名。
    """

    row_id: int
    score: float
    scene_key: str
    scene: str
    text: str
    label: int
    label_zh: str
    label_en: str


class SceneRetriever:
    """C2D2 场景/想法检索器。

    典型用法::

        retriever = SceneRetriever.load(index_dir)          # 或 build_from_csv(...)
        hits = retriever.search("我觉得我什么都做不好", top_k=8)
        context = retriever.build_context("我觉得我什么都做不好")
    """

    def __init__(
        self,
        embedder: Embedder,
        entries: Sequence[dict],
        vectors: np.ndarray,
    ) -> None:
        """
        Args:
            embedder: 向量编码器（必须与建索引时同款、同配置）。
            entries: 与 ``vectors`` 行序一一对应的元数据。
            vectors: shape ``(len(entries), dim)`` 的 L2 归一化向量矩阵。
        """
        if len(entries) != len(vectors):
            raise ValueError(
                f"entries({len(entries)}) 与 vectors({len(vectors)}) 行数不一致"
            )
        self._embedder = embedder
        self._entries = list(entries)
        self._vectors = vectors
        # 场景 → 标签分布（建索引时懒计算并缓存）
        self._scene_label_dist: Optional[dict[str, Counter]] = None

    # ----------------------------------------------------------
    # 属性
    # ----------------------------------------------------------

    @property
    def size(self) -> int:
        """索引条目数（想法级别）。"""
        return len(self._entries)

    @property
    def dim(self) -> int:
        """向量维度。"""
        return int(self._vectors.shape[1])

    @property
    def scene_count(self) -> int:
        """去重后的场景数。"""
        return len({e["scene_key"] for e in self._entries})

    # ----------------------------------------------------------
    # 构建 / 持久化
    # ----------------------------------------------------------

    @classmethod
    def build_from_csv(
        cls,
        csv_path: str | Path,
        embedder: Optional[Embedder] = None,
        *,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> "SceneRetriever":
        """从清洗后的 ``c2d2_clean.csv`` 构建索引。

        Args:
            csv_path: ``data/c2d2/c2d2_clean.csv`` 路径。
            embedder: 编码器，默认本地 BGE。
            batch_size: 编码批大小。
            verbose: 是否打印进度。

        Returns:
            SceneRetriever: 建好的检索器。
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"未找到 C2D2 清洗数据: {csv_path}")

        embedder = embedder or get_default_embedder()

        entries: list[dict] = []
        with csv_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                # 清洗后的 CSV 沿用 ``subject_key`` 作为场景分组键
                # （该列名是为了兼容 training 侧的 load_dataset/make_group_folds），
                # 这里同时接受 ``scene_key`` 以便复用其它来源的表。
                scene_key = row.get("scene_key") or row.get("subject_key") or ""
                entries.append(
                    {
                        "row_id": int(row["num"]),
                        "scene_key": scene_key,
                        "scene": row["scene"],
                        "text": row["text"],
                        "label": int(row["label"]),
                        "label_zh": row["label_zh"],
                        "label_en": row["label_en"],
                    }
                )

        if not entries:
            raise ValueError(f"{csv_path} 中没有任何数据行")

        texts = [e["text"] for e in entries]
        t0 = time.time()
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunks.append(embedder.encode(texts[start:start + batch_size]))
            if verbose:
                done = min(start + batch_size, len(texts))
                print(f"  编码进度 {done}/{len(texts)}", end="\r", flush=True)

        vectors = np.vstack(chunks).astype(np.float32)
        if verbose:
            print(f"\n  编码完成：{vectors.shape}，耗时 {time.time() - t0:.1f}s")

        return cls(embedder, entries, vectors)

    def save(self, out_dir: str | Path, *, source_csv: Optional[str | Path] = None) -> None:
        """把索引写入目录。

        Args:
            out_dir: 输出目录（不存在则创建）。
            source_csv: 数据来源路径，写入 manifest 便于溯源。
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(out_dir / _INDEX_FILE, vectors=self._vectors)
        (out_dir / _ENTRIES_FILE).write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

        manifest: dict = {
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "entries": self.size,
            "scenes": self.scene_count,
            "dim": self.dim,
            "seed": 42,
            "embedder": type(self._embedder).__name__,
            "embedder_model_path": getattr(self._embedder, "model_path", None),
            "label_distribution": dict(
                Counter(e["label_zh"] for e in self._entries).most_common()
            ),
        }
        if source_csv is not None:
            src = Path(source_csv)
            manifest["source_csv"] = str(src)
            if src.exists():
                manifest["source_csv_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()

        (out_dir / _MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        embedder: Optional[Embedder] = None,
    ) -> "SceneRetriever":
        """从目录加载索引。

        Args:
            index_dir: ``save`` 写入的目录。
            embedder: 编码器，默认本地 BGE（必须与建索引时一致）。

        Returns:
            SceneRetriever: 加载好的检索器。
        """
        index_dir = Path(index_dir)
        vectors_path = index_dir / _INDEX_FILE
        entries_path = index_dir / _ENTRIES_FILE

        if not vectors_path.exists() or not entries_path.exists():
            raise FileNotFoundError(
                f"索引不完整，缺少 {_INDEX_FILE} 或 {_ENTRIES_FILE}（目录: {index_dir}）。"
                f"请先运行 tools/build_scene_index.py 生成索引。"
            )

        with np.load(vectors_path) as data:
            vectors = data["vectors"].astype(np.float32)
        entries = json.loads(entries_path.read_text(encoding="utf-8"))

        embedder = embedder or get_default_embedder()
        if embedder.dim != vectors.shape[1]:
            raise ValueError(
                f"编码器维度({embedder.dim})与索引维度({vectors.shape[1]})不一致，"
                f"请使用与建索引时相同的编码器"
            )
        return cls(embedder, entries, vectors)

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        max_per_scene: int = 2,
        min_score: float = 0.0,
        include_non_distorted: bool = True,
    ) -> list[SceneHit]:
        """检索与 ``query`` 最接近的想法。

        Args:
            query: 用户当前发言。
            top_k: 返回条数上限。
            max_per_scene: 同一场景最多取几条，避免单个大场景刷屏。
            min_score: 余弦相似度下限。
            include_non_distorted: 是否保留「非扭曲」条目
                （它们是有用的健康反例，可喂给替代想法环节）。

        Returns:
            list[SceneHit]: 按相似度降序。
        """
        query = (query or "").strip()
        if not query:
            return []

        qvec = self._embedder.encode([query], is_query=True)[0]
        scores = self._vectors @ qvec  # 两侧均已 L2 归一化，点积即余弦

        order = np.argsort(-scores)
        hits: list[SceneHit] = []
        per_scene: Counter = Counter()

        for idx in order:
            if len(hits) >= top_k:
                break
            score = float(scores[idx])
            if score < min_score:
                break
            entry = self._entries[int(idx)]
            if not include_non_distorted and entry["label_zh"] == NON_DISTORTED_ZH:
                continue
            key = entry["scene_key"]
            if per_scene[key] >= max_per_scene:
                continue
            per_scene[key] += 1
            hits.append(SceneHit(score=score, **entry))

        return hits

    def scene_label_distribution(self, scene_key: str) -> Counter:
        """返回某场景内的扭曲标签分布。

        Args:
            scene_key: 场景分组键。

        Returns:
            Counter: ``{标签中文名: 条数}``。
        """
        if self._scene_label_dist is None:
            dist: dict[str, Counter] = {}
            for e in self._entries:
                dist.setdefault(e["scene_key"], Counter())[e["label_zh"]] += 1
            self._scene_label_dist = dist
        return self._scene_label_dist.get(scene_key, Counter())

    def distortion_votes(self, hits: Iterable[SceneHit]) -> Counter:
        """统计命中条目里的扭曲标签分布（排除「非扭曲」）。

        Args:
            hits: ``search`` 的返回值。

        Returns:
            Counter: ``{标签中文名: 条数}``，按条数降序。
        """
        votes: Counter = Counter()
        for h in hits:
            if h.label_zh != NON_DISTORTED_ZH:
                votes[h.label_zh] += 1
        return Counter(dict(votes.most_common()))

    # ----------------------------------------------------------
    # prompt 注入
    # ----------------------------------------------------------

    def build_context(
        self,
        query: str,
        *,
        top_k: int = 8,
        max_per_scene: int = 2,
        min_score: float = MIN_SCENE_SCORE,
        max_scenes: int = 3,
        socratic: bool = True,
        distortion_hint: str = "",
    ) -> str:
        """检索并格式化为可注入 system prompt 的上下文块。

        风格与 ``assessment.memory_extractor.build_memory_context`` 保持一致：
        方括号分节 + 末尾给出明确使用要求。

        Args:
            query: 用户当前发言。
            top_k: 检索条数上限。
            max_per_scene: 同一场景最多取几条。
            min_score: 余弦相似度下限，默认 :data:`MIN_SCENE_SCORE`。
                **不再允许 0**：见该常量的取值依据。
            max_scenes: 展示的场景数上限。
            socratic: 是否允许在追问里带苏格拉底式反问。
                False（用户情绪激动/偏内向，或本轮不适合追问）时，
                要求模型**不要追问**，只做共情反映。
            distortion_hint: 由 8 分类模型产出的扭曲倾向提示串。
                **非空时优先于检索标签投票** —— 模型逐条判定比"命中条目的统计"
                精度更高。为空则回退到投票，保证模型不可用时功能不缺失。

        Returns:
            str: 格式化上下文；最佳命中低于 ``min_score`` 或索引不可用时返回空串。
        """
        hits = self.search(
            query, top_k=top_k, max_per_scene=max_per_scene, min_score=min_score
        )
        if not hits:
            return ""

        # 最佳命中仍低于门槛 —— 说得上「像」才注入
        if hits[0].score < min_score:
            return ""
        # 窄化到真正过线的条目，避免把 0.4 分的语料当"相近处境"用
        hits = [h for h in hits if h.score >= min_score]
        if not hits:
            return ""

        # 按场景聚合，保持相似度降序
        grouped: "OrderedDict[str, list[SceneHit]]" = OrderedDict()
        for h in hits:
            grouped.setdefault(h.scene_key, []).append(h)

        lines = [
            "[情境检索] 以下是从青少年认知扭曲语料库（C2D2）中检索到的、"
            "与该用户当前表达最接近的真实情境与想法："
        ]
        for scene_key, group in list(grouped.items())[:max_scenes]:
            lines.append(f"- 情境：{group[0].scene}")
            for h in group:
                lines.append(f"  相近想法：「{h.text}」→ {h.label_zh}")

        # 「扭曲」这一轴：优先用 8 分类模型逐条判定的结果，
        # 模型不可用时回退到「命中条目的标签统计」（精度更低，只是软提示）。
        if distortion_hint:
            lines.append(
                f"[认知扭曲倾向] {distortion_hint}"
                f"（8 分类模型输出，仅供参考，不是诊断结论）"
            )
        else:
            votes = self.distortion_votes(hits)
            if votes:
                vote_str = "、".join(f"{name}×{n}" for name, n in votes.most_common(4))
                lines.append(
                    f"[认知扭曲倾向] {vote_str}（检索统计，仅供参考，不是诊断结论）"
                )

        # 使用要求不再按模式二分：话术形态由 system prompt 统一规定
        # （接住具体内容、最多一个追问、不套公式），这里只约束「怎么用这些语料」。
        # 两个分支各留一个固定标记串，测试据此断言"本轮到底允不允许追问"：
        #   socratic=True  → "接 ta 自己说的那句话"
        #   socratic=False → "本轮不要追问"
        if socratic:
            question_rule = (
                "3. 这些情境只是帮你听懂 ta 在说什么，**不要拿它们去替用户定义处境**；"
                "要接的话，接 ta 自己说的那句话，不要接语料库里的句子。"
            )
        else:
            question_rule = (
                "3. 本轮用户情绪未平复或不愿多谈，只回应 ta 说的内容，"
                "先让用户感到被听到，本轮不要追问。"
            )
        requirement = (
            "[使用要求]\n"
            "1. 只有当上面的情境确实贴近用户说的内容时，才可以借用其具体细节，"
            "让用户感到你听懂了；**不要照搬原句、不要罗列、更不要提及语料库**。\n"
            "2. 绝对不要说出扭曲类型的名称、不要给建议、不要替用户下结论、不要说教。\n"
            "3. **不许拿这些例子去描述用户此刻的状态。** 上面的情境是别人的，"
            "不是 ta 说的；说出「你看起来有点没精神」「你是不是最近很累」这类"
            "观察句就是凭空捏造。只有用户自己提过的内容才能出现在你的回复里。\n"
            f"{question_rule}"
        )
        lines.append(requirement)
        return "\n".join(lines)

    # ----------------------------------------------------------
    # 诊断
    # ----------------------------------------------------------

    def stats(self) -> dict:
        """返回索引概览统计。"""
        return {
            "entries": self.size,
            "scenes": self.scene_count,
            "dim": self.dim,
            "embedder": type(self._embedder).__name__,
            "label_distribution": dict(
                Counter(e["label_zh"] for e in self._entries).most_common()
            ),
        }


__all__ = [
    "NON_DISTORTED_ZH",
    "SceneHit",
    "SceneRetriever",
]
