"""只读概念证据：默认拒绝未经人工审核的条目，不生成个人建议。

边界（来自 round19 协议）：
- 图谱 63 节点 / 56 边全部为 pending，专家审核 0 条；默认不返回任何条目；
- 研究使用须同时显式 ``research=True`` 与 ``include_unreviewed=True``；
- 同主体/关系/客体/范围出现正反断言时，相关边从结果中隔离；
- 只返回公开概念与出处，不写入用户事实，也不参与风险判定、升级或安全阈值。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "support_knowledge"
SCHEMA = "aic.support_concepts.v2"
GOALS = {"listen", "understand", "act"}
SETTINGS = {"general_concept", "parent_child"}
_LOCK = threading.RLock()
_CACHE: dict[str, Any] = {}


def sha256_file(path: Path) -> str:
    """流式计算图谱文件摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _conflicted_edges(graph: dict[str, Any]) -> set[str]:
    """仅同一范围出现正反断言才隔离，不把不同场景差异当矛盾。"""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for edge in graph["edges"]:
        key = json.dumps([edge["subject"], edge["predicate"], edge["object"], edge.get("scope")],
                         sort_keys=True, ensure_ascii=False)
        buckets.setdefault(key, []).append(edge)
    return {edge["id"] for rows in buckets.values() if len({row["polarity"] for row in rows}) > 1 for edge in rows}


def _pattern(alias: str) -> re.Pattern[str]:
    """英文别名按词边界匹配，中文按子串匹配。"""
    if re.fullmatch(r"[A-Za-z -]+", alias):
        return re.compile(r"\b" + re.escape(alias) + r"\b", re.I)
    return re.compile(re.escape(alias), re.I)


def _negated(text: str, start: int) -> bool:
    """命中别名前若是明确否定、引述或假设，则不把该概念当作本轮目标。"""
    prefix = text[max(0, start - 16):start]
    return bool(re.search(r"不想|不要|不用|不需要|拒绝|朋友说|他说|她说|假设|假如|"
                          r"don't want|do not want|friend said", prefix, re.I))


class SupportKnowledge:
    """按别名检索公开概念，附出处与审核状态；不做语义推断。"""

    def __init__(self, folder: Path | None = None) -> None:
        directory = Path(folder or ARTIFACTS)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        raw = (directory / "graph.json").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != manifest.get("graph_sha256"):
            raise ValueError("知识图谱摘要与清单不符")
        graph = json.loads(raw.decode("utf-8"))
        if graph.get("schema") != SCHEMA:
            raise ValueError("知识图谱 schema 不符")
        ids = [node["id"] for node in graph["nodes"]]
        edge_ids = [edge["id"] for edge in graph["edges"]]
        if len(ids) != len(set(ids)) or len(edge_ids) != len(set(edge_ids)):
            raise ValueError("知识图谱存在重复 ID")
        for edge in graph["edges"]:
            if edge["subject"] not in ids or edge["object"] not in ids:
                raise ValueError("知识图谱存在悬空关系")
            if edge.get("source_id") not in graph["sources"]:
                raise ValueError("知识图谱关系缺少来源")
        self.graph = graph
        self.manifest = manifest
        self.nodes = graph["nodes"]
        self.edges = graph["edges"]
        self.by_id = {node["id"]: node for node in self.nodes}
        self.conflicted = _conflicted_edges(graph)
        self.pending = sum(1 for item in self.nodes + self.edges if item.get("review_status") == "pending")
        self.sha256 = digest

    def _allowed(self, item: dict[str, Any], research: bool, include_unreviewed: bool) -> bool:
        status = item.get("review_status")
        if status == "rejected":
            return False
        if status == "approved":
            return True
        return bool(status == "pending" and research and include_unreviewed)

    def evidence(self, query: str, goal: str, *, setting: str = "general_concept",
                 budget: int = 4000, research: bool = False, include_unreviewed: bool = False) -> dict[str, Any]:
        """返回完整出处概念块；预算不足时整块舍弃，不截断引用。"""
        base = {"graph_sha256": self.sha256, "nodes": len(self.nodes), "edges": len(self.edges),
                "pending_items": self.pending, "not_personalized_advice": True,
                "clinical_inference": False, "production_connected": False}
        if goal == "pause":
            return {**base, "status": "abstain", "reason": "explicit_goal", "evidence": []}
        if goal not in GOALS:
            return {**base, "status": "abstain", "reason": "invalid_goal", "evidence": []}
        if not isinstance(query, str) or not query.strip():
            return {**base, "status": "abstain", "reason": "empty_query", "evidence": []}
        if not 100 <= budget <= 8000:
            raise ValueError("证据预算需在 100–8000 字符之间")
        if setting not in SETTINGS:
            raise ValueError("未知的来源场景")
        if re.search(r"诊断|确诊|用药|药量|吃药|药物|治愈|抑郁症|焦虑症|dosage|medication", query, re.I):
            return {**base, "status": "abstain", "reason": "outside_concept_scope", "evidence": []}
        clean = re.sub(r'“[^”]*”|「[^」]*」|"[^"\n]*"', lambda m: " " * len(m.group()), query)
        hit_pending = 0
        hit_budget = 0
        blocks: list[dict[str, Any]] = []
        size = 0
        for node in sorted(self.nodes, key=lambda item: item["id"]):
            if node.get("source_scope") not in {"general_concept", setting}:
                continue
            if node.get("requires_goal") == "act" and goal not in {"act", "understand"}:
                continue
            matches = []
            for alias in node.get("aliases", []):
                for match in _pattern(alias).finditer(clean):
                    if not _negated(clean, match.start()):
                        matches.append(match.start())
            if not matches:
                continue
            if not self._allowed(node, research, include_unreviewed):
                hit_pending += 1
                continue
            facts = []
            for edge in self.edges:
                if node["id"] not in {edge["subject"], edge["object"]} or edge["id"] in self.conflicted:
                    continue
                if edge.get("scope", {}).get("setting") not in {"general_stress_concept", setting}:
                    continue
                if not all(self._allowed(item, research, include_unreviewed)
                           for item in (edge, self.by_id[edge["subject"]], self.by_id[edge["object"]])):
                    continue
                facts.append({"edge_id": edge["id"], "subject": edge["subject"], "predicate": edge["predicate"],
                              "object": edge["object"], "url": self.graph["sources"][edge["source_id"]]["url"],
                              "locator": edge.get("locator"), "scope": edge.get("scope"),
                              "derivation": edge.get("derivation"), "source_sha256": edge.get("source_sha256"),
                              "review_status": edge.get("review_status")})
            if not facts:
                continue
            block = {"concept": node["label"], "definition": node.get("definition"), "concept_id": node["id"],
                     "source_setting": node.get("source_scope"), "review_status": node.get("review_status"),
                     "facts": facts, "not_a_user_fact": True}
            length = len(json.dumps(block, ensure_ascii=False))
            if size + length > budget:
                hit_budget += 1
                continue
            blocks.append(block)
            size += length
        if blocks:
            return {**base, "status": "evidence", "reason": "research_opt_in" if research else "approved_only",
                    "evidence": blocks, "encoded_evidence_chars": size, "conflicts_withheld": len(self.conflicted)}
        reason = "pending_review" if hit_pending else ("budget_exhausted" if hit_budget else "no_match")
        return {**base, "status": "abstain", "reason": reason, "evidence": [],
                "conflicts_withheld": len(self.conflicted)}


def load(folder: Path | None = None) -> SupportKnowledge:
    """按路径与摘要缓存图谱；文件变化时重新加载。"""
    directory = Path(folder or ARTIFACTS)
    path = directory / "graph.json"
    key = (str(directory.resolve()), path.stat().st_mtime_ns)
    with _LOCK:
        if _CACHE.get("key") != key:
            _CACHE.clear()
            _CACHE.update({"key": key, "index": SupportKnowledge(directory)})
        return _CACHE["index"]


def capabilities(folder: Path | None = None) -> dict[str, Any]:
    """返回图谱规模与审核状态，供运维确认默认关闭与待审数量。"""
    index = load(folder)
    return {"graph_schema": index.graph.get("schema"), "graph_version": index.graph.get("version"),
            "nodes": len(index.nodes), "edges": len(index.edges), "pending_items": index.pending,
            "human_expert_reviews": 0, "default_unreviewed_allowed": False,
            "research_opt_in_required": True, "production_connected": False,
            "graph_sha256": index.sha256}
