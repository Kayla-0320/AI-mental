"""带来源和审核状态的知识候选索引；默认仅允许已审核条目。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


class KnowledgeIndex:
    """确定性检索及散列编码；不使用训练或测试标签生成知识。

    候选模式仅用于离线消融，不能将其叫作已通过专家审核的知识图谱。
    每条关系是可检查的候选概念关联，不代表对具体用户的因果判断。
    """

    def __init__(self, path: Path, allow_candidates: bool = False, dimensions: int = 32,
                 allow_user_attested: bool = False) -> None:
        self.path, self.dimensions = path, dimensions
        if dimensions < 2:
            raise ValueError("知识编码维数至少为2")
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        graph = json.loads(raw)
        if not isinstance(graph, dict) or graph.get("schema_version") != "emotion_knowledge.v1":
            raise ValueError("知识图谱版本不匹配")
        if not isinstance(graph.get("nodes"), list) or not all(isinstance(e, dict) for e in graph["nodes"]):
            raise ValueError("知识节点必须是对象列表")
        attestation = graph.get("review_attestation", {})
        self.attestation_verified = False
        if any(e.get("review_status") == "user_attested" for e in graph["nodes"]):
            payload = json.dumps(graph["nodes"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            if (not isinstance(attestation, dict) or attestation.get("type") != "user_confirmation_of_expert_review"
                    or not isinstance(attestation.get("source_statement"), str) or not attestation["source_statement"].strip()
                    or not isinstance(attestation.get("confirmation_date"), str) or not attestation["confirmation_date"].strip()
                    or attestation.get("nodes_sha256") != hashlib.sha256(payload).hexdigest()):
                raise ValueError("用户背书未绑定当前知识节点，拒绝准入")
            self.attestation_verified = True
        self.entries: list[dict] = []
        ids: set[str] = set()
        for entry in graph["nodes"]:
            if not isinstance(entry.get("id"), str) or not entry["id"] or entry["id"] in ids or not isinstance(entry.get("source_url"), str) or not entry["source_url"].startswith("https://"):
                raise ValueError("知识条目缺少唯一ID或来源")
            ids.add(entry["id"])
            status = entry.get("review_status")
            if status not in {"candidate", "expert_reviewed", "rejected", "user_attested"}:
                raise ValueError("知识审核状态未知")
            if status == "expert_reviewed" and not (entry.get("reviewer") and entry.get("reviewed_at")):
                raise ValueError("已审核知识必须记录审核人和时间")
            if not isinstance(entry.get("aliases"), list) or not all(isinstance(x, str) and x.strip() for x in entry["aliases"]):
                raise ValueError("检索别名必须是非空字符串列表")
            relations=entry.get("relations",[])
            if not isinstance(relations,list) or not all(isinstance(r,dict) and all(isinstance(r.get(k),str) and r[k] for k in ("predicate","target")) for r in relations):
                raise ValueError("知识关系必须记录谓词与目标")
            if status == "expert_reviewed" or (status == "candidate" and allow_candidates) or (status == "user_attested" and allow_user_attested):
                self.entries.append(entry)
        self.review_mode = "offline_candidates" if allow_candidates else ("user_attested_only" if allow_user_attested else "expert_reviewed_only")
        self.semantic_vectors: dict[str, np.ndarray] | None = None
        self.hash_dimensions = dimensions

    def attach_semantic_vectors(self, vectors: dict[str, np.ndarray]) -> None:
        """绑定已准入节点的冻结语义向量，保留独立关系散列空间。"""
        ids={e["id"] for e in self.entries}
        if self.semantic_vectors is not None or set(vectors)!=ids or not ids:
            raise ValueError("语义向量必须逐项对应准入节点且仅绑定一次")
        dimensions={np.asarray(v).shape for v in vectors.values()}
        if len(dimensions)!=1:
            raise ValueError("节点语义向量形状不同")
        shape=next(iter(dimensions))
        if len(shape)!=1 or not 1<=shape[0]<=2048 or not all(np.isfinite(v).all() for v in vectors.values()):
            raise ValueError("节点语义向量必须有限且有界")
        self.semantic_vectors={k:np.asarray(v,dtype=np.float32) for k,v in vectors.items()}
        self.dimensions=self.hash_dimensions+shape[0]

    def retrieve(self, text: str, limit: int = 4) -> list[dict]:
        """按显式概念词匹配候选，返回来源及审核状态；未命中不猜测。"""
        normalized = text.casefold()
        matches: list[tuple[int, dict]] = []
        for entry in self.entries:
            score = 0
            for alias in entry["aliases"]:
                # 英文词边界避免 sad 命中 said 一类子串；中文采用原文子串。
                if re.fullmatch(r"[A-Za-z ]+", alias):
                    score += bool(re.search(r"\b" + re.escape(alias.casefold()) + r"\b", normalized))
                else:
                    score += alias.casefold() in normalized
            if score:
                matches.append((score, entry))
        matches.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [{"id": e["id"], "text": e["text"], "source_url": e["source_url"],
                 "review_status": e["review_status"], "score": score,
                 "relations": e.get("relations", [])} for score, e in matches[:limit]]

    def encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """将命中的节点和关系散列到固定维数，不拟合标签与测试统计。"""
        output = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        present = np.zeros(len(texts), dtype=bool)
        for i, text in enumerate(texts):
            for entry in self.retrieve(text):
                present[i] = True
                tokens = [entry["id"], *(r["predicate"] + ":" + r["target"] for r in entry["relations"])]
                for token in tokens:
                    h = hashlib.sha256(token.encode()).digest()
                    output[i, int.from_bytes(h[:4], "little") % self.hash_dimensions] += 1 if h[4] % 2 else -1
                if self.semantic_vectors is not None:
                    output[i,self.hash_dimensions:]+=self.semantic_vectors[entry["id"]]
            for part in ([slice(0,self.hash_dimensions),slice(self.hash_dimensions,None)] if self.semantic_vectors is not None else [slice(None)]):
                norm=np.linalg.norm(output[i,part])
                if norm:output[i,part]/=norm
            norm=np.linalg.norm(output[i])
            if norm:output[i]/=norm
        return output, present

