"""第十二轮离线融合头重放接口，推理参数不接受任何情绪金标。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch
from experiments.round12_model import EvidenceHead, evidence_inputs


class OfflinePredictor:
    """加载锁定来源留出头；基础编码器和原专家须按该折的训练边界配套。"""

    def __init__(self, path: Path) -> None:
        bundle: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        self.config = bundle["config"]
        self.model = EvidenceHead(self.config, bundle["knowledge_dim"], bundle["prior"])
        self.model.load_state_dict(bundle["state_dict"], strict=True)
        self.model.eval()

    def predict(self, evidence: np.ndarray, quality: np.ndarray, alignment: np.ndarray,
                knowledge: np.ndarray, memory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """只消费观测证据；知识和监督记忆需遵循对应训练来源索引。"""
        x, mask = evidence_inputs(evidence, quality)
        return self.predict_prepared(x, mask, alignment, knowledge, memory)

    def predict_prepared(self, x: np.ndarray, mask: np.ndarray, alignment: np.ndarray,
                         knowledge: np.ndarray, memory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """重放已构造的缺失/错配观测，不重新拟合或读取标签。"""
        values = (x.astype(np.float32), mask.astype(bool), alignment.astype(np.float32),
                  knowledge.astype(np.float32), memory.astype(np.float32))
        if any(len(v) != len(x) for v in values):
            raise ValueError("推理观测行数不一致")
        # 缺失位置可含无效缓存，进入模型前应被严格隔离。
        if not np.isfinite(x[mask]).all():
            raise ValueError("可见模态包含非有限观测")
        with torch.inference_mode():
            probability, gate = self.model(*(torch.from_numpy(v) for v in values))
        return probability.exp().numpy(), gate.numpy()
