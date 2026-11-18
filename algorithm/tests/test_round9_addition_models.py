"""I²MoE / FuseMoE 增补架构的契约测试。

覆盖四件事，缺一不可：
1. 形状与门控归一化（权重逐行和为 1，且不出现 NaN）；
2. 缺失模态必须被掩掉（缺一路时该路专家权重为 0），全缺失必须拒绝预测；
3. 相同种子下可复现（保证与九臂同协议）；
4. 梯度能回流到门控（否则"学习型路由"是空话）。

不测试识别准确率——准确率只在 round9_moe_baselines 的同协议实验里报。
"""
from __future__ import annotations

import pytest
import torch

from experiments.round9_moe_baselines import FuseMoE, I2MoE

DIMENSIONS = {"text": 16, "audio": 8, "visual": 16}


def _batch(n: int = 6) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    features = {name: torch.randn(n, dim, generator=generator) for name, dim in DIMENSIONS.items()}
    mask = torch.ones(n, len(DIMENSIONS), dtype=torch.bool)
    quality = torch.ones(n, len(DIMENSIONS))
    return features, mask, quality


@pytest.mark.parametrize("factory", [I2MoE, FuseMoE])
def test_shapes_and_gate_normalisation(factory) -> None:
    features, mask, quality = _batch()
    model = factory(DIMENSIONS, classes=3, hidden=8)
    logits, gates, balance = model(features, mask, quality)
    assert logits.shape == (6, 3)
    assert torch.isfinite(logits).all()
    assert gates.shape[0] == 6
    assert torch.allclose(gates.sum(dim=1), torch.ones(6), atol=1e-5)
    assert torch.isfinite(balance)
    assert float(balance.detach()) >= 0.0


@pytest.mark.parametrize("factory", [I2MoE, FuseMoE])
def test_missing_modality_is_masked(factory) -> None:
    """缺 text 时，依赖 text 的专家必须权重为 0（不能拿默认值冒充观测）。"""
    features, mask, quality = _batch()
    model = factory(DIMENSIONS, classes=3, hidden=8)
    mask[:, 0] = False
    quality[:, 0] = 0.0
    logits, gates, _ = model(features, mask, quality)
    assert torch.isfinite(logits).all()
    # I2MoE 的专家顺序：text, audio, visual, text+audio, text+visual, audio+visual
    # FuseMoE 的专家顺序：text, audio, visual, shared
    if factory is I2MoE:
        assert float(gates[:, 0].abs().max()) == 0.0  # 单 text 专家
        assert float(gates[:, 3].abs().max()) == 0.0  # text+audio
        assert float(gates[:, 4].abs().max()) == 0.0  # text+visual
    else:
        assert float(gates[:, 0].abs().max()) == 0.0  # text 专家


@pytest.mark.parametrize("factory", [I2MoE, FuseMoE])
def test_all_missing_is_rejected(factory) -> None:
    features, mask, quality = _batch()
    mask[:] = False
    quality[:] = 0.0
    model = factory(DIMENSIONS, classes=3, hidden=8)
    with pytest.raises(ValueError):
        model(features, mask, quality)


@pytest.mark.parametrize("factory", [I2MoE, FuseMoE])
def test_reproducible_under_same_seed(factory) -> None:
    def run() -> torch.Tensor:
        torch.manual_seed(42)
        model = factory(DIMENSIONS, classes=3, hidden=8)
        features, mask, quality = _batch()
        with torch.no_grad():
            return model(features, mask, quality)[0]

    assert torch.allclose(run(), run())


@pytest.mark.parametrize("factory", [I2MoE, FuseMoE])
def test_gradient_reaches_router(factory) -> None:
    """路由必须是可学习的；若梯度为零，说明训练时门控根本没被优化。"""
    model = factory(DIMENSIONS, classes=3, hidden=8)
    if factory is I2MoE:
        router_out = model.router[2]
    else:
        router_out = model.router[2]
    features, mask, quality = _batch()
    logits, _, _ = model(features, mask, quality)
    logits.sum().backward()
    assert router_out.weight.grad is not None
    assert float(router_out.weight.grad.abs().sum()) > 0.0
