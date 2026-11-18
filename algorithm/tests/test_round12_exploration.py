"""第十二轮验证：缺失隔离、供体来源、检索泄漏和实际稀疏分派。"""
from __future__ import annotations

import numpy as np
import torch
from experiments.round12_model import EvidenceHead, change_inputs, cross_source_memory, evidence_inputs
from experiments.round12_explore import configurations, donors_for
from experiments.round12_audio_fallback import audio_absent_route


def inputs() -> tuple[torch.Tensor, ...]:
    """固定观测与不同可用性，用于检查模型实际计算。"""
    torch.manual_seed(42)
    x = torch.randn(5, 3, 7)
    mask = torch.tensor([[True, True, True], [True, False, False], [False, True, False],
                         [False, False, True], [False, False, False]])
    return x, mask, torch.rand(5, 3), torch.rand(5, 32), torch.rand(5, 6)


def test_missing_values_cannot_affect_prediction_and_all_absent_uses_training_prior() -> None:
    config = next(c for c in configurations() if c["id"] == "moe_knowledge_memory")
    # 每模态为3概率+熵+期望+回归+质量，共7维。
    model = EvidenceHead(config, 32, np.array([.2, .3, .5])).eval()
    x, mask, alignment, knowledge, memory = inputs()
    with torch.inference_mode():
        p, gate = model(x, mask, alignment, knowledge, memory)
        changed = x.clone()
        changed[~mask] = float("nan")
        q, _ = model(changed, mask, alignment, knowledge, memory)
    assert torch.equal(p, q)
    assert torch.allclose(p[-1].exp(), torch.tensor([.2, .3, .5]))
    assert torch.count_nonzero(gate[-1]) == 0
    assert torch.isfinite(p).all()


def test_top2_executes_only_selected_sample_expert_pairs() -> None:
    config = next(c for c in configurations() if c["id"] == "top2_aug")
    model = EvidenceHead(config, 32, np.array([.2, .3, .5])).eval()
    x, mask, alignment, knowledge, memory = inputs()
    calls: dict[int, int] = {}
    hooks = []
    for j, head in enumerate(model.experts):
        def record(module: torch.nn.Module, args: tuple[torch.Tensor, ...], index: int = j) -> None:
            calls[index] = len(args[0])
        hooks.append(head.register_forward_pre_hook(record))
    with torch.inference_mode():
        p, gate = model(x, mask, alignment, knowledge, memory)
    for hook in hooks:
        hook.remove()
    # 全缺失行暂时执行joint，输出再覆盖为先验；其他行最多执行两头。
    for j in range(4):
        expected = int((gate[:, j] > 0).sum()) + int(j == 3)
        assert calls.get(j, 0) == expected
    assert torch.all((gate > 0).sum(1) <= 2)
    assert torch.allclose(p.exp().sum(1), torch.ones(5))
    assert gate[1, 1] == 0 and gate[1, 2] == 0


def test_memory_excludes_same_source_and_normalized_duplicate_text() -> None:
    bank = np.array([[1., 0], [.99, .01], [.8, .2], [0., 1.]])
    p, neighbors = cross_source_memory(bank[:1], bank, np.array([0, 0, 2, 1]),
        np.array(["s1"]), np.array(["s1", "s2", "s3", "s4"]),
        np.array(["same"]), np.array(["same", "same", "different", "other"]), k=2)
    assert set(neighbors[0]) == {2, 3}
    assert p[0, 0] == 0 and p[0, -1] == 1
    empty, ids = cross_source_memory(bank[:1], bank[:1], np.array([0]), np.array(["s1"]),
        np.array(["s1"]), np.array(["same"]), np.array(["same"]), k=2)
    assert not empty.any() and np.all(ids == -1)


def test_missing_text_removes_knowledge_memory_and_alignment() -> None:
    x, mask, alignment, knowledge, memory = (v.numpy() for v in inputs())
    changed = change_inputs(x, mask, alignment, knowledge, memory, "missing_T")
    assert not changed[1][:, 0].any()
    assert not changed[0][:, 0].any()
    assert not any(v.any() for v in changed[2:])
    assert np.array_equal(x[:, 1:], changed[0][:, 1:])


def test_audio_swap_moves_quality_and_uses_recipient_donor_asr_alignment() -> None:
    x, mask, alignment, knowledge, memory = (v.numpy() for v in inputs())
    donors = np.array([2, 0, 1, 2, 0])
    mismatch_alignment = np.full((5, 3), .123, np.float32)
    changed = change_inputs(x, mask, alignment, knowledge, memory, "audio_mismatch", donors, mismatch_alignment)
    assert np.array_equal(changed[0][:, 1], x[donors, 1])
    assert np.array_equal(changed[1][:, 1], mask[donors, 1])
    assert np.allclose(changed[2][0], .123)
    assert np.array_equal(changed[0][:, 0], x[:, 0])


def test_donors_never_cross_oof_partition_or_share_source() -> None:
    groups = np.array(["a", "b", "c", "d", "e", "f"])
    partition = np.array([0, 0, 1, 1, 2, 2])
    donor = donors_for(groups, partition)
    assert np.all(groups != groups[donor])
    assert np.array_equal(partition, partition[donor])


def test_evidence_offsets_match_original_expert_layout() -> None:
    evidence = np.arange(80).reshape(2, 40).astype(float)
    evidence[:, 24:28] = 1
    x, mask = evidence_inputs(evidence, np.ones((2, 3)))
    assert x.shape == (2, 3, 7)
    assert mask.all()
    assert np.array_equal(x[:, 0, :3], evidence[:, 3:6])
    assert np.array_equal(x[:, 0, 3], evidence[:, 13])
    assert np.array_equal(x[:, 0, 4], evidence[:, 17])
    assert np.array_equal(x[:, 0, 5], evidence[:, 21])


def test_audio_fallback_preserves_all_audio_present_and_no_text_decisions() -> None:
    old = np.array([[.1, .2, .7], [.8, .1, .1], [.3, .4, .3], [.3, .3, .4]])
    new = np.full((4, 3), 1/3)
    mask = np.array([[True, True, True], [True, False, True], [False, False, True], [True, True, True]])
    quality = np.ones((4, 3))
    quality[3, 1] = 0
    result = audio_absent_route(old, new, mask, quality)
    assert np.array_equal(result[[0, 2]], old[[0, 2]])
    assert np.array_equal(result[[1, 3]], new[[1, 3]])
    assert np.array_equal(old[0], [.1, .2, .7])


def test_runtime_predictor_matches_saved_head_without_any_label_argument(tmp_path) -> None:
    from experiments.round12_runtime import OfflinePredictor
    config = next(c for c in configurations() if c["id"] == "mlp_clean")
    prior = np.array([.2, .3, .5])
    model = EvidenceHead(config, 32, prior).eval()
    path = tmp_path / "head.pt"
    torch.save(dict(config=config, knowledge_dim=32, prior=prior, state_dict=model.state_dict()), path)
    values = inputs()
    with torch.inference_mode():
        expected, _ = model(*values)
    actual, _ = OfflinePredictor(path).predict_prepared(*(v.numpy() for v in values))
    assert np.array_equal(expected.exp().numpy(), actual)
