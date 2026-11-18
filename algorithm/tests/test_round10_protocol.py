"""来源隔离、真实缺失、严格固定轮次和反事实配对回归。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments import round9_counterfactual as cf
from experiments import round9_moe_baselines as baseline
from experiments.round10_fusion import features, folds, take
from experiments.round10_neural import FusionHead
from experiments.round10_sequence import windows
from experiments.round10_represented_fusion import transform
from experiments.round10_final import predict
from experiments.round10_fusion import make_model
from experiments import round10_evidence as evidence
from sklearn.dummy import DummyClassifier, DummyRegressor


def tiny_data() -> dict[str, dict[str, np.ndarray]]:
    """三个互不重叠的小型划分，仅用于训练协议测试。"""
    rng = np.random.default_rng(42)
    parts = {}
    for split in ("train", "selection", "test"):
        n = 24
        parts[split] = {"text": rng.normal(size=(n, 4)).astype(np.float32),
                        "audio": rng.normal(size=(n, 2)).astype(np.float32),
                        "visual": rng.normal(size=(n, 4)).astype(np.float32),
                        "modality_mask": np.ones((n, 3), bool), "quality": np.ones((n, 3), np.float32),
                        "labels": np.tile(np.arange(3), 8), "raw_text": np.array(["样本"] * n),
                        "source_groups": np.array([f"{split}_{i // 3}" for i in range(n)])}
    return parts


def test_fixed_epoch_returns_last_even_when_validation_prefers_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """固定轮次不得偷偷回载validation最优checkpoint。"""
    original = torch.nn.functional.cross_entropy
    evaluation_calls = []

    def controlled(*args, **kwargs):
        if not torch.is_grad_enabled():
            evaluation_calls.append(1)
            return torch.tensor(float(len(evaluation_calls)))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", controlled)
    _, info, _ = baseline.train_once("concat", 42, 8, tiny_data(), 3,
                                    epochs=3, patience=1, use_last_checkpoint=True)
    assert info["epoch"] == 3
    assert len(evaluation_calls) == 3


def test_counterfactual_without_valid_donor_is_excluded() -> None:
    """-1捐赠者不能变成最后一条训练标签。"""
    data = tiny_data()
    baseline.seed_all()
    model, _, batches = baseline.train_once("moe", 42, 8, data, 3, epochs=1)
    result, _, _, valid = cf.forward_with_swap(model, batches["train"], torch.arange(4),
        torch.full((4,), -1), torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.bool))
    assert not valid.any()
    assert len(result) == 0


def test_counterfactual_zero_weight_matches_training_recipe() -> None:
    """本地无CF训练和基线必须包含相同增广，逐参数一致。"""
    data = tiny_data()
    baseline.seed_all()
    first, _, _ = baseline.train_once("moe", 42, 8, data, 3, epochs=2)
    baseline.seed_all()
    second, _, _ = cf.train_once("moe", False, 42, 8, data, 3, epochs=2)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key


def test_laplace_large_finite_logits_remain_normalized() -> None:
    """指数下溢不能使门控权重和变成零。"""
    model = baseline.FuseMoE({"text": 4}, 3, 8)
    with torch.no_grad():
        model.router[-1].weight.zero_()
        model.router[-1].bias.fill_(1000.)
    _, weights, _ = model({"text": torch.zeros(2, 4)}, torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 1))
    assert torch.allclose(weights.sum(1), torch.ones(2))


def test_scaler_never_fits_missing_face_or_validation_values() -> None:
    """验证观测不改变均值；无脸记录不进入表征标准化。"""
    data = tiny_data()["train"]
    data["roberta_mean"] = data["text"]
    data["bge_cls"] = data["text"]
    data["face_summary"] = np.ones((24, 29), np.float32)
    data["face_summary"][0] = 0.
    config = {"pack": "textface", "face_weight": 1.}
    _, scalers = features(take(data, np.arange(12)), config)
    assert np.allclose(scalers["face_summary"].mean_, 1.)
    before = scalers["roberta_mean"].mean_.copy()
    data["roberta_mean"][12:] += 1000
    features(take(data, np.arange(12, 24)), config, scalers)
    assert np.array_equal(before, scalers["roberta_mean"].mean_)
    for train, valid in folds(data):
        assert not set(data["source_groups"][train]) & set(data["source_groups"][valid])


@pytest.mark.parametrize("mode", ["uniform", "global", "dynamic", "top2"])
def test_expert_voting_never_uses_missing_face(mode: str) -> None:
    """缺失人脸及其联合专家不可被投票或Top-k选中。"""
    model = FusionHead({"semantic": 4, "face": 3}, {"mode": mode}).eval()
    x = {"semantic": torch.randn(4, 4), "face": torch.full((4, 3), float("nan"))}
    logits, weights = model(x, torch.tensor([[True, False]] * 4))
    assert torch.isfinite(logits).all()
    assert torch.allclose(weights[:, 0], torch.ones(4))
    assert torch.count_nonzero(weights[:, 1:]) == 0


def test_forecast_future_annotation_cannot_change_past_features() -> None:
    """修改目标或未来标签不能改变该预测窗口的输入。"""
    rows = [{"id": f"G00001_01_01_{i:03d}", "turn": i, "group": "G00001", "labels": [i % 7]*3}
            for i in range(12)]
    before = windows({"speaker": rows}, 10, 1)
    rows[5]["labels"] = [6, 6, 6]
    after = windows({"speaker": rows}, 10, 1)
    assert np.array_equal(before["x"][0], after["x"][0])
    assert before["labels"][0] != after["labels"][0]
    assert before["cutoff_turn"][0] < 5


def test_final_prediction_never_reads_true_labels() -> None:
    """最终推理支持无标签输入，修改金标不改变结果。"""
    data = tiny_data()["train"]
    data["sample_ids"] = np.array([f"sample_{i}" for i in range(24)])
    data["aux_mean"] = data["text"]
    data["bge_cls"] = data["text"]
    data["face_summary"] = np.ones((24, 29), np.float32)
    config = {"pack": "aux_textface", "face_weight": 1., "kind": "lr", "C": .1, "class_weight": "balanced"}
    x, scalers = transform(data, config)
    model = make_model(config).fit(x, data["labels"])
    bundle = {"bias": [0., -.25, 0.], "components": [{"weight": 1., "config": config,
              "scalers": scalers, "model": model, "logit_scale": 1.}]}
    before = predict(bundle, data)
    data.pop("labels")
    assert np.array_equal(before, predict(bundle, data))


def test_unimodal_supervision_preserves_distinct_labels(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """M为中性不代表T/A/V都中性，别名对齐不能偷偷改M金标。"""
    (tmp_path / "official_meta.csv").write_text(
        "video_id,clip_id,label,label_T,label_A,label_V,annotation\n"
        "csbgs2_0003,00000,0.0,-0.8,-0.2,0.6,Neutral\n", encoding="utf-8")
    monkeypatch.setattr(evidence, "OUTPUT", tmp_path)
    data = {"sample_ids": np.array(["csbdgs2_0003_00000"]), "labels": np.array([1])}
    classes, values = evidence.supervision(data)
    assert classes.tolist() == [[1, 0, 0, 2]]
    assert values[0, 0] == 0.
    data["labels"] = np.array([2])
    with pytest.raises(ValueError, match="冻结任务不一致"):
        evidence.supervision(data)


@pytest.mark.parametrize("classes", [3, 5])
def test_evidence_never_reads_labels_and_marks_missing_expert(classes: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """推理禁止调用监督标注；缺失声音的证据为零且可用标志为假。"""
    n = 10
    x = np.zeros((n, 2))
    classifier = DummyClassifier(strategy="prior").fit(x, np.arange(n) % classes)
    regression = DummyRegressor().fit(x, np.linspace(-1, 1, n))
    models = {v: {"config": {}, "scaler": None, "model": classifier, "regression": regression} for v in evidence.VIEWS}
    monkeypatch.setattr(evidence.represented, "transform", lambda data, config, scaler: (x, None))

    def forbidden(_):
        raise AssertionError("推理不允许读取监督元数据")

    monkeypatch.setattr(evidence, "supervision", forbidden)
    data = {"sample_ids": np.array([str(i) for i in range(n)]), "labels": np.arange(n),
            "modality_mask": np.ones((n, 3), bool), "quality": np.ones((n, 3)),
            "face_summary": np.ones((n, 29))}
    data["modality_mask"][:, 1] = False
    before = evidence.expert_evidence(models, data)
    probabilities = before[:, :4*classes].reshape(n, 4, classes)
    assert np.allclose(probabilities[:, 0].sum(1), 1.)
    assert np.allclose(probabilities[:, 2].sum(1), 0.)
    data["labels"] = np.full(n, -999)
    assert np.array_equal(before, evidence.expert_evidence(models, data))


def test_meta_crossfit_experts_never_see_their_scored_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """元训练证据必须由未见该来源的专家生成，不能用普通堆叠假冒嵌套OOF。"""
    data = tiny_data()["train"]
    data["sample_ids"] = np.array([f"sample_{i}" for i in range(24)])
    calls = []

    def fitting(part):
        return set(part["source_groups"])

    def scoring(seen, part):
        assert not seen & set(part["source_groups"])
        calls.extend(part["sample_ids"].tolist())
        return np.ones((len(part["sample_ids"]), 4))

    monkeypatch.setattr(evidence, "fit_experts", fitting)
    monkeypatch.setattr(evidence, "expert_evidence", scoring)
    result = evidence.crossfit(data)
    assert result.shape == (24, 4)
    assert len(calls) == len(set(calls)) == 24


def test_completed_experiment_is_never_overwritten(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """重放旧命令必须先拒绝覆盖，不能改写已审核数字。"""
    monkeypatch.setattr(baseline, "OUTPUT", tmp_path)
    (tmp_path / f"moe_baselines{baseline.TAG}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        baseline.main()


def test_speech_final_routes_only_uncertain_observations_without_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """最终路由跳过高置信行和零权重专家，并在无金标时保留五种子输出。"""
    from experiments import round10_speech_final as final
    monkeypatch.setattr(final.speech, "configure", lambda: None)
    cheap = np.array([[.95, .03, .02], [.35, .30, .35]])
    classifier = DummyClassifier(strategy="prior").fit(np.zeros((3, 1)), [0, 1, 2])

    def cheap_only(member, data):
        assert member["stage"] == "cheap"
        assert np.all(data["labels"] == 0)
        return cheap.copy()

    def uncertain_only(models, data):
        assert data["sample_ids"].tolist() == ["b"]
        assert np.all(data["labels"] == 0)
        return np.zeros((1, 1))

    monkeypatch.setattr(final, "member_predict", cheap_only)
    monkeypatch.setattr(final.evidence, "expert_evidence", uncertain_only)
    bundle = {"selection": {"winner": {"weights": [.75, 0., .25], "route": .8, "bias": [0., 0., 0.]}},
              "other_members": [{"stage": "unused"}, {"stage": "cheap"}],
              "experts": {}, "meta_models": [classifier]*5}
    data = {"sample_ids": np.array(["a", "b"])}
    prediction, needed = final.predict(bundle, data)
    assert prediction.shape == (5, 2, 3)
    assert needed.tolist() == [False, True]
    assert np.allclose(prediction[:, 0], cheap[0])
    assert np.allclose(prediction[:, 1], .75/3+.25*cheap[1])
    data["labels"] = np.array([2, 1])
    assert np.array_equal(prediction, final.predict(bundle, data)[0])


def test_output_adaptation_never_sees_its_scored_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """703保留集中的适配选型也必须来源隔离，不能在拟合行上报分。"""
    from experiments import round10_holdout as holdout
    data = tiny_data()["train"]
    indices = np.arange(len(data["labels"]))
    features = np.repeat(np.column_stack([indices, indices, indices])[None].astype(float), 5, axis=0)
    observed = []

    class SourceCheckingAdapter:
        def fit(self, x, y, sample_weight=None):
            self.seen = set(data["source_groups"][x[:, 0].astype(int)])
            return self

        def predict_proba(self, x):
            scored = x[:, 0].astype(int)
            assert not self.seen & set(data["source_groups"][scored])
            observed.extend(scored.tolist())
            return np.full((len(x), 3), 1/3)

    monkeypatch.setattr(holdout, "calibrator", lambda config, n: SourceCheckingAdapter())
    probability = holdout.adaptation_oof(features, data, {"C": 1., "class_power": .5})
    assert probability.shape == (5, 24, 3)
    assert np.allclose(probability.sum(-1), 1.)
    assert len(observed) == 5*24


def test_raw_foundation_evidence_excludes_labels_and_masks_unobserved_nan() -> None:
    """原始输出证据不能带入标签，未观测的NaN也不能污染模型。"""
    from experiments.round10_direct_evidence import raw_evidence
    data = tiny_data()["train"]
    data["llm_scores"] = np.zeros((24, 3))
    data["speech_emotion"] = np.ones((24, 3))
    data["speech_available"] = np.ones(24, bool)
    data["face_summary"] = np.ones((24, 29))
    data["speech_available"][0] = False
    data["speech_emotion"][0] = np.nan
    data["face_summary"][0, :24] = np.nan
    data["face_summary"][0, 24] = 0.
    before = raw_evidence(data)
    assert np.isfinite(before).all()
    data.pop("labels")
    assert np.array_equal(before, raw_evidence(data))
    data["speech_available"][0] = True
    with pytest.raises(ValueError, match="非有限"):
        raw_evidence(data)


def test_soft_counterfactual_target_detaches_teachers_and_route() -> None:
    """软目标不能倒流更新教师或靠调整目标权重规避损失。"""
    source = torch.tensor([[5., 0., -5.]]*3, requires_grad=True)
    donor = torch.tensor([[-5., 0., 5.]]*3, requires_grad=True)
    route = torch.tensor([0., .5, 1.], requires_grad=True)
    target = cf.soft_counterfactual_target(source, donor, route)
    assert not target.requires_grad
    assert torch.allclose(target[0], source[0].softmax(0))
    assert torch.allclose(target[2], donor[2].softmax(0))
    assert torch.allclose(target[1], .5*(target[0]+target[2]))
    logits = torch.zeros_like(source, requires_grad=True)
    torch.nn.functional.kl_div(logits.log_softmax(1), target, reduction="batchmean").backward()
    assert logits.grad.abs().sum() > 0
    assert source.grad is None and donor.grad is None and route.grad is None


def test_counterfactual_quality_follows_donor_and_teacher_only_sees_swapped_modality() -> None:
    """低质量捐赠不能继承源的高质量，单模态教师不能读取其他模态。"""
    class RecordingModel(torch.nn.Module):
        classes = 3

        def forward(self, values, mask, quality):
            self.observed = mask.clone()
            self.quality = quality.clone()
            return torch.zeros(len(mask), 3), torch.ones(len(mask), 1), torch.tensor(0.)

    model = RecordingModel()
    x = {name: torch.ones(2, 4) for name in cf.MODALITIES}
    mask = torch.ones(2, 3, dtype=torch.bool)
    quality = torch.tensor([[.9, .8, .7], [.5, .2, .3]])
    batch = (x, mask, quality, None, None, torch.tensor([0, 2]))
    _, _, _, valid = cf.forward_with_swap(model, batch, torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), torch.tensor([True]))
    assert valid.all()
    assert torch.allclose(model.quality, torch.tensor([[.9, .2, .7]]))
    cf.unimodal_teacher(model, batch, torch.tensor([1]), torch.tensor([1]))
    assert model.observed.tolist() == [[False, True, False]]
    assert torch.allclose(model.quality, torch.tensor([[0., .2, 0.]]))


def test_soft_counterfactual_training_produces_finite_updates() -> None:
    """新的软目标正则实际进入反向训练，合法配对与参数均有限。"""
    model, info, _ = cf.train_once("moe", True, 42, 8, tiny_data(), 3, epochs=2)
    assert info["cf_pairs"] > 0
    assert all(torch.isfinite(value).all() for value in model.parameters())


def test_rebuild_child_inherits_startup_environment_not_project_import_side_effects(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """项目导入把线程改成1时，重训子进程仍须使用调用方原环境。"""
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[2] / "scripts" / "round10_reproduce_selected.py"
    spec = importlib.util.spec_from_file_location("round10_reproduction_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "STARTUP_ENVIRONMENT", {"OMP_NUM_THREADS": "20", "EXAMPLE": "preserved"})
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    actual = module.child_environment(tmp_path)
    assert actual["OMP_NUM_THREADS"] == "20"
    assert actual["EXAMPLE"] == "preserved"
    assert actual["ROUND10_OUTPUT"] == str(tmp_path)
    assert "ROUND10_OUTPUT" not in module.STARTUP_ENVIRONMENT


def test_top2_dispatch_does_not_execute_unselected_expert() -> None:
    """Top-2必须真正省掉未选专家的前向，而非全部计算后乘零。"""
    model = FusionHead({"semantic": 4, "face": 3}, {"mode": "top2", "hidden": 8}).eval()
    with torch.no_grad():
        model.router[-1].weight.zero_()
        model.router[-1].bias.copy_(torch.tensor([3., -1., 2.]))
    calls = []
    handles = [expert.register_forward_pre_hook(lambda module, inputs, index=i: calls.append((index, len(inputs[0]))))
               for i, expert in enumerate(model.experts)]
    try:
        _, weights = model({"semantic": torch.randn(5, 4), "face": torch.randn(5, 3)}, torch.ones(5, 2, dtype=torch.bool))
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [(0, 5), (2, 5)]
    assert torch.equal((weights > 0).sum(1), torch.full((5,), 2))
