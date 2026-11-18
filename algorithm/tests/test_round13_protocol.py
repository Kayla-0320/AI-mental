"""新语料封存、可用性专头和冻结专家稀疏分派的关键验证。"""
from __future__ import annotations

import io
import pickle
import numpy as np
import pytest
import torch
from experiments.round13_data import ArrayUnpickler
from experiments.round13_models import Projection,modify,observed,fit_heads,predict_heads,configurations,SentimentLexicon
from experiments.round13_neural import FrozenBankRouter,bank_predict,dispatch,flat


def data() -> dict[str,np.ndarray]:
    """固定的小型三模态M3样本，包含真实不可用位置。"""
    rng = np.random.default_rng(42)
    return dict(T=rng.normal(size=(30,7)),A=rng.normal(size=(30,5)),V=rng.normal(size=(30,3)),
        labels=np.tile(np.arange(3),10),raw_text=np.array(["开心" for _ in range(30)]),
        sample_ids=np.array([str(i) for i in range(30)]),source_groups=np.array([str(i//3) for i in range(30)]))


def test_array_unpickler_rejects_callables() -> None:
    with pytest.raises(pickle.UnpicklingError):
        ArrayUnpickler(io.BytesIO(pickle.dumps(eval))).load()
    a = np.arange(6).reshape(2,3)
    assert np.array_equal(ArrayUnpickler(io.BytesIO(pickle.dumps(a))).load(),a)


def test_hidden_modalities_are_removed_before_projection() -> None:
    td = data()
    projection = Projection().fit(td)
    changed = modify(td,"missing_A")
    before = projection.transform(changed)
    changed["A"][:] = float("nan")
    after = projection.transform(changed)
    assert np.array_equal(before["A"],after["A"])
    assert not before["A"].any()
    assert not before["mask"][:,1].any()


def test_specialist_never_consumes_an_absent_modality_and_inference_has_no_labels() -> None:
    td = data()
    config = configurations()[0]
    projection = Projection().fit(td)
    bundle = fit_heads(projection.transform(td),td["labels"],config,np.zeros((30,12)))
    changed = modify(td,"missing_A")
    changed.pop("labels")
    p = predict_heads(bundle,projection.transform(changed),np.zeros((30,12)))
    changed["A"][:] = 1e20
    q = predict_heads(bundle,projection.transform(changed),np.zeros((30,12)))
    assert np.array_equal(p,q)
    absent = projection.transform(modify(td,"all_missing"))
    prior = predict_heads(bundle,absent,np.zeros((30,12)))
    assert np.allclose(prior,bundle["prior"])


def test_actual_top2_dispatch_agrees_with_dense_bank_math() -> None:
    td = data()
    projection = Projection().fit(td)
    x = projection.transform(td)
    bundle = fit_heads(x,td["labels"],configurations()[0],np.zeros((30,12)))
    model = FrozenBankRouter(flat(x).shape[1],dict(mode="top2"),bundle["full"],bundle["prior"]).eval()
    with torch.inference_mode():
        expected,gate = model(torch.from_numpy(flat(x)),torch.from_numpy(x["mask"]),torch.from_numpy(bank_predict(bundle,x)))
    actual,statistics = dispatch(model,bundle,x)
    assert np.allclose(actual,expected.exp().numpy(),atol=1e-7)
    assert statistics["executed_pairs"] <= 2*len(td["labels"])
    assert torch.all((gate>0).sum(1)<=2)


def test_ntusd_has_real_coverage_and_negation_changes_evidence() -> None:
    lexicon = SentimentLexicon()
    assert len(lexicon.words)>10000
    p = lexicon.encode(np.array(["非常开心", "并不开心", "但是非常开心"]))
    assert p[0,9]>0
    assert p[1,4]>0
    assert p[1,3]>p[0,3]
    assert p[2,10]==1


def test_projection_statistics_do_not_change_during_validation() -> None:
    td = data()
    projection = Projection().fit(td)
    means = {k:v[0].mean_.copy() for k,v in projection.members.items() if v is not None}
    vd = {k:v.copy() for k,v in td.items()}
    vd["T"] += 1000
    projection.transform(vd)
    assert all(np.array_equal(mean,projection.members[k][0].mean_) for k,mean in means.items())


def test_test_labels_are_opened_only_after_model_and_predictions_lock(tmp_path,monkeypatch) -> None:
    from experiments import round13_evaluate as evaluation
    import json
    monkeypatch.setattr(evaluation,"OUTPUT",tmp_path)
    choice = dict(winner=dict(family="specific",config=dict(id="new")),ordinary=dict(config=dict(id="base")))
    monkeypatch.setattr(evaluation,"choose",lambda dataset:choice)
    for name in ("fold0/projection.pkl","base/fold0/model.pkl","new/fold0/model.pkl"):
        path = tmp_path/"training/demo"/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b"test model identity")
    folder = tmp_path/"datasets/demo"
    folder.mkdir(parents=True)
    ids = np.array(["a","b","c","d","e","f"])
    np.savez_compressed(folder/"test_inputs.npz",sample_ids=ids,source_groups=np.array(["s1","s1","s2","s2","s3","s3"]))
    np.savez_compressed(folder/"sealed_test_labels.npz",sample_ids=ids,labels=np.tile(np.arange(3),2))
    p = np.tile(np.array([[.7,.2,.1]]),(6,1))
    predictions = {kind+"_"+c:p.copy() for kind in ("baseline","candidate") for c in evaluation.CONDITIONS}
    monkeypatch.setattr(evaluation,"predict",lambda dataset,chosen,inputs:(predictions,{}))
    monkeypatch.setattr(evaluation,"interval",lambda *args,**kwargs:dict(delta=0.,paired_group_ci95=[0.,0.],permutation_p_improvement=1.))
    original_read = evaluation.read
    accesses = []
    def guarded_read(path) -> dict[str,np.ndarray]:
        if path.name == "sealed_test_labels.npz":
            results = tmp_path/"new_test/demo"
            assert (results/"evaluation_lock.json").is_file()
            assert (results/"predictions_locked.json").is_file()
            assert (results/"predictions_no_labels.npz").is_file()
            accesses.append(path.name)
        return original_read(path)
    monkeypatch.setattr(evaluation,"read",guarded_read)
    evaluation.score("demo")
    evaluation.score("demo")
    assert accesses == ["sealed_test_labels.npz"]
    assert not json.loads((tmp_path/"new_test/demo/results.json").read_text("utf-8"))["selection_after_test"]
