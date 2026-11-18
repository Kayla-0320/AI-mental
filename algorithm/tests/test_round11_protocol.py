"""科学协议回归：来源隔离、缺失证据和标签不进入推理。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from experiments import round11_evidence as evidence
from experiments import round11_features as features
from round11_input_audit import presentation_span


def synthetic() -> dict[str,np.ndarray]:
    """构造带不同观测与来源的数值输入，包含真实缺失mask。"""
    rng=np.random.default_rng(42)
    n=90
    fields=set(features.TEXT+features.VISUAL+features.AUDIO_PACKS["both"])
    data={name:rng.normal(size=(n,6)) for name in fields}
    data["face_summary"]=rng.normal(size=(n,29))
    data["face_summary"][:,24]=1
    data.update(sample_ids=np.array([f"id{i}" for i in range(n)]),
        labels=np.arange(n)%3,source_groups=np.array([f"source{i//5}" for i in range(n)]),
        raw_text=np.array(["文字"]*n),modality_mask=np.ones((n,3),bool),quality=np.ones((n,3)),
        coverage=np.ones(n),speech_available=np.ones(n,bool),sensevoice_available=np.ones(n,bool),
        asr_alignment=np.ones((n,3)))
    return data


def test_pts_span_uses_timestamps_instead_of_frame_count() -> None:
    value=presentation_span([0.,.04,.12,.20],[.04]*4)
    assert value["duration"] == pytest.approx(.24)
    assert value["duration"] != pytest.approx(4/25)
    assert not presentation_span([0.,.1,.05],[.1]*3)["verified"]


def test_rejected_audio_cannot_become_observed_from_cache() -> None:
    data=synthetic()
    data["modality_mask"][0,1]=False
    data["quality"][1,1]=0
    data["sensevoice_available"][2]=False
    assert features.observed(data,"sensevoice")[:3].tolist() == [False,False,False]
    values,_=features.transform(data,("sensevoice",))
    assert np.array_equal(values[:3],np.zeros_like(values[:3]))


def test_validation_extremes_do_not_fit_statistics() -> None:
    train=synthetic()
    validation=synthetic()
    _,scalers=features.transform(train,("sensevoice","deep_face"))
    mean=scalers["sensevoice"]["scaler"].mean_.copy()
    projection=scalers["deep_face"]["pca"].components_.copy()
    validation["sensevoice"][:]=1e10
    validation["deep_face"][:]=-1e10
    transformed,_=features.transform(validation,("sensevoice","deep_face"),scalers)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(mean,scalers["sensevoice"]["scaler"].mean_)
    np.testing.assert_array_equal(projection,scalers["deep_face"]["pca"].components_)


def test_prediction_ignores_labels_sources_and_supervision(monkeypatch: pytest.MonkeyPatch) -> None:
    data=synthetic()
    classes=np.column_stack([np.roll(data["labels"],j) for j in range(4)])
    monkeypatch.setattr(evidence,"supervision",lambda _data:(classes,(classes-1).astype(float)))
    fitted=evidence.fit_experts(data,{"classes":3,"encoder":"sense"})
    expected=evidence.expert_evidence(fitted,data)
    def forbidden(_data: dict[str,np.ndarray]) -> None:
        raise AssertionError("预测阶段调用了标签读取器")
    monkeypatch.setattr(evidence,"supervision",forbidden)
    data["labels"][:]=-999
    data["source_groups"][:]="arbitrary"
    data["raw_text"][:]="unrelated"
    actual=evidence.expert_evidence(fitted,data)
    np.testing.assert_array_equal(actual,expected)
    data["modality_mask"][:,1]=False
    missing=evidence.expert_evidence(fitted,data)
    assert np.array_equal(missing[:,:12].reshape(-1,4,3)[:,2],np.zeros((len(actual),3)))


def test_asr_special_tags_are_not_alignment_text() -> None:
    exact=features.alignment_feature("今天很好。","<|zh|><|NEUTRAL|><|Speech|>今天很好")
    np.testing.assert_array_equal(exact,np.ones(3))
    assert features.alignment_feature("今天很好", "")[2] == 0


def test_top2_dispatch_executes_only_selected_M_experts() -> None:
    import torch
    from experiments.round11_neural import RoutedHead
    model=RoutedHead({name:4 for name in ("T","A","V")},"top2",hidden=8).eval()
    with torch.no_grad():
        for parameter in model.router.parameters():parameter.zero_()
        model.router[-1].bias.copy_(torch.tensor([4.,3.,2.,1.]))
    executed=[0,0,0,0]
    handles=[]
    for i,head in enumerate([*model.single,model.joint_classifier]):
        def capture(_module: object,args: tuple[torch.Tensor,...],_result: object,index: int=i) -> None:
            executed[index]+=len(args[0])
        handles.append(head.register_forward_hook(capture))
    x={name:torch.randn(7,4) for name in ("T","A","V")}
    with torch.inference_mode():
        p,gate,_,_=model(x,torch.ones(7,3,dtype=torch.bool),torch.ones(7,3),auxiliary=False)
    for handle in handles:handle.remove()
    assert executed==[7,7,0,0]
    assert (gate>0).sum(1).tolist()==[2]*7
    torch.testing.assert_close(p.exp().sum(1),torch.ones(7))


def test_soft_swap_teacher_detached_and_equal_forward_budget() -> None:
    import torch
    from experiments.round11_neural import RoutedHead,soft_swap_target
    torch.manual_seed(42)
    model=RoutedHead({name:4 for name in ("T","A","V")},"dynamic",hidden=8).train()
    x={name:torch.randn(6,4,requires_grad=True) for name in ("T","A","V")}
    mask=torch.ones(6,3,dtype=torch.bool)
    quality=torch.ones(6,3)
    donors=torch.tensor([1,2,3,4,5,0])
    calls=[]
    handle=model.register_forward_hook(lambda _m,_a,_r:calls.append(1))
    first=soft_swap_target(model,x,mask,quality,donors,"soft")
    soft_calls=len(calls)
    calls.clear()
    second=soft_swap_target(model,x,mask,quality,donors,"consistency")
    handle.remove()
    assert soft_calls==len(calls)==2
    assert model.training and not first.requires_grad and not second.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    torch.testing.assert_close(first.sum(1),torch.ones(6))
    # eval教师无dropout噪声；学生模式必须在调用后恢复。
    repeated=soft_swap_target(model,x,mask,quality,donors,"soft")
    torch.testing.assert_close(first,repeated,rtol=0,atol=0)


def test_swapped_quality_follows_audio_and_invalid_pairs_excluded() -> None:
    import torch
    from experiments.round11_neural import paired_swap,soft_swap_target,RoutedHead
    x={name:torch.arange(12).reshape(3,4).float() for name in ("T","A","V")}
    mask=torch.tensor([[True,True,True],[True,False,True],[True,True,True]])
    quality=torch.tensor([[1.,.2,.8],[1.,0.,.7],[1.,.9,.5]])
    donors=torch.tensor([2,0,1])
    changed,visible,moved,valid=paired_swap(x,mask,quality,donors)
    torch.testing.assert_close(changed["A"],x["A"][donors])
    torch.testing.assert_close(changed["T"],x["T"])
    torch.testing.assert_close(moved[:,1],quality[donors,1])
    assert valid.tolist()==[True,False,False]
    assert visible[:,1].tolist()==[True,True,False]
    model=RoutedHead({name:4 for name in x},"dynamic",hidden=8)
    with pytest.raises(ValueError,match="可见"):
        soft_swap_target(model,x,mask,quality,donors,"soft")


def test_sequence_uses_completed_history_and_never_target_observation() -> None:
    from experiments.round11_sequence_inputs import index_windows,materialize
    rows=[{"id":str(i),"start":2.*i,"end":2.*i+1,"conversation":"c","group":"g",
           "split":"train","speaker":"s","M_emotion7":i%7,"text":f"future{i}"} for i in range(12)]
    observations={str(i):np.array([i/20,1-i/20],np.float32) for i in range(12)}
    windows,_=index_windows(rows,5,1)
    first=materialize(windows[:1],observations,5)
    target=windows[0]["target_id"]
    observations[target][:]=999
    for row in rows:
        if int(row["id"])>=int(target):
            row["M_emotion7"]=100
            row["text"]="mutated future"
    again,_=index_windows(rows,5,1)
    second=materialize(again[:1],observations,5)
    np.testing.assert_array_equal(first["sequence"],second["sequence"])
    rows[4]["end"]=rows[5]["start"]+.5
    eligible,audit=index_windows(rows,5,1)
    assert target not in {row["target_id"] for row in eligible}
    assert any(row["target_id"]==target and row["reason"]=="overlap" for row in audit["excluded"])


def test_sequence_latency_eligibility_does_not_use_future_labels() -> None:
    from experiments.round11_sequence_inputs import index_windows
    rows=[{"id":str(i),"start":2.*i,"end":2.*i+1.8,"conversation":"c","group":"g",
           "split":"train","speaker":"s"} for i in range(12)]
    ideal,_=index_windows(rows,5,1,0.)
    qualified,audit=index_windows(rows,5,1,1.)
    assert len(ideal)==7 and not qualified
    assert all(row["reason"]=="insufficient_inference_lead" for row in audit["excluded"])


def test_external_targets_stay_sealed_without_verified_model_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments import round11_external_data as external
    def forbidden(_split: str) -> list[dict[str,object]]:
        raise AssertionError("封存检查前不应读取输入或目标")
    monkeypatch.setattr(external,"read_rows",forbidden)
    with pytest.raises(ValueError,match="封存"):
        external.targets(("test_inputs",),"M3")


def test_external_target_lock_checks_actual_model_hash(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from experiments import round11_external_data as external
    model=tmp_path/"model.pkl"
    model.write_bytes(b"changed model")
    lock=tmp_path/"evaluation_lock.json"
    lock.write_text(json.dumps({"all_selection_complete":True,"tasks":["M3"],
        "model_files":[{"path":str(model),"sha256":"wrong digest"}]}),"utf-8")
    with pytest.raises(ValueError,match="摘要"):
        external.targets(("test_inputs",),"M3",lock)


def test_forecast_m7_lock_cannot_open_m3(tmp_path: Path) -> None:
    import json
    from experiments.round11_external_data import targets
    lock=tmp_path/"lock.json"
    lock.write_text(json.dumps({"tasks":["M_emotion7"],"all_selection_complete":True}),"utf-8")
    with pytest.raises(ValueError,match="当前任务"):
        targets(("test_inputs",),"M3",lock)


def test_gru_padding_and_future_metadata_do_not_affect_prediction() -> None:
    import torch
    from experiments.round11_forecast_finish import ForecastGRU,prepare,infer
    torch.manual_seed(42)
    sequence=np.zeros((2,10,10),np.float32)
    sequence[:,5:,:7]=np.array([.1,.1,.1,.2,.2,.2,.1],np.float32)
    sequence[:,5:,7]=1;sequence[:,5:,8:]=.5
    x,lengths,scaling=prepare(sequence,np.array([True,False]))
    model=ForecastGRU().eval();p=infer(model,x,lengths)
    x[:,5:]=999
    np.testing.assert_array_equal(p,infer(model,x,lengths))
    changed=sequence.copy();changed[1,:,8:]=999
    _,_,again=prepare(changed,np.array([True,False]))
    np.testing.assert_array_equal(scaling["mean"],again["mean"])


def test_drift_features_ignore_every_future_observation() -> None:
    from experiments.round11_forecast import summary
    window={"history_ids":[str(i) for i in range(5)],"prefix_ids":[str(i) for i in range(5)]}
    observed={str(i):np.full(7,i/100) for i in range(9)}
    expected=summary([window],observed,True)
    for i in range(5,9):observed[str(i)][:]=1e6
    np.testing.assert_array_equal(expected,summary([window],observed,True))


def test_missing_text_route_keeps_clean_and_uses_only_available_evidence() -> None:
    from experiments.round11_missing_route import route
    p=np.array([[.2,.3,.5],[.2,.3,.5],[.2,.3,.5]])
    evidence=np.zeros((3,44));evidence[:,6:9]=[.7,.2,.1];evidence[2,6:9]=0
    evidence[2,9:12]=[.1,.3,.6]
    actual=route(p,evidence,np.array([False,True,True]),0.,[0.,0.,0.],np.array([.2,.3,.5]))
    np.testing.assert_array_equal(actual[0],p[0])
    np.testing.assert_allclose(actual[1],[.7,.2,.1])
    np.testing.assert_allclose(actual[2],[.1,.3,.6])
