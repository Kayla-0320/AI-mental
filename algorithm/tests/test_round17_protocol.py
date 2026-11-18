"""双指标锁型、真实像素质量、专家资格与未来信息边界回归。"""
from __future__ import annotations
import json
import numpy as np
import pytest
from experiments.round17_forecast import OUT,balanced_forecast,candidates
from experiments.round17_pixels import quality_from_frames
from experiments.round17_quality import availability,text_usable,route
from experiments.round17_switch import guarded_mask
from experiments.round17_runtime import predict_day


def fixture() -> dict:
    """构建临时数值夹具，不作为真实用户记录。"""
    return {"dataset":"westhoff","source_schema":"openesm:westhoff:affect_valence:0_100","target_day":3,
        "observations":[{"person":"fixture","day":i,"value":50.} for i in range(3)]}


@pytest.mark.parametrize("change",["future","duplicate","unsorted","schema","nan","person"])
def test_invalid_daily_history_rejected(change: str) -> None:
    p=fixture()
    if change=="future":p["observations"][-1]["day"]=3
    if change=="duplicate":p["observations"][-1]["day"]=1
    if change=="unsorted":p["observations"].reverse()
    if change=="schema":p["source_schema"]="project_mood"
    if change=="nan":p["observations"][-1]["value"]=float("nan")
    if change=="person":p["observations"][-1]["person"]="other"
    with pytest.raises(ValueError):predict_day(p)


def test_gap_reset_abstains_without_imputation() -> None:
    p=fixture();p["target_day"]=10;p["observations"]=[{"person":"p","day":i,"value":50.} for i in (0,1,2,8,9)]
    assert predict_day(p)["status"]=="abstain"


def test_forecasting_candidates_preserve_constant_and_bounds() -> None:
    for config in candidates():
        assert balanced_forecast([.4]*10,config)==pytest.approx(.4)
        assert 0<=balanced_forecast([0.,1.,0.,1.,0.,1.],config)<=1


def test_locked_policy_satisfies_both_development_objectives() -> None:
    p=json.loads((OUT/"forecast/selection_lock.json").read_text(encoding="utf-8"));chosen=p["scores"][p["winner"]["id"]];base=p["scores"]["baseline"]
    ratios=np.array([[chosen[s][m]/base[s][m] for m in ("person_mae_0_100","rmse_0_100")] for s in base])
    assert np.all(ratios.mean(0)<=1+1e-12) and ratios.max()<=1.02+1e-12


def test_meaningful_short_or_negated_text_not_discarded() -> None:
    assert text_usable("好") and text_usable("我并不难过") and text_usable("但是我想继续")
    assert not text_usable("？！…") and not text_usable("https://example.org")


def test_disagreeing_experts_do_not_become_bad_quality() -> None:
    mask=np.array([[True,False,True]]);bank=np.array([[[.99,.005,.005],[.005,.005,.99],[.2,.6,.2]]])
    result=guarded_mask(mask,np.array(["我很开心"]),np.array([1.]),bank,{"threshold":.51,"text_confidence":.6})
    np.testing.assert_array_equal(mask,result)


def test_soft_degradation_requires_adequate_replacement() -> None:
    mask=np.array([[True,False,True]]);bank=np.array([[[.34,.33,.33],[.1,.1,.8],[.2,.2,.6]]])
    assert guarded_mask(mask,np.array(["普通内容"]),np.array([.25]),bank,{"threshold":.25,"text_confidence":.8})[0,2]
    bank[0,0]=[.9,.05,.05]
    assert not guarded_mask(mask,np.array(["普通内容"]),np.array([.25]),bank,{"threshold":.25,"text_confidence":.8})[0,2]


def test_black_frames_and_uniform_center_occlusion_detected() -> None:
    blank=np.zeros((3,128,128,3),np.uint8)
    black=quality_from_frames(blank)
    assert black.quality_score<=.25 and "underexposure_proxy" in black.reason_codes
    rng=np.random.default_rng(42);image=rng.integers(20,235,(3,128,128,3),dtype=np.uint8)
    clean=quality_from_frames(image);image[:,24:104,24:104]=127;blocked=quality_from_frames(image)
    assert blocked.quality_score<=.25 and clean.quality_score>blocked.quality_score
    assert "uniform_center_occlusion_proxy" in blocked.reason_codes


def test_unmeasured_quality_not_assumed_corrupt() -> None:
    mask=np.array([[True,False,True]])
    np.testing.assert_array_equal(availability(mask,np.array(["内容"]),np.array([np.nan]),.51),mask)


def test_unavailable_expert_never_contributes() -> None:
    bank=np.array([[[.8,.1,.1],[.1,.1,.8],[.1,.8,.1]]]);prior=np.array([.2,.3,.5])
    np.testing.assert_allclose(route(bank,np.array([[True,False,False]]),prior)[0],bank[0,0])
    np.testing.assert_allclose(route(bank,np.array([[False,False,False]]),prior)[0],prior)


def test_quality_selection_has_no_source_overlap() -> None:
    records=json.loads((OUT/"quality/guarded_selection_locks.json").read_text(encoding="utf-8"))
    for fold in records:assert not set(fold["train_groups"])&set(fold["held_groups"])


def test_correction_rejects_audio_and_invalid_probabilities(tmp_path: object) -> None:
    from experiments.round17_correction import predict_correction
    from pathlib import Path
    bank=np.ones((1,3,3))/3;mask=np.array([[True,True,True]])
    with pytest.raises(ValueError,match="音频"):predict_correction(bank,mask,np.array([1.]),Path("unused"),np.ones(3)/3)
    mask[:,1]=False;bank[:]=0
    with pytest.raises(ValueError,match="概率"):predict_correction(bank,mask,np.array([1.]),Path("unused"),np.ones(3)/3)
