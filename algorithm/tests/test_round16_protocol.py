"""新来源锁型、跨日推理、真实人评与专家资格的必要回归。"""
from __future__ import annotations
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
from experiments.round16_external import OUT,DATA,SOURCES,prepare,robust_forecast
from experiments.round16_runtime import predict_observations,verify
from experiments.round16_human import analyze
from experiments.round16_moe import mix,gate_features
from experiments.round16_statistics import holm


def payload() -> dict:
    """返回隔离的输入契约夹具，不写入实际研究数据。"""
    return {"dataset":"dejonckheere","source_schema":"openesm:dejonckheere:sad:0_100","target_day":3,
        "observations":[{"person":"fixture","day":i,"value":40.} for i in range(3)]}


@pytest.mark.parametrize("change",["future","duplicate","unsorted","scale","person","nan","range"])
def test_runtime_rejects_invalid_history(change: str) -> None:
    p=payload()
    if change=="future":p["observations"][-1]["day"]=3
    if change=="duplicate":p["observations"][-1]["day"]=1
    if change=="unsorted":p["observations"].reverse()
    if change=="scale":p["source_schema"]="project_mood"
    if change=="person":p["observations"][-1]["person"]="another"
    if change=="nan":p["observations"][-1]["value"]=float("nan")
    if change=="range":p["observations"][-1]["value"]=101.
    with pytest.raises(ValueError):predict_observations(p)


def test_long_gap_resets_without_filling_zero() -> None:
    p=payload();p["target_day"]=10
    p["observations"]=[{"person":"p","day":i,"value":40.} for i in (0,1,2,8,9)]
    assert predict_observations(p)["status"]=="abstain"


def test_constant_history_and_robust_bounds() -> None:
    config={"kind":"robust","eta":10.,"mix":1.}
    assert robust_forecast([.4]*12,config)==pytest.approx(.4)
    for h in ([0.,0.,0.,0.,1.,1.],[1.,1.,1.,1.,0.,0.]):assert 0<=robust_forecast(list(h),config)<=1


def test_conflicting_public_beeps_quarantined(tmp_path: Path) -> None:
    p=tmp_path/"data.tsv"
    pd.DataFrame({"id":["p"]*4,"day":[1,1,2,2],"beep":[1,1,1,2],"sad":[20,80,0,0]}).to_csv(p,sep="\t",index=False)
    daily,audit=prepare(p,SOURCES["dejonckheere"])
    assert audit["conflicting_rows_quarantined"]==2 and daily.day.tolist()==[2] and daily.value.tolist()==[0.]


def test_published_source_digests_and_no_target_training() -> None:
    for spec in SOURCES.values():assert hashlib.md5((DATA/spec["filename"]).read_bytes()).hexdigest()==spec["md5"]
    p=json.loads((OUT/"forecast/external_protocol.json").read_text(encoding="utf-8"))
    lock=json.loads((OUT/"forecast/source_selection_lock.json").read_text(encoding="utf-8"))
    assert p["fresh_source_training"] is False and p["methods"]["robust"]==lock["robust_candidate"]


def test_all_external_runtime_predictions_reproduce() -> None:
    results=verify()
    assert sum(r["windows"] for r in results.values())==4546
    assert all(r["maximum_error"]<1e-12 for r in results.values())


def test_empty_real_ratings_stay_unknown(tmp_path: Path) -> None:
    shutil.copytree(OUT/"human/blind",tmp_path/"blind");shutil.copytree(OUT/"human/private",tmp_path/"private")
    r=analyze(tmp_path)
    assert r["completed_semantic_rows"]==0 and r["mean_semantic_quality"] is None and not r["model_used_as_rater"]
    assert all(v["action_matches_concern_rate"] is None for v in r["semantic_by_arm"].values())


def test_filled_scores_without_human_declaration_rejected(tmp_path: Path) -> None:
    shutil.copytree(OUT/"human/blind",tmp_path/"blind");shutil.copytree(OUT/"human/private",tmp_path/"private")
    path=tmp_path/"blind/rater_1.csv"
    with path.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    rows[0].update(fidelity_1_5="4",appropriateness_1_5="4",correction_violation="0",goal_violation="0",unsupported_details="0",repeat_question="0")
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match="独立人评声明"):analyze(tmp_path)


def test_expert_eligibility_excludes_invisible_features() -> None:
    bank=np.array([[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]]*3)
    mask=np.array([[True,False,False],[False,False,True],[False,False,False]])
    prior=np.array([.2,.3,.5]);p=mix(bank,mask,None,"uniform",prior)
    np.testing.assert_allclose(p[0],bank[0,0]);np.testing.assert_allclose(p[1],bank[1,1]);np.testing.assert_allclose(p[2],prior)
    assert gate_features(bank,mask).shape==(3,14)


def test_holm_preserves_order_and_bounds() -> None:
    assert holm([.03,.001,.2])==pytest.approx([.06,.003,.2])


def test_semantic_evidence_must_match_actual_reply(tmp_path: Path) -> None:
    shutil.copytree(OUT/"human/blind",tmp_path/"blind");shutil.copytree(OUT/"human/private",tmp_path/"private")
    path=tmp_path/"blind/rater_1.csv"
    with path.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    rows[0].update(fidelity_1_5="4",appropriateness_1_5="4",correction_violation="0",goal_violation="0",unsupported_details="0",repeat_question="0",
        action_matches_concern="NA",reasks_known_information="0",unfounded_interpretation="0",evidence_user_quote="fixture",evidence_reply_quote="not an actual response quote")
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (tmp_path/"blind/rater_declarations.json").write_text(json.dumps([{"rater_id":"rater_1","human_completed":True,"independent_of_development":True,"declared_at":"2026-11-17T12:00:00+08:00"}]),encoding="utf-8")
    with pytest.raises(ValueError,match="回复证据"):analyze(tmp_path)


def test_exported_tv_inference_matches_saved_smoke() -> None:
    from experiments.round16_tv_runtime import predict_tv
    z=np.load(OUT/"moe/export/smoke_input.npz",allow_pickle=False)
    actual=predict_tv(z["T"],z["V"],z["mask"],OUT/"moe/export/fold0")
    np.testing.assert_allclose(actual,z["expected_uniform"],rtol=0,atol=1e-12)


def test_modified_tv_model_rejected(tmp_path: Path) -> None:
    from experiments.round16_tv_runtime import predict_tv
    shutil.copytree(OUT/"moe/export/fold0",tmp_path/"model")
    path=tmp_path/"model/gate.pkl";path.write_bytes(path.read_bytes()+b"changed")
    z=np.load(OUT/"moe/export/smoke_input.npz",allow_pickle=False)
    with pytest.raises(ValueError,match="摘要"):predict_tv(z["T"],z["V"],z["mask"],tmp_path/"model")
