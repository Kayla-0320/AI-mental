"""中性保护、像素重编码、跨用户校准和图谱待审边界的回归检查。"""
from __future__ import annotations
import copy
import json
from typing import Any
import numpy as np
import pytest
from experiments.round18_neutral import apply_policy
from experiments.round18_visual_guard import unusable
from experiments.round18_images import altered,CONDITIONS
from experiments.round18_uncertainty import weighted_quantile,evaluate_band,scale,point
from experiments.round18_runtime import OUT,predict_day,predict_neutral,predict_quality_features
from experiments.round18_graph import build,validate,retrieve,jsonld


def payload() -> dict[str,Any]:
    """原量表夹具，不是真人自评或真人评分。"""
    return {"source_schema":"openesm:beck:sad:1_5","target_day":3,"observations":[{"person":"fixture","day":i,"value":3.} for i in range(3)]}


def test_neutral_exit_protection_only_blocks_small_margin() -> None:
    b=np.array([[.2,.6,.2],[.2,.6,.2]]);m=np.array([[.5,.4,.1],[.8,.1,.1]])
    p=apply_policy(b,m,{"weight":1.,"bias":0.,"leave_margin":.2})
    np.testing.assert_allclose(p[0],b[0]);assert p[1].argmax()==0


def test_zero_weight_is_baseline_fallback() -> None:
    b=np.array([[.2,.5,.3]])
    np.testing.assert_allclose(apply_policy(b,b[:,::-1],{"weight":0.,"bias":0.,"leave_margin":0.}),b,atol=1e-15)


def test_meta_selection_uses_disjoint_source_groups() -> None:
    locks=json.loads((OUT/"neutral/selection_locks.json").read_text(encoding="utf-8"))
    for row in locks:
        assert not set(row["train_groups"])&set(row["held_groups"])
        a,b=row["winner"]["metrics"],row["base_inner"]
        assert a["neutral_f1"]>=b["neutral_f1"]-1e-12
        assert a["neutral_recall"]>=b["neutral_recall"]-.01-1e-12
        assert a["accuracy"]>=b["accuracy"]-.002-1e-12


@pytest.mark.parametrize("change",["audio","missing","nan","sum","quality"])
def test_neutral_runtime_rejects_invalid_or_unvalidated_inputs(change: str) -> None:
    b=np.full((1,3,3),1/3);m=np.array([[True,False,True]]);q=np.array([.8])
    if change=="audio":m[0,1]=True
    if change=="missing":m[0,2]=False
    if change=="nan":b[0,0,0]=np.nan
    if change=="sum":b[0,0,0]=.9
    if change=="quality":q[0]=1.1
    with pytest.raises(ValueError):predict_neutral(b,m,q,OUT/"neutral/fold0")


def test_hard_pixel_failure_does_not_discard_partial_information() -> None:
    x=np.random.default_rng(42).integers(20,235,(3,128,128,3),dtype=np.uint8)
    assert unusable(altered(x,"black"))
    assert unusable(np.empty((0,128,128,3),dtype=np.uint8))
    for name in ("clean","dark","blur","center_occlusion"):assert not unusable(altered(x,name))
    constant=x.copy();constant[:]=[20,30,40];assert unusable(constant)


def test_saved_corruptions_really_change_encoded_visual_vectors() -> None:
    z=np.load(OUT/"images/reencoded.npz",allow_pickle=False)
    for c in CONDITIONS[1:]:assert not np.allclose(z["clean"],z[c])
    assert z["clean"].shape==(512,512) and int(z["frame_counts"].sum())==1522
    assert json.loads((OUT/"images/encoder.json").read_text(encoding="utf-8"))["audio_decoded"] is False


def test_image_training_source_isolation_and_conservative_routing() -> None:
    for r in json.loads((OUT/"images/fold_audits.json").read_text(encoding="utf-8")):
        assert not set(r["train_groups"])&set(r["held_groups"])
    for c in CONDITIONS:
        g=np.load(OUT/f"images/{c}_guard.npz",allow_pickle=False);old=np.load(OUT/f"images/{c}_predictions.npz",allow_pickle=False)
        np.testing.assert_array_equal(g["selected"][~g["drop"]],old["uniform"][~g["drop"]])
        assert int(g["drop"].sum())==(512 if c=="black" else 0)


@pytest.mark.parametrize("change",["schema","future","duplicate","order","nan","scale","person"])
def test_forecast_rejects_schema_leakage_and_mixed_people(change: str) -> None:
    p=payload()
    if change=="schema":p["source_schema"]="project:mood"
    if change=="future":p["observations"][-1]["day"]=3
    if change=="duplicate":p["observations"][-1]["day"]=1
    if change=="order":p["observations"].reverse()
    if change=="nan":p["observations"][-1]["value"]=float("nan")
    if change=="scale":p["observations"][-1]["value"]=6
    if change=="person":p["observations"][-1]["person"]="other"
    with pytest.raises(ValueError):predict_day(p)


def test_forecast_gap_reset_and_constant_prediction() -> None:
    assert predict_day(payload())["prediction"]==pytest.approx(3.)
    p=payload();p["target_day"]=10;p["observations"]=[{"person":"p","day":i,"value":3.} for i in (0,1,2,8,9)]
    assert predict_day(p)["status"]=="abstain"


def test_calibration_people_never_used_in_selection() -> None:
    lock=json.loads((OUT/"forecast/calibration_lock.json").read_text(encoding="utf-8"))
    assert "beck" not in lock["partitions"]
    for r in lock["partitions"].values():assert not set(r["selection_people"])&set(r["calibration_people"])
    assert lock["point_method"]=="ema4"


def test_uncertainty_width_penalty_and_past_only_scale() -> None:
    a=evaluate_band(np.array([.5]),np.array([.5]),np.array([.1]));b=evaluate_band(np.array([.5]),np.array([.5]),np.array([.5]))
    assert a["coverage"]==b["coverage"]==1 and a["mean_interval_score_0_100"]<b["mean_interval_score_0_100"]
    assert scale([.5]*6)==.05
    assert weighted_quantile(np.array([0.,1.]),np.array([9.,1.]),.5)==0
    assert point([.4]*8,"ema4")==pytest.approx(.4)


def test_graph_every_edge_has_distinguished_provenance() -> None:
    g=build();assert not validate(g) and len(g["nodes"])==28 and len(g["edges"])==29
    assert sum(e["evidence_type"]=="public_guideline" for e in g["edges"])==18
    for e in g["edges"]:
        assert e["source_id"] in g["sources"] and e["locator"] and e["review_status"]=="pending"
        assert (e["evidence_type"]=="engineering_contract")== (e["source_id"]=="PROJECT")


def test_graph_requires_explicit_research_optin_for_pending_concepts() -> None:
    g=build()
    for kwargs in ({},{"research":True},{"include_unreviewed":True}):assert not retrieve(g,"锚定当下","act",**kwargs)["facts"]
    r=retrieve(g,"锚定当下","understand",research=True,include_unreviewed=True)
    assert r["facts"] and all(not e["applied_to_user"] and not e["human_reviewed"] for e in r["facts"])


@pytest.mark.parametrize("query,goal",[("锚定当下","pause"),("锚定当下","listen"),("锚定当下","unknown"),("我不要锚定当下","act"),("朋友说锚定当下","act"),("他说“锚定当下”","act"),("锚定能诊断什么","act")])
def test_research_graph_respects_goal_quote_and_negation(query: str,goal: str) -> None:
    assert not retrieve(build(),query,goal,research=True,include_unreviewed=True)["facts"]


@pytest.mark.parametrize("corruption",["dangling","relation","source","node_approval","edge_approval","duplicate"])
def test_graph_rejects_invalid_or_fake_review_records(corruption: str) -> None:
    g=build()
    if corruption=="dangling":g["edges"][0]["object"]="missing"
    if corruption=="relation":g["edges"][0]["predicate"]="diagnoses"
    if corruption=="source":g["edges"][0]["source_id"]="missing"
    if corruption=="node_approval":g["nodes"][1]["review_status"]="approved"
    if corruption=="edge_approval":g["edges"][0]["review_status"]="approved"
    if corruption=="duplicate":g["edges"].append(copy.deepcopy(g["edges"][0]))
    assert validate(g)
    with pytest.raises(ValueError):retrieve(g,"锚定当下","act")


def test_rejected_graph_items_never_appear_even_in_research() -> None:
    g=build()
    for n in g["nodes"]:n["review_status"]="rejected"
    assert not retrieve(g,"锚定当下","act",research=True,include_unreviewed=True)["facts"]


def test_jsonld_relations_have_defined_iri_mapping_and_provenance_resources() -> None:
    g=build();j=jsonld(g);context=j["@context"]
    for key in ("subject","predicate","object","source"):assert context[key]["@type"]=="@id"
    statements=[n for n in j["@graph"] if n.get("@type")=="rdf:Statement"]
    assert len(statements)==29 and all(n["predicate"].startswith("urn:aic:relation:") for n in statements)
    ids={n["@id"] for n in j["@graph"]};assert all(n["source"] in ids for n in statements)


def test_all_saved_runtime_replays_match() -> None:
    r=json.loads((OUT/"verification.json").read_text(encoding="utf-8"))
    assert sum(x["samples"] for x in r["neutral"])==2876
    assert all(x["max_error"]<1e-12 for x in r["neutral"]+r["images"])
    assert r["forecast"]["windows"]==1753 and r["forecast"]["max_error"]<1e-12
    assert r["human_ratings_generated"]==0 and r["unreviewed_default_abstains"]


def test_natural_quality_probe_does_not_fake_ranking_or_missing_labels() -> None:
    r=json.loads((OUT/"natural_quality/results.json").read_text(encoding="utf-8"))
    assert r["n"]+len(r["unmatched_names_quarantined_without_replacement"])==128
    assert r["constant_proxy_correlation_undefined"] and r["spearman_quality_MOS"] is None
    assert r["auc_low_MOS_below_2_5"]==.5 and not r["parameters_tuned_on_MOS"]


def test_quality_head_holdout_is_disjoint_from_selection_and_initial_probe() -> None:
    s=json.loads((OUT/"quality_model/split_lock.json").read_text(encoding="utf-8"));prior=set(json.loads((OUT/"natural_quality/selected_ids.json").read_text(encoding="utf-8")))
    assert [len(s[k]) for k in ("train","validation","test")]==[768,192,256]
    assert not (set(s["train"])&set(s["validation"]) or set(s["train"])&set(s["test"]) or set(s["validation"])&set(s["test"]))
    assert not set(sum(s.values(),[]))&prior
    hashes=json.loads((OUT/"quality_model/image_hashes.json").read_text(encoding="utf-8"));assert len({r["sha256"] for r in hashes})==1216
    lock=json.loads((OUT/"quality_model/selection_lock.json").read_text(encoding="utf-8"));assert lock["no_test_labels_used"]


@pytest.mark.parametrize("change",["dimension","nonfinite"])
def test_quality_head_rejects_incompatible_features(change: str) -> None:
    x=np.zeros((2,522))
    if change=="dimension":x=x[:,:512]
    if change=="nonfinite":x[0,0]=np.nan
    with pytest.raises(ValueError):predict_quality_features(x,OUT/"quality_model")


def test_quality_head_runtime_exact_and_original_scale_preserved() -> None:
    x=np.load(OUT/"quality_model/features.npz",allow_pickle=False);y=np.load(OUT/"quality_model/test_predictions.npz",allow_pickle=False)
    p=predict_quality_features(x["X"][960:],OUT/"quality_model")
    np.testing.assert_allclose(p,y["selected"],atol=1e-12);assert np.all((p>=1)&(p<=5))
    assert json.loads((OUT/"quality_model/manifest.json").read_text(encoding="utf-8"))["production_promoted"] is False


def test_constant_MOS_prediction_correlation_remains_undefined_without_nan() -> None:
    from experiments.round18_quality_model import metrics
    p=np.full(256,3.141592653589793)
    r=metrics(np.linspace(1.,5.,256),p)
    assert r["spearman"] is None and r["auc_MOS_below_2_5"]==.5
    json.dumps(r,allow_nan=False)
