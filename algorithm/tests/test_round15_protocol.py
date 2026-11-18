"""验证因果适应、用户隔离、回退和具体回应实验的信息边界。"""
from __future__ import annotations
import copy
import json
import numpy as np
import pytest
from experiments.round15_forecast import forecast,configs,choose,OUT
from experiments.round15_runtime import reliable_choice,predict_new_user
from experiments.round15_support import new_cases,messages


@pytest.mark.parametrize("invalid",[[],[.1,.2],[.1,float('nan'),.3],[.1,1.1,.2]])
def test_invalid_history(invalid: list[float]) -> None:
    with pytest.raises(ValueError):forecast(invalid,{"kind":"last"})


def test_local_adaptation_has_no_future_dependency() -> None:
    history=[.1,.2,.3,.2,.1,.5,.4]
    before=forecast(history,{"kind":"local"})
    full=history+[.9,.9]
    assert forecast(full[:len(history)],{"kind":"local"})==before


def test_cold_start_falls_back_without_population_shrinkage() -> None:
    assert forecast([0.,0.,0.],{"kind":"local"})==0.
    assert forecast([1.,1.,1.],{"kind":"local"})==1.


def test_small_validation_conservatively_reverts() -> None:
    choice,reason=reliable_choice([{"person":"p"}]*43)
    assert choice["kind"]=="last" and reason["fallback"]


def test_wrong_scale_rejected() -> None:
    with pytest.raises(ValueError):predict_new_user([.1,.2,.3],{"config":{"kind":"last"},"selection":{}},"project_stress")


def test_mutating_heldout_labels_does_not_affect_selection() -> None:
    rows=[{"person":str(i),"history":[.2,.3,.4],"y":.3} for i in range(12)]
    first,_=choose(rows,configs()); separate_test=[{"y":.9}]
    separate_test[0]["y"]=0
    second,_=choose(rows,configs())
    assert first==second


def test_generation_never_receives_scoring_keywords() -> None:
    for case in new_cases():
        altered=copy.deepcopy(case);altered["corrected_pattern"]="SECRET_EVAL_PATTERN";altered["old_claim"]="SECRET_OLD_LABEL"
        assert messages(case,"F")==messages(altered,"F")
        assert messages(case,"E")[1:]==messages(case,"F")[1:]


def test_new_case_ids_do_not_overlap_previous_cases() -> None:
    from experiments.round14_support import cases
    fresh=new_cases()
    assert len(fresh)==40 and len({c["case_id"] for c in fresh})==40
    assert not {c["case_id"] for c in fresh}&{c["case_id"] for c in cases()}


def test_actual_outer_users_excluded_from_selection() -> None:
    locks=json.loads((OUT/"forecast/lopo_selection_locks.json").read_text(encoding="utf-8"))
    protocol=json.loads((OUT/"forecast/protocol.json").read_text(encoding="utf-8"))
    assert len(locks)==49
    for person,lock in locks.items():
        assert person not in lock["validation_people"]
        assert lock["validation_max_target"]<protocol["strict_split"]["test_start_day"]


def test_saved_predictions_replay_exactly() -> None:
    locks=json.loads((OUT/"forecast/lopo_selection_locks.json").read_text(encoding="utf-8"))
    rows=json.loads((OUT/"forecast/lopo_windows.json").read_text(encoding="utf-8"))
    expected=np.load(OUT/"forecast/lopo_predictions.npz",allow_pickle=False)["selected"]
    actual=[forecast(r["history"],locks[r["person"]]["winner"]) for r in rows]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)


def test_expanded_checks_catch_observed_goal_failures() -> None:
    from experiments.round15_repair import extra_checks
    assert extra_checks({"goal":"pause"},"明白了，你先联系管理员吧。")["pause_adds_action_proxy"]
    assert extra_checks({"goal":"understand"},"那你更关心的是电子车票打不开的问题吗？")["reconfirm_known_concern_proxy"]
    assert not any(extra_checks({"goal":"pause"},"好的，今天先到这里。").values())


def test_generated_pairs_complete_and_blinded() -> None:
    rows=[json.loads(x) for x in (OUT/"support/generations.jsonl").read_text(encoding="utf-8").splitlines()]
    selected=[r for r in rows if r["split"]=="new_authored_challenge"]
    assert len(selected)==80 and len({(r["case_id"],r["arm"]) for r in selected})==80
    for identifier in {r["case_id"] for r in selected}:
        pair=[r for r in selected if r["case_id"]==identifier]
        assert {r["arm"] for r in pair}=={"E","F"} and pair[0]["history"]==pair[1]["history"]
    public=json.loads((OUT/"support/blind/replies.json").read_text(encoding="utf-8"))
    assert len(public)==80 and all("arm" not in r and "case_id" not in r for r in public)
