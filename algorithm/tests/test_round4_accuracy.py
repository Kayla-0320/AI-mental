"""第四轮评测边界及已发现的偏好误判回归。"""
from __future__ import annotations
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

from experiments.round4_accuracy import folds_for, knowledge_batch, linear_features, predict
from intervention.support_understanding import prepare_support, apply_feedback
from perception.knowledge_moe.features import FeatureStandardizer
from shared import SupportFeedback


def script(name: str) -> object:
    """加载仓库脚本，不依赖当前工作目录。"""
    path=Path(__file__).resolve().parents[2]/"scripts"/(name+".py")
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_spearman_ties_permutation_and_undefined_constant() -> None:
    module=script("round3_longitudinal_predict")
    truth=np.array([1.,1.,2.,3.]);prediction=np.array([1.,2.,2.,3.])
    actual=module.regression_metrics(truth,prediction)["spearman"]
    assert actual==pytest.approx(5/6)
    order=np.array([3,1,2,0])
    assert module.regression_metrics(truth[order],prediction[order])["spearman"]==pytest.approx(actual)
    assert module.regression_metrics(truth,np.ones(4))["spearman"] is None
    assert module.regression_metrics(np.ones(4),prediction)["r2"] is None
    assert module.regression_metrics(np.array([1.,2.]),np.array([2.,1.]))["spearman"]==pytest.approx(-1.)


@pytest.mark.parametrize("truth,prediction",[([],[]),([1.,2.],[1.]),([1.,np.nan],[1.,2.]),([[1.,2.]],[[1.,2.]])])
def test_regression_rejects_invalid_or_misaligned_inputs(truth: list,prediction: list) -> None:
    with pytest.raises(ValueError):script("round3_longitudinal_predict").regression_metrics(np.array(truth),np.array(prediction))


def test_group_folds_never_mix_sources_and_cover_every_row_once() -> None:
    data={"labels":np.tile([0,1,2],20),"source_groups":np.repeat(np.arange(20),3)}
    splits=folds_for(data)
    assert np.array_equal(np.sort(np.concatenate([valid for _,valid in splits])),np.arange(60))
    for train,valid in splits:
        assert not set(data["source_groups"][train]) & set(data["source_groups"][valid])


def test_linear_preprocessing_keeps_training_statistics_and_ignores_missing_nan() -> None:
    scaler=FeatureStandardizer().fit({"text":np.array([[1.,2.],[3.,4.]])},np.ones((2,1),bool))
    before=__import__('copy').deepcopy(scaler.parameters)
    data={"text":np.array([[100.,200.],[np.nan,np.nan]]),"modality_mask":np.array([[1,0,0],[0,0,0]],bool),
          "quality":np.array([[1.,0.,0.],[0.,0.,0.]])}
    values=linear_features(data,["text"],scaler)
    assert np.isfinite(values).all() and not values[1].any()
    assert scaler.parameters==before


def test_knowledge_negative_control_preserves_match_mask_without_labels() -> None:
    kg=torch.arange(24,dtype=torch.float32).reshape(6,4)
    present=torch.tensor([1,0,1,1,0,1],dtype=torch.bool);kg[~present]=0
    original=({},torch.ones(6,3,dtype=torch.bool),torch.ones(6,3),kg,present,torch.arange(6))
    shuffled=knowledge_batch(original,"shuffled")
    assert torch.equal(shuffled[4],present) and torch.equal(shuffled[5],original[5])
    assert sorted(map(tuple,shuffled[3][present].tolist()))==sorted(map(tuple,kg[present].tolist()))
    assert torch.equal(original[3],kg) and not knowledge_batch(original,"null")[4].any()


def test_linear_prediction_abstains_without_any_positive_quality_observation() -> None:
    class NeverCalled:
        def decision_function(self, values: np.ndarray) -> np.ndarray:
            raise AssertionError("不能给全缺失输入调用分类器")
    data={"text":np.full((2,2),np.nan),"modality_mask":np.array([[False,False,False],[True,False,False]]),
          "quality":np.zeros((2,3),np.float32),"labels":np.array([0,1])}
    fitted={"config":{"kind":"linear","names":["text"]},"model":NeverCalled(),"scaler":FeatureStandardizer()}
    assert np.isnan(predict(fitted,data,None)).all()


@pytest.mark.parametrize("message,goal,plan",[
    ('今天朋友说“帮我想办法”，我只是在转述。',"unspecified","listen"),
    ('今天考试失利，先听我说，但是现在帮我理清。',"understand","reflect_check"),
    ('今天考试失利，先听我说，再一起想办法。',"listen","listen"),
    ('今天考试失利，想先理清，再帮我想个办法。',"understand","reflect_check"),
    ('今天考试失利，我不是不想说。',"unspecified","listen"),
    ('今天考试失利，我不想休息，想知道怎么做。',"act","collaborate_action"),
    ('今天同学说想休息，我想理清自己的事。',"understand","reflect_check"),
    ('今天考试失利，只想被理解。',"understand","reflect_check"),
])
def test_direct_preference_respects_negation_quotation_and_sequence(message: str,goal: str,plan: str) -> None:
    state=prepare_support("s","m",message,[])
    assert state.goal==goal and state.plan.action==plan
    assert all(message[item.start:item.end]==item.quote for item in state.items)


def test_negating_old_action_preference_does_not_keep_action_plan() -> None:
    original=prepare_support("s","m","今天考试失利，帮我想办法。",[])
    updated=prepare_support("s","m2","现在不要帮我想办法",[{"role":"user","id":"m","content":"今天考试失利，帮我想办法。"}],asdict(original))
    assert updated.goal=="unspecified" and updated.plan.action=="listen"


def test_explicit_decline_still_dominates_new_free_text_action_request() -> None:
    state=prepare_support("s","m","今天考试失利，只想被理解。",[])
    state=apply_feedback(state,SupportFeedback("decline",state.revision,"decline"))
    updated=prepare_support("s","m2","帮我想办法",[],asdict(state))
    assert updated.declined and updated.plan.action=="listen"


@pytest.mark.parametrize("message",['朋友说“换个话题”，但我还想理清这件事。','今天不用换个话题，继续一起梳理。'])
def test_quoted_or_negated_topic_change_keeps_confirmed_context(message: str) -> None:
    original=prepare_support("s","m","今天朋友没回消息，我觉得被忽视了。",[])
    original=apply_feedback(original,SupportFeedback("confirm",original.revision,"confirm",original.items[0].id))
    updated=prepare_support("s","m2",message,[{"role":"user","id":"m","content":"今天朋友没回消息，我觉得被忽视了。"}],asdict(original))
    assert any(item.id==original.items[0].id and item.status=="confirmed" for item in updated.items)
    assert updated.plan.method=="explicit_goal_rules_v2"


def test_correction_log_cannot_be_reassigned_to_another_evidence_item() -> None:
    state=prepare_support("s","m","今天朋友没回消息，我觉得被忽视了。",[])
    state=apply_feedback(state,SupportFeedback("correct",state.revision,"correct",state.items[0].id,"我只是担心朋友没看到消息"))
    payload=asdict(state);payload["items"][0]["id"]="fabricated-item"
    updated=prepare_support("s","m2","我想继续理清这件事情",[],payload)
    assert all(item.id!="fabricated-item" for item in updated.items)


def test_automatic_quality_decision_is_blind_to_scenario_name_and_truth_label() -> None:
    module=script("round4_quality_benchmark")
    data={"text":np.ones((2,4)),"audio":np.ones((2,3)),"visual":np.ones((2,4)),
          "modality_mask":np.ones((2,3),bool),"quality":np.ones((2,3),np.float32),"labels":np.array([0,2])}
    entry={"audio":data["audio"],"visual":data["visual"],"quality":np.array([[1,.1,1],[1,1,.1]],np.float32),"usable":np.ones((2,3),bool)}
    first=module.apply_condition(data,{"silence":entry},"silence","automatic",.25)
    changed={**data,"labels":np.array([2,0])}
    second=module.apply_condition(changed,{"clean":entry},"clean","automatic",.25)
    assert np.array_equal(first["modality_mask"],second["modality_mask"])
    assert np.array_equal(first["quality"],second["quality"])
    assert data["modality_mask"].all()
