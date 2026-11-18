"""新来源标签、缺失模态及合成实验边界回归。"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from experiments.round5_finetune import MultimodalBERT,collate
from experiments.round5_face_search import features
from experiments.round5_availability_router import route_scores


def script(name: str) -> object:
    """加载本地脚本，不访问网络。"""
    source=Path(__file__).resolve().parents[2]/"scripts"/(name+".py")
    spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
    previous=list(sys.path)
    try:
        sys.path.insert(0,str(source.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:]=previous
    return module


def test_ecisa_explicit_label_mapping_never_defaults_unlabeled_to_neutral(tmp_path: Path) -> None:
    source=tmp_path/"source.xml"
    source.write_text('<Root><Doc ID="1"><Sentence ID="1" label="0">没有情感</Sentence><Sentence ID="2" label="1">隐式褒义</Sentence><Sentence ID="3" label="2">隐式贬义</Sentence><Sentence ID="4">没有标签</Sentence></Doc></Root>',encoding="utf-8")
    rows=script("round5_prepare_external").read_xml(source,"train")
    assert [row["label"] for row in rows]==[1,2,0]
    assert all(row["text"]!="没有标签" for row in rows)
    assert len({row["group"] for row in rows})==1


def test_missing_media_nan_cannot_contaminate_text_only_logits() -> None:
    class Encoder(nn.Module):
        def forward(self,input_ids: torch.Tensor,attention_mask: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(last_hidden_state=torch.ones((*input_ids.shape,4)))
    torch.manual_seed(42)
    model=MultimodalBERT.__new__(MultimodalBERT);nn.Module.__init__(model)
    model.encoder=Encoder();model.text=nn.Linear(8,128);model.visual=nn.Linear(512,64);model.audio=nn.Linear(46,32);model.head=nn.Linear(226,3);model.eval()
    tokens=torch.ones((2,3),dtype=torch.long);q=torch.zeros((2,2))
    a=model(tokens,tokens,torch.zeros((2,512)),torch.zeros((2,46)),q)
    b=model(tokens,tokens,torch.full((2,512),float('nan')),torch.full((2,46),float('nan')),q)
    assert torch.isfinite(b).all() and torch.equal(a,b)


def test_dynamic_padding_masks_only_real_tokens() -> None:
    def row(ids: list[int],label: int) -> tuple:return ids,torch.zeros(512),torch.zeros(46),torch.zeros(2),torch.tensor(label)
    batch=collate([row([101,8,102],1),row([101,102],0)])
    assert batch["input_ids"].tolist()==[[101,8,102],[101,102,0]]
    assert batch["attention_mask"].tolist()==[[1,1,1],[1,1,0]]
    assert batch["labels"].tolist()==[1,0]


def test_face_feature_scaling_is_fitted_only_on_training_data() -> None:
    training={"face_summary":np.array([[0.,1.],[2.,3.]],np.float32)}
    _,scalers=features(training,["face_summary"])
    mean=scalers["face_summary"].mean_.copy()
    evaluation={"face_summary":np.full((3,2),100.,np.float32)}
    transformed,_=features(evaluation,["face_summary"],scalers)
    assert np.array_equal(mean,scalers["face_summary"].mean_)
    assert np.isfinite(transformed).all()


def test_simulated_support_uses_real_planner_and_marks_generated_traces() -> None:
    result=script("round5_simulated_users").support_simulation(4,15)
    assert result["synthetic"] is True and result["turns"]==60
    assert not result["failures"] and all(row["synthetic"] for row in result["sample_trace"])


def test_synthetic_forecast_keeps_people_disjoint_and_does_not_claim_clinical_score(tmp_path: Path) -> None:
    result=script("round5_simulated_users").forecast_simulation(tmp_path,20,30)
    assert result["synthetic"] is True and result["participant_split"]==[14,3,3]
    with np.load(tmp_path/"synthetic_forecast.npz",allow_pickle=False) as saved:
        assert bool(saved["synthetic"]) and len(np.unique(saved["test_users"]))==3
    assert 'not questionnaire/clinical score' in result["target"]


def test_availability_router_does_not_read_labels_or_source_and_never_uses_absent_face() -> None:
    data={"face_summary":np.zeros((3,29)),"modality_mask":np.array([[1,1,1],[1,1,0],[1,1,1]],bool),
          "quality":np.array([[1,1,1],[1,1,0],[1,1,1]],np.float32),"labels":np.array([0,1,2])}
    data["face_summary"][:,24]=[.6,1.,.4]
    primary=np.array([[1.,2.,3.],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]])
    fallback=np.array([[3.,2.,1.],[3.,2.,1.],[3.,2.,1.]])
    scores,mask=route_scores(data,primary,fallback,.6)
    assert mask.tolist()==[True,False,False] and np.isfinite(scores).all()
    changed={**data,"labels":np.array([2,0,1]),"source":np.array(['different']*3)}
    assert np.array_equal(route_scores(changed,primary,fallback,.6)[0],scores)


def test_router_abstains_when_required_text_is_missing_even_with_plausible_scores() -> None:
    data={"face_summary":np.zeros((2,29)),"modality_mask":np.array([[0,1,1],[1,0,0]],bool),
          "quality":np.array([[0,1,1],[0,0,0]],np.float32)}
    data["face_summary"][:,24]=1.
    scores,used_face=route_scores(data,np.ones((2,3)),np.ones((2,3)),.6)
    assert np.isnan(scores).all() and not used_face.any()


def test_official_holdout_parser_repairs_literal_xml_without_dropping_labels(tmp_path: Path) -> None:
    import json
    archive=tmp_path/'archive';archive.mkdir();external=tmp_path/'external';external.mkdir();bundle=tmp_path/'bundle';bundle.mkdir()
    (archive/'SMP2021-ECISA_Test_答案.txt').write_text('1-1\t0$\n1-1\t0$\n2-1\t2$\n',encoding='utf-8')
    (archive/'SMP2021-ECISA_Test.xml').write_text('<SMP2021-ECISA><Doc ID="1"><Sentence ID="1">数学&诗歌的<故事</Sentence></Doc><Doc ID="2"><Sentence ID="1">这句话没有重叠内容</Sentence></Doc></SMP2021-ECISA>',encoding='utf-8')
    for name in ['train','external_valid']:(external/(name+'.jsonl')).write_text('',encoding='utf-8')
    for name in ['train','selection','calibration','test']:np.savez(bundle/(name+'.npz'),raw_text=np.array(['完全不同']))
    result=script('round5_prepare_holdout').run(archive,external,bundle,tmp_path/'output')
    rows=[json.loads(line) for line in (tmp_path/'output/holdout.jsonl').read_text(encoding='utf-8').splitlines()]
    assert result['official_answer_count']==2 and result['identical_duplicate_answers_collapsed']==1
    assert result['xml_repairs']=={'unescaped_ampersands':1,'literal_lt_in_body':1}
    assert rows[0]['text']=='数学&诗歌的<故事' and [row['label'] for row in rows]==[1,0]
