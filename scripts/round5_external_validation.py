"""锁定研究模型后，在未参与任何选择的ECISA2021保留集上评测文字迁移。"""
from __future__ import annotations
import argparse
import gc
import json
import os
import pickle
from pathlib import Path
import sys
from typing import Any
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[key]="1"
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"algorithm"))
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from round5_text_features import encode
from experiments.round5_face_search import features
from experiments.round5_frozen_search import summary
from experiments.round5_ensemble import old_scores,normalized
from experiments.round5_finetune import MultimodalBERT,Rows,collate,logits_for,external_data
from experiments.round2_moe_train import grouped_delta_ci
from perception.knowledge_moe.features import FrozenCLIPEncoder,FeatureStandardizer,save_json,sha256_file


def run(root: Path,encoders: Path,previous: Path) -> dict[str,Any]:
    """先按开发分数锁定总方案，外部评测只报告，不触发改选。"""
    output=root/"external_validation";output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():raise ValueError("外部结果已存在，拒绝重复选择或覆盖")
    selection={}
    for component,key in [("frozen_search","selected"),("finetune",None),("ensemble","selection"),("face_search","validation"),("llm_search","validation")]:
        report=json.loads((root/component/"results.json").read_text(encoding="utf-8"));winner=report["winner"]
        score=(report["development"][winner]["best_macro_f1"] if component=="finetune" else report["development"][winner][key]["macro_f1"])
        selection[component]={"candidate":winner,"development_macro_f1":score,"historical_not_used_to_choose":True}
    winner=max(selection,key=lambda component:selection[component]["development_macro_f1"])
    if winner!="face_search":raise ValueError("当前外部复现仅实现face_search；不能据外部成绩更换模型")
    save_json(root/"final_selection_locked.json",{"selection":selection,"winner":winner,"criterion":"development macro-F1 only; never external/historical performance"})
    source=root/"fresh_holdout/holdout.jsonl"
    protocol={"selected_component":winner,"selected_candidate":selection[winner]["candidate"],"holdout_sha256":sha256_file(source),
        "selection_lock_sha256":sha256_file(root/"final_selection_locked.json"),"code_sha256":sha256_file(Path(__file__)),
        "modality":"text only; no face/video/audio available, explicitly masked; not a new multimodal movie benchmark",
        "reference":"original frozen concat evaluated with same text-only availability", "no_test_reselection":True}
    save_json(output/"preregister.json",protocol)
    data=external_data(source)
    data["source_groups"]=np.array([json.loads(line)["group"] for line in source.read_text(encoding="utf-8").splitlines()])
    n=len(data["labels"]);data["face_summary"]=np.zeros((n,29),np.float32)
    feature_cache=output/"text_features.npz"
    if feature_cache.exists():
        with np.load(feature_cache,allow_pickle=False) as saved:
            if str(saved["source_sha256"])!=sha256_file(source):raise ValueError("外部编码顺序不符")
            for key in ["roberta_cls","roberta_mean","bge_cls","bge_mean","text"]:data[key]=saved[key]
    else:
        texts=data["raw_text"].astype(str).tolist()
        for name,folder in [("roberta","chinese-roberta-wwm-ext"),("bge","bge-base-zh-v1.5")]:
            cls,mean=encode(texts,encoders/folder);data[name+"_cls"]=cls;data[name+"_mean"]=mean
        encoder=FrozenCLIPEncoder(encoders/"chinese-clip-vit-base-patch16","cuda")
        data["text"]=np.concatenate([encoder.encode_text(texts[start:start+32]) for start in range(0,n,32)])
        del encoder;gc.collect();torch.cuda.empty_cache()
        np.savez_compressed(feature_cache,source_sha256=np.array(sha256_file(source)),**{key:data[key] for key in ["roberta_cls","roberta_mean","bge_cls","bge_mean","text"]})
    reference=old_scores(data,previous)
    with (root/"face_search/selected.pkl").open("rb") as handle:selected=pickle.load(handle)
    x,_=features(data,selected["config"]["names"],selected["scalers"],selected["config"]["face_weight"])
    face=normalized(selected["model"].decision_function(x));face[:,1]+=selected["bias"]
    ft=json.loads((root/"finetune/results.json").read_text(encoding="utf-8"));name=ft["winner"]
    scaler=FeatureStandardizer();scaler.parameters=json.loads((root/"finetune/side_standardizer.json").read_text(encoding="utf-8"))
    tokenizer=AutoTokenizer.from_pretrained(encoders/"chinese-roberta-wwm-ext",local_files_only=True)
    rows=Rows(data,tokenizer,scaler);model=MultimodalBERT(encoders/"chinese-roberta-wwm-ext").cuda()
    model.load_state_dict(torch.load(root/"finetune"/(name+".pt"),map_location="cpu",weights_only=True))
    tuned=logits_for(model,DataLoader(rows,batch_size=64,collate_fn=collate));tuned[:,1]+=ft["development"][name]["bias"]
    scores={"original_concat_text_only":reference,"selected_face_model_text_only":face,"auxiliary_finetuned_text_only":tuned}
    metrics={name:summary(data["labels"],values) for name,values in scores.items()}
    ci=grouped_delta_ci(data["labels"],face.argmax(1),reference.argmax(1),data["source_groups"],3)
    for name,values in scores.items():np.savez_compressed(output/(name+".npz"),scores=values,labels=data["labels"],sample_ids=data["sample_ids"])
    report={"protocol":protocol,"n":n,"metrics":metrics,"selected_minus_original_macro_f1_ci95":ci,
        "human_participants_recruited":0,"synthetic":False,"model_choice_changed_after_external_results":False,"deployed":False}
    save_json(output/"results.json",report);print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["root","encoders","previous"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.root,args.encoders,args.previous)
