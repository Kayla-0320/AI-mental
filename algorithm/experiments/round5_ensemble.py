"""只在开发数据选择集成与中性判决；原历史测试样本不参与拟合。"""
from __future__ import annotations
import argparse
import gc
import json
import os
import pickle
from pathlib import Path
import sys
from typing import Any
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from experiments.round2_moe_train import load_split,grouped_delta_ci
from experiments.round5_frozen_search import summary,matrix
from experiments.round5_finetune import MultimodalBERT,Rows,collate,logits_for
from perception.knowledge_moe.features import FeatureStandardizer,save_json,sha256_file
from perception.knowledge_moe.model import ConcatenationHead


def normalized(scores: np.ndarray) -> np.ndarray:
    """统一不同分类头分数尺度，不宣称概率校准。"""
    return (scores-scores.mean(1,keepdims=True))/np.maximum(scores.std(1,keepdims=True),1e-6)


def select_bias(y: np.ndarray,scores: np.ndarray) -> tuple[float,dict]:
    """固定网格选择中性偏置，附带accuracy下降约束。"""
    raw=summary(y,scores);options=[]
    for bias in np.arange(-1.,1.51,.25):
        modified=scores.copy();modified[:,1]+=bias;score=summary(y,modified)
        if score["accuracy"]>=raw["accuracy"]-.03:options.append((score["macro_f1"],score["accuracy"],-abs(bias),float(bias),score))
    best=max(options,key=lambda row:row[:3]);return best[3],best[4]


def old_scores(data: dict[str,np.ndarray],previous: Path) -> np.ndarray:
    """相同固定拼接头的开发或历史输出。"""
    manifest=json.loads((previous/"concat/manifest.json").read_text(encoding="utf-8"))
    scaler=FeatureStandardizer();scaler.parameters=manifest["standardizer"]
    x=scaler.transform({name:data[name] for name in ["text","audio","visual"]},data["modality_mask"])
    model=ConcatenationHead(manifest["dimensions"],3,manifest["hidden"])
    model.load_state_dict(torch.load(previous/"concat/weights.pt",map_location="cpu",weights_only=True));model.eval()
    with torch.inference_mode():value,_,_=model({k:torch.tensor(v) for k,v in x.items()},torch.tensor(data["modality_mask"]),torch.tensor(data["quality"]))
    return value.numpy()


def run(bundle: Path,finetune: Path,frozen: Path,encoder: Path,previous: Path,output: Path) -> dict[str,Any]:
    """开发来源分组OOF选择stacker，锁定后才运行历史821个样本。"""
    from transformers import AutoTokenizer
    output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():raise ValueError("结果不可覆盖")
    torch.manual_seed(42);torch.set_num_threads(1)
    valid=load_split(bundle/"calibration.npz")
    progress=json.loads((finetune/"results.json").read_text(encoding="utf-8"))
    names=list(progress["development"])
    values=[]
    for name in names:
        with np.load(finetune/(name+"_validation.npz"),allow_pickle=False) as saved:
            assert np.array_equal(saved["labels"],valid["labels"]);values.append(normalized(saved["logits"]))
    with (frozen/"selected.pkl").open("rb") as handle:svm=pickle.load(handle)
    values.append(normalized(svm["model"].decision_function(svm["scaler"].transform(matrix(valid,svm["config"]["names"])))))
    names.append("frozen_semantic_svm")
    values.append(normalized(old_scores(valid,previous)));names.append("original_concat")
    x=np.column_stack(values);y=valid["labels"]
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(x,y,valid["source_groups"]))
    for tr,va in folds:
        if set(valid["source_groups"][tr]) & set(valid["source_groups"][va]):raise ValueError("开发stacking来源泄漏")
    configs=[{"id":f"stack_{c}_{weight}","kind":"stack","C":c,"weight":weight} for c in [.01,.1,1.,10.] for weight in [None,"balanced"]]
    ranked=sorted(range(len(progress["development"])),key=lambda i:progress["development"][names[i]]["best_macro_f1"],reverse=True)
    configs += [{"id":f"mean_{n}","kind":"mean","indices":ranked[:n]} for n in [1,2,3,5]]
    configs += [{"id":"mean_all","kind":"mean","indices":list(range(len(names)))}]
    protocol={"seed":42,"base_models":names,"configs":configs,"selection":"3-fold source-group out-of-fold meta selection on703 development samples",
        "limitation":"base checkpoints were selected on these703; meta OOF is not an independent end-to-end validation",
        "bias_grid":[float(v) for v in np.arange(-1.,1.51,.25)],"history":"fixed821, not used to fit or select meta model",
        "code_sha256":sha256_file(Path(__file__)),"finetune_results_sha256":sha256_file(finetune/"results.json")}
    save_json(output/"preregister.json",protocol)
    results={};candidates={}
    for config in configs:
        if config["kind"]=="stack":
            scores=np.zeros((len(y),3))
            for tr,va in folds:
                model=LogisticRegression(C=config["C"],class_weight=config["weight"],max_iter=500,random_state=42)
                model.fit(x[tr],y[tr]);scores[va]=model.decision_function(x[va])
        else:scores=np.mean([values[i] for i in config["indices"]],axis=0)
        bias,metric=select_bias(y,scores);results[config["id"]]={"selection":metric,"bias":bias};candidates[config["id"]]=config
    winner=max(results,key=lambda name:(results[name]["selection"]["macro_f1"],results[name]["selection"]["accuracy"]))
    save_json(output/"selection_locked.json",{"winner":winner,"development":results})
    config=candidates[winner];meta=None
    if config["kind"]=="stack":
        meta=LogisticRegression(C=config["C"],class_weight=config["weight"],max_iter=500,random_state=42).fit(x,y)
    test=load_split(bundle/"test.npz");tokenizer=AutoTokenizer.from_pretrained(encoder,local_files_only=True)
    scaler=FeatureStandardizer();scaler.parameters=json.loads((finetune/"side_standardizer.json").read_text(encoding="utf-8"))
    dataset=Rows(test,tokenizer,scaler);test_values=[]
    for name in names[:len(progress["development"])]:
        model=MultimodalBERT(encoder).cuda()
        model.load_state_dict(torch.load(finetune/(name+".pt"),map_location="cpu",weights_only=True))
        scores=logits_for(model,DataLoader(dataset,batch_size=64,collate_fn=collate));test_values.append(normalized(scores))
        np.savez_compressed(output/(name+"_historical.npz"),logits=scores,labels=test["labels"],sample_ids=test["sample_ids"])
        del model;gc.collect();torch.cuda.empty_cache()
    test_values.append(normalized(svm["model"].decision_function(svm["scaler"].transform(matrix(test,svm["config"]["names"])))))
    test_values.append(normalized(old_scores(test,previous)))
    scores=meta.decision_function(np.column_stack(test_values)) if meta is not None else np.mean([test_values[i] for i in config["indices"]],axis=0)
    scores[:,1]+=results[winner]["bias"];historical=summary(test["labels"],scores)
    with np.load(previous/"concat/test_predictions.npz",allow_pickle=False) as old:baseline=old["logits"].argmax(1)
    ci=grouped_delta_ci(test["labels"],scores.argmax(1),baseline,test["source_groups"],3)
    np.savez_compressed(output/"historical_predictions.npz",scores=scores,labels=test["labels"],sample_ids=test["sample_ids"])
    with (output/"selected.pkl").open("wb") as handle:pickle.dump({"config":config,"model":meta,"bias":results[winner]["bias"],"base_models":names},handle)
    report={"protocol":protocol,"winner":winner,"development":results,"historical":historical,"relative_f1_improvement":historical["macro_f1"]/.5657375344573995-1,
        "macro_f1_delta_ci95":ci,"deployed":False}
    save_json(output/"results.json",report);print(json.dumps({"winner":winner,"historical":historical,"relative":report["relative_f1_improvement"],"ci":ci}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["bundle","finetune","frozen","encoder","previous","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.finetune,args.frozen,args.encoder,args.previous,args.output)
