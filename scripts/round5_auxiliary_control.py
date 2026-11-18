"""同架构文字模型的无辅助数据对照，不按外部结果重新选型。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"algorithm"))
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from experiments.round5_finetune import external_data,Rows,collate,MultimodalBERT,logits_for
from experiments.round5_frozen_search import summary
from experiments.round2_moe_train import grouped_delta_ci
from perception.knowledge_moe.features import FeatureStandardizer,save_json,sha256_file


def run(root: Path,encoder: Path) -> dict:
    """训练配方匹配的对照和开发选中的最强target-only对照均保留。"""
    output=root/"auxiliary_control";output.mkdir(exist_ok=True)
    progress=json.loads((root/"finetune/results.json").read_text(encoding="utf-8"))
    target_only=[name for name in progress["development"] if name.startswith("target_")]
    best=max(target_only,key=lambda name:progress["development"][name]["best_macro_f1"])
    names=sorted(set(["target_1e-05_0.5",best]))
    save_json(output/"preregister.json",{"controls":names,"choice":"same LR/weight as auxiliary winner plus highest development target-only candidate",
        "external_results_not_used_to_choose_control":True,"code_sha256":sha256_file(Path(__file__))})
    data=external_data(root/"router_confirmation/holdout.jsonl")
    groups=np.array([json.loads(line)["group"] for line in (root/"router_confirmation/holdout.jsonl").read_text(encoding="utf-8").splitlines()])
    scaler=FeatureStandardizer();scaler.parameters=json.loads((root/"finetune/side_standardizer.json").read_text(encoding="utf-8"))
    tokenizer=AutoTokenizer.from_pretrained(encoder,local_files_only=True);rows=Rows(data,tokenizer,scaler)
    with np.load(root/"router_confirmation/predictions.npz",allow_pickle=False) as saved:augmented=saved["scores"].copy()
    results={};comparisons={};torch.set_num_threads(1)
    for name in names:
        model=MultimodalBERT(encoder).cuda();model.load_state_dict(torch.load(root/"finetune"/(name+".pt"),map_location="cpu",weights_only=True))
        scores=logits_for(model,DataLoader(rows,batch_size=64,collate_fn=collate));scores[:,1]+=progress["development"][name]["bias"]
        results[name]=summary(data["labels"],scores)
        comparisons["auxiliary_minus_"+name]={"delta_macro_f1":summary(data["labels"],augmented)["macro_f1"]-results[name]["macro_f1"],
            "ci95":grouped_delta_ci(data["labels"],augmented.argmax(1),scores.argmax(1),groups,3)}
        np.savez_compressed(output/(name+".npz"),scores=scores,labels=data["labels"],sample_ids=data["sample_ids"])
        del model;torch.cuda.empty_cache()
    report={"n":len(data["labels"]),"target_only":results,"auxiliary":summary(data["labels"],augmented),"comparisons":comparisons,"no_model_reselection":True}
    save_json(output/"results.json",report);print(json.dumps(report),flush=True);return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--encoder",type=Path,required=True)
    args=parser.parse_args();run(args.root,args.encoder)
