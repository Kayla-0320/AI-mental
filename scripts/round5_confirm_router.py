"""在第二份未参与开发的官方保留集上确认可用性路由；不再改选。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"algorithm"))
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from round5_prepare_external import read_xml,normalize
from experiments.round5_finetune import external_data,Rows,collate,MultimodalBERT,logits_for
from experiments.round5_availability_router import route_scores
from experiments.round5_ensemble import old_scores
from experiments.round5_frozen_search import summary
from experiments.round2_moe_train import grouped_delta_ci
from perception.knowledge_moe.features import FrozenCLIPEncoder,FeatureStandardizer,save_json,sha256_file


def run(root: Path,archive: Path,bundle: Path,encoders: Path,previous: Path) -> dict:
    """先绑定路由协议，再读取新保留集；所有缺媒体输入必须走fallback。"""
    output=root/"router_confirmation";output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():raise ValueError("确认集不可用于重复开发")
    router=json.loads((root/"availability_router/results.json").read_text(encoding="utf-8"))
    protocol={"router_lock_sha256":sha256_file(root/"availability_router/selection_locked.json"),"minimum_face_coverage":router["minimum_face_coverage"],
        "source":"official SMP2019_ECISA_Test_答案.xml; never used in previous selection or fitting",
        "construct":"text implicit sentiment; not unseen multimodal films or clinical validation","code_sha256":sha256_file(Path(__file__)),
        "no_further_model_selection":True}
    save_json(output/"preregister.json",protocol)
    rows=read_xml(archive/"SMP2019_ECISA_Test_答案.xml","fresh_confirmation")
    seen=set()
    for source in [root/"external/train.jsonl",root/"external/external_valid.jsonl",root/"fresh_holdout/holdout.jsonl"]:
        seen.update(normalize(json.loads(line)["text"]) for line in source.read_text(encoding="utf-8").splitlines())
    for split in ["train","selection","calibration","test"]:
        with np.load(bundle/(split+".npz"),allow_pickle=False) as saved:seen.update(normalize(str(t)) for t in saved["raw_text"])
    grams={t[i:i+12] for t in seen for i in range(max(0,len(t)-11))};kept=[];removed=0
    for row in rows:
        text=normalize(row["text"])
        if not text or text in seen or any(text[i:i+12] in grams for i in range(max(0,len(text)-11))):removed+=1;continue
        seen.add(text);kept.append(row)
    source=output/"holdout.jsonl";source.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in kept)+'\n',encoding="utf-8")
    save_json(output/"data_manifest.json",{"raw_n":len(rows),"retained":len(kept),"removed_for_overlap":removed,"class_counts":dict(Counter(row["label"] for row in kept)),
        "prepared_sha256":sha256_file(source),"original_sha256":sha256_file(archive/"SMP2019_ECISA_Test_答案.xml")})
    data=external_data(source);n=len(kept);data["face_summary"]=np.zeros((n,29),np.float32);data["source_groups"]=np.array([row["group"] for row in kept])
    torch.set_num_threads(1)
    clip=FrozenCLIPEncoder(encoders/"chinese-clip-vit-base-patch16","cuda")
    texts=data["raw_text"].astype(str).tolist();data["text"]=np.concatenate([clip.encode_text(texts[i:i+32]) for i in range(0,n,32)])
    reference=old_scores(data,previous)
    del clip;torch.cuda.empty_cache()
    ft=json.loads((root/"finetune/results.json").read_text(encoding="utf-8"));name=ft["winner"]
    tokenizer=AutoTokenizer.from_pretrained(encoders/"chinese-roberta-wwm-ext",local_files_only=True)
    scaler=FeatureStandardizer();scaler.parameters=json.loads((root/"finetune/side_standardizer.json").read_text(encoding="utf-8"))
    dataset=Rows(data,tokenizer,scaler);model=MultimodalBERT(encoders/"chinese-roberta-wwm-ext").cuda()
    model.load_state_dict(torch.load(root/"finetune"/(name+".pt"),map_location="cpu",weights_only=True))
    fallback=logits_for(model,DataLoader(dataset,batch_size=64,collate_fn=collate));fallback[:,1]+=ft["development"][name]["bias"]
    scores,mask=route_scores(data,np.full((n,3),np.nan),fallback,router["minimum_face_coverage"])
    if mask.any() or not np.isfinite(scores).all():raise ValueError("无媒体样本未正确回退")
    metrics={"original_concat_text_only":summary(data["labels"],reference),"availability_router_text_only":summary(data["labels"],scores)}
    ci=grouped_delta_ci(data["labels"],scores.argmax(1),reference.argmax(1),data["source_groups"],3)
    np.savez_compressed(output/"predictions.npz",reference=reference,scores=scores,labels=data["labels"],sample_ids=data["sample_ids"],used_face=mask)
    report={"protocol":protocol,"n":n,"metrics":metrics,"delta_macro_f1_ci95":ci,"fallback_count":n,
        "no_reselection_after_results":True,"synthetic":False,"same_auxiliary_collection_family":True,"human_participants_recruited":0}
    save_json(output/"results.json",report);print(json.dumps(report,ensure_ascii=False),flush=True);return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["root","archive","bundle","encoders","previous"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.root,args.archive,args.bundle,args.encoders,args.previous)
