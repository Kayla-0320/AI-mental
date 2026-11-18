"""开放本地7B模型的少样本极性判别及融合；示例只从训练集检索。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from experiments.round2_moe_train import load_split,grouped_delta_ci
from experiments.round4_accuracy import merge
from experiments.round5_frozen_search import summary
from experiments.round5_ensemble import normalized,select_bias
from perception.knowledge_moe.features import save_json,sha256_file

INSTRUCTION=("你是中文情感极性标注员。A代表负向，B代表中性，C代表正向。"
    "中性包括客观叙述、信息询问、不带明确好恶的普通对话。"
    "区分句子主题和说话人表达的态度；注意否定与讽刺，不要因为句中提到积极或消极事件就自动判为相同情感。"
    "参考样例的标签规律，只输出A、B、C中的一个字母。")


def prompts_for(data: dict[str,np.ndarray],training: dict[str,np.ndarray],tokenizer: Any,shots: int) -> list[str]:
    """每类选相同数量的最近训练示例，不检索开发/历史标签。"""
    train_vectors=training["bge_cls"]/np.maximum(np.linalg.norm(training["bge_cls"],axis=1,keepdims=True),1e-8)
    query=data["bge_cls"]/np.maximum(np.linalg.norm(data["bge_cls"],axis=1,keepdims=True),1e-8)
    similarity=query@train_vectors.T if shots else None
    prompts=[]
    for i,text in enumerate(data["raw_text"]):
        examples=[]
        if shots:
            for label in [0,1,2]:
                indices=np.flatnonzero(training["labels"]==label)
                nearest=indices[np.argsort(-similarity[i,indices])[:shots//3]]
                examples.extend(f"台词：{training['raw_text'][j]}\n类别：{'ABC'[label]}" for j in nearest)
        content=("标注示例：\n"+'\n'.join(examples)+'\n\n' if examples else '')+f"待判断台词：{text}\n类别："
        prompts.append(tokenizer.apply_chat_template([{"role":"system","content":INSTRUCTION},{"role":"user","content":content}],tokenize=False,add_generation_prompt=True))
    return prompts


def infer(model: Any,tokenizer: Any,prompts: list[str],cache: Path) -> np.ndarray:
    """只计算标签token分数，缓存绑定完整提示摘要，支持中断续跑。"""
    digest=hashlib.sha256('\n'.join(prompts).encode()).hexdigest()
    values=[]
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            if str(saved["prompt_sha256"])!=digest:raise ValueError("LLM缓存提示不符")
            values=list(saved["scores"])
    label_ids=[tokenizer.encode(label,add_special_tokens=False) for label in ['A','B','C']]
    if any(len(ids)!=1 for ids in label_ids):raise ValueError("标签必须是单token")
    ids=[v[0] for v in label_ids]
    with torch.inference_mode():
        for start in range(len(values),len(prompts),8):
            batch=tokenizer(prompts[start:start+8],padding=True,return_tensors="pt").to("cuda")
            # 不截断提示和标签；预先限制训练示例数量与原文长度。
            hidden=model.model(**batch,use_cache=False).last_hidden_state[:,-1]
            logits=torch.nn.functional.linear(hidden,model.lm_head.weight[ids]).float().cpu().numpy()
            values.extend(logits)
            np.savez_compressed(cache,scores=np.stack(values),prompt_sha256=np.array(digest))
            if (start//8)%10==0:print(json.dumps({"llm_cache":cache.name,"rows":len(values),"total":len(prompts)}),flush=True)
    return np.stack(values)


def fused(q: np.ndarray,base: np.ndarray,config: dict) -> np.ndarray:
    """比较软融合与只替换中性判决的固定门控，不改变样本或真值。"""
    q=normalized(q)
    if config["mode"]=="blend":return config["weight"]*q+(1-config["weight"])*base
    scores=base.copy();gate=q[:,1]-np.maximum(q[:,0],q[:,2])>config["threshold"]
    scores[:,1]=-5.;scores[gate,1]=5.
    return scores


def run(bundle: Path,model_path: Path,finetune: Path,ensemble: Path,previous: Path,output: Path) -> dict:
    """开发选择全部提示和融合后锁定，只评测最终方案的历史留出。"""
    output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():raise ValueError("拒绝覆盖已有结果")
    torch.manual_seed(42);torch.set_num_threads(1)
    training=merge([load_split(bundle/(name+".npz")) for name in ["train","selection"]]);valid=load_split(bundle/"calibration.npz")
    configs=[{"id":f"shots{shots}_blend{weight}","shots":shots,"mode":"blend","weight":weight}
             for shots in [0,3] for weight in [0.,.25,.5,.75,1.]]
    configs += [{"id":f"shots{shots}_gate{threshold}","shots":shots,"mode":"gate","threshold":threshold} for shots in [0,3] for threshold in [-.5,0.,.5,1.]]
    protocol={"seed":42,"instruction":INSTRUCTION,"configs":configs,"retrieval":"BGE cosine, nearest equal-per-class training examples only",
        "base":"uniform normalized logits of five previously locked fine-tuned checkpoints","model":str(model_path),"quantization":"bitsandbytes NF4 double quantization, bfloat16",
        "selection":"703 development; fixed821 history never enters prompt examples, fitting or candidate selection",
        "code_sha256":sha256_file(Path(__file__)),"model_config_sha256":sha256_file(model_path/"config.json"),
        "auxiliary_model_results_sha256":sha256_file(finetune/"results.json"),"limitations":["history is repeatedly examined diagnostics","no claim about unknown LLM pretraining contamination"]}
    save_json(output/"preregister.json",protocol)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True);tokenizer.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,device_map={"":"cuda:0"},dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    development=json.loads((finetune/"results.json").read_text(encoding="utf-8"))["development"]
    base=[]
    for name in development:
        with np.load(finetune/(name+"_validation.npz"),allow_pickle=False) as saved:base.append(normalized(saved["logits"]))
    base=np.mean(base,axis=0)
    llm={shots:infer(model,tokenizer,prompts_for(valid,training,tokenizer,shots),output/(f"dev_shots{shots}.npz")) for shots in [0,3]}
    results={}
    for config in configs:
        scores=fused(llm[config["shots"]],base,config);bias,metric=select_bias(valid["labels"],scores)
        results[config["id"]]={"validation":metric,"bias":bias}
    winner=max(results,key=lambda name:(results[name]["validation"]["macro_f1"],results[name]["validation"]["accuracy"]))
    config=next(c for c in configs if c["id"]==winner)
    save_json(output/"selection_locked.json",{"winner":winner,"development":results})
    test=load_split(bundle/"test.npz")
    llm_test=infer(model,tokenizer,prompts_for(test,training,tokenizer,config["shots"]),output/"historical_llm.npz")
    base=[]
    for name in development:
        with np.load(ensemble/(name+"_historical.npz"),allow_pickle=False) as saved:base.append(normalized(saved["logits"]))
    scores=fused(llm_test,np.mean(base,axis=0),config);scores[:,1]+=results[winner]["bias"]
    historical=summary(test["labels"],scores)
    with np.load(previous/"concat/test_predictions.npz",allow_pickle=False) as old:reference=old["logits"].argmax(1)
    ci=grouped_delta_ci(test["labels"],scores.argmax(1),reference,test["source_groups"],3)
    np.savez_compressed(output/"historical_predictions.npz",scores=scores,labels=test["labels"],sample_ids=test["sample_ids"])
    report={"protocol":protocol,"winner":winner,"development":results,"historical":historical,"relative_f1_improvement":historical["macro_f1"]/.5657375344573995-1,
        "macro_f1_delta_ci95":ci,"deployed":False}
    save_json(output/"results.json",report);print(json.dumps({"winner":winner,"historical":historical,"relative":report["relative_f1_improvement"],"ci":ci}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["bundle","model-path","finetune","ensemble","previous","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.model_path,args.finetune,args.ensemble,args.previous,args.output)
