"""冻结语言特征只运行原解码器与三行原输出权重，不分配无用的全词表头。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np


def build(mode: str) -> tuple[Any,Any,Any,dict[str,Any]]:
    """保持全部解码器权重/NF4与精度，拒绝任何缺失或形状失配。"""
    import torch
    from transformers import AutoModel,AutoTokenizer,BitsAndBytesConfig
    from safetensors import safe_open
    path=Path(r"D:\ai\models\Qwen2.5-7B-Instruct" if mode=="7b" else r"D:\ai\models\Qwen2.5-3B-Instruct-nf4")
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);tokenizer.padding_side="left"
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokens=[tokenizer.encode(word,add_special_tokens=False)[0] for word in ("负向","中性","正向")]
    if len(set(tokens))!=3:raise ValueError("类别首token不唯一")
    extra={"quantization_config":BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)} if mode=="7b" else {}
    model,info=AutoModel.from_pretrained(path,local_files_only=True,device_map={"":"cuda:0"},dtype=torch.bfloat16,
        attn_implementation="sdpa",output_loading_info=True,**extra)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError(f"解码器存在未加载权重:{info}")
    unexpected=[key for key in info.get("unexpected_keys",[]) if not key.startswith("lm_head.")]
    if unexpected:raise ValueError(f"除未使用输出头外存在额外权重:{unexpected}")
    if model.config.tie_word_embeddings:
        head=model.get_input_embeddings().weight[tokens].detach().to(dtype=torch.bfloat16)
    else:
        index=json.loads((path/"model.safetensors.index.json").read_text("utf-8")) if (path/"model.safetensors.index.json").exists() else None
        headpath=path/index["weight_map"]["lm_head.weight"] if index else path/"model.safetensors"
        with safe_open(headpath,framework="pt",device="cpu") as source:
            view=source.get_slice("lm_head.weight")
            head=torch.cat([view[token:token+1] for token in tokens]).to(device="cuda",dtype=torch.bfloat16)
    metadata={"mode":mode,"model_class":type(model).__name__,"omitted":"full-vocabulary lm_head only;three exact checkpoint rows retained",
        "label_first_tokens":tokens,"loading_info":info,"decoder_4bit_linear_count":sum(type(m).__name__=="Linear4bit" for m in model.modules()),
        "prompt_and_precision":"unchanged frozen Round10;NF4,bf16,max192,left padding"}
    return model.eval(),tokenizer,head,metadata


def encode(model: Any,tokenizer: Any,head: Any,texts: list[str]) -> dict[str,np.ndarray]:
    """与原model.model前向相同，只省去从不执行的全词表输出头驻留。"""
    import torch
    prompts=[tokenizer.apply_chat_template([{"role":"system","content":"你是中文情感分析助手。根据句子的整体表达区分负向、中性和正向。"},
        {"role":"user","content":f"句子：{text}\n只回答“负向”“中性”或“正向”。"}],tokenize=False,add_generation_prompt=True) for text in texts]
    row=tokenizer(prompts,truncation=True,max_length=192,padding=True,return_tensors="pt").to("cuda")
    with torch.inference_mode():
        hidden=model(**row,use_cache=False).last_hidden_state;last=hidden[:,-1];mask=row["attention_mask"].unsqueeze(-1)
        return {"llm_last":last.float().cpu().numpy(),"llm_mean":((hidden.float()*mask).sum(1)/mask.sum(1)).cpu().numpy(),
            "llm_scores":torch.nn.functional.linear(last,head).float().cpu().numpy()}


def encode_one(text: str,mode: str) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    """当前请求构建模型和提取特征，由隔离子进程负责释放整个上下文。"""
    model,tokenizer,head,metadata=build(mode)
    return encode(model,tokenizer,head,[text]),metadata
