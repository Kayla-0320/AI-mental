"""本地冻结语言模型的情感任务表征；不使用少样本答案或目标标签。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from experiments.round10_fusion import ROOT, OUTPUT, load_npz, save
from perception.knowledge_moe.features import sha256_file

MODEL = Path(r"D:\ai\models\Qwen2.5-3B-Instruct-nf4")


def main() -> None:
    """同一固定提示对所有样本单独编码，无目标样本检索或跨样本上下文。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "selection"])
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    model_path = Path(r"D:\ai\models\Qwen2.5-7B-Instruct") if args.large else MODEL
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)} if args.large else {}
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        device_map={"": "cuda:0"}, dtype=torch.bfloat16, attn_implementation="sdpa", **quantization).eval()
    tokens = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ("负向", "中性", "正向")]
    if len(set(tokens)) != 3:
        raise ValueError("类别首token不唯一")
    output = OUTPUT / ("llm_large_features" if args.large else "llm_features")
    output.mkdir(parents=True, exist_ok=True)
    system = "你是中文情感分析助手。根据句子的整体表达区分负向、中性和正向。"
    for split in args.splits:
        if (output / f"{split}.npz").exists():
            raise ValueError("拒绝覆盖冻结语言表征")
        data = load_npz(ROOT / "round5" / "text_features" / f"{split}.npz")
        prompts = [tokenizer.apply_chat_template([
            {"role": "system", "content": system},
            {"role": "user", "content": f"句子：{text}\n只回答“负向”“中性”或“正向”。"},
        ], tokenize=False, add_generation_prompt=True) for text in data["raw_text"].astype(str)]
        last, means, scores = [], [], []
        with torch.inference_mode():
            head = model.get_output_embeddings().weight[tokens]
            batch = 8 if args.large else 24
            for start in range(0, len(prompts), batch):
                row = tokenizer(prompts[start:start+batch], truncation=True, max_length=192, padding=True, return_tensors="pt").to("cuda")
                encoded = model.model(**row, use_cache=False).last_hidden_state
                final = encoded[:, -1]
                mask = row["attention_mask"].unsqueeze(-1)
                last.append(final.float().cpu().numpy())
                means.append(((encoded.float()*mask).sum(1)/mask.sum(1)).cpu().numpy())
                scores.append(torch.nn.functional.linear(final, head).float().cpu().numpy())
                if (start//batch+1) % 20 == 0:
                    print(json.dumps({"stage": output.name, "split": split, "encoded": min(len(prompts), start+batch)}), flush=True)
        np.savez_compressed(output / f"{split}.npz", llm_last=np.concatenate(last), llm_mean=np.concatenate(means),
                            llm_scores=np.concatenate(scores), sample_ids=data["sample_ids"])
    files = sorted([*model_path.glob("*.safetensors"), *model_path.glob("pytorch_model*.bin"), model_path/"config.json"])
    save(output / "manifest.json", {"model": str(model_path), "fingerprints": {p.name: sha256_file(p) for p in files},
         "target_training": False, "few_shot_examples": 0, "system_prompt": system,
         "runtime_quantization": "NF4,double_quant,bfloat16" if args.large else "local NF4 configuration",
         "user_template": "句子：{text}\\n只回答“负向”“中性”或“正向”。", "label_first_token_ids": tokens,
         "last_representation": "assistant generation prefix, before any answer token", "upstream_target_overlap_verified": False})


if __name__ == "__main__":
    main()
