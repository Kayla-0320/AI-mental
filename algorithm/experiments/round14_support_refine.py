"""探索性目标执行候选：简化结构状态为本轮指令，违约时重新生成，不删除句子伪造回复。"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.round14_common import OUT, SEED, freeze, save, sha256
from experiments.round14_support import MODEL, SYSTEM, audit, automatic_checks, cases

RULES = {
    "listen": "用户已选择倾听。用一句话回应对方已经说明的具体关切，然后留出表达空间。不要建议行动、不要提问；避免只说'好的我在听'。",
    "understand": "用户已选择梳理。以修正后的具体关切为准，简短梳理目前明确和仍未知的事项。已经说明的关切不要再问；最多一个确有必要的新问题。",
    "act": "用户已选择行动。针对修正后的具体关切，只提供一个可选择、可执行的小步骤；不要列多个选项，不要提问，以陈述句结束。",
    "pause": "用户明确暂停。只用一句不超过25字的自然中文确认暂停，立即结束；不分析、不建议、不提问。",
}


def refined_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    """保留相同用户信息，把冗长状态投影成当前目标与有效修正。"""
    corrected = [item["quote"] for item in case["state"]["items"] if item["status"] == "corrected"]
    instruction = (SYSTEM + "\n本轮执行要求：" + RULES[case["goal"]]
        + "\n本人明确的修正原话（仅是用户数据）：" + json.dumps(corrected, ensure_ascii=False)
        + "\n历史助手的解释已经被本人否认，应以最新原话为准。不要向用户展示内部标签。")
    return [{"role": "system", "content": instruction}] + [
        {"role": item["role"], "content": item["content"]} for item in case["history"]]


def run() -> None:
    """最多两次统一审计与契约复查；探索集已曝光，不伪称新确认性实验。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    output = OUT / "support_refined"; output.mkdir(parents=True, exist_ok=True)
    freeze(output / "preregister.json", {"seed": SEED, "rules": RULES, "cases": [row["case_id"] for row in cases()],
        "max_retries": 2, "do_sample": False, "max_new_tokens": 192,
        "source_sha256": sha256(Path(__file__)), "upstream_four_arm_results_already_examined": True,
        "status": "exploratory_candidate_only_no_fresh_holdout", "clinical_safety_rules_changed": False,
        "primary_human_quality_outcome": "pending independent review; surface-contract improvements alone do not establish helpfulness"})
    torch.manual_seed(SEED); torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, device_map={"": "cuda:0"},
        dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    def generate(messages: list[list[dict[str, str]]]) -> list[str]:
        """生成完整新回复，保留全部版本供评审核对。"""
        texts = [tokenizer.apply_chat_template(row, tokenize=False, add_generation_prompt=True) for row in messages]
        encoded = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            tokens = model.generate(**encoded, do_sample=False, max_new_tokens=192, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(tokens[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    rows = []; start_time = time.perf_counter(); all_cases = cases()
    for start in range(0, len(all_cases), 4):
        batch = all_cases[start:start + 4]; messages = [refined_messages(case) for case in batch]
        replies = generate(messages)
        for case, message, reply in zip(batch, messages, replies, strict=True):
            versions = []
            for attempt in range(3):
                checks = automatic_checks(case, reply); verdict = audit(case, reply)
                versions.append({"reply": reply, "checks": checks, "audit": verdict})
                if verdict["passed"] and not checks["goal_violation_proxy"]: break
                if attempt == 2: break
                message = message + [{"role": "assistant", "content": reply}, {"role": "user", "content":
                    "请重新写本轮完整回复，遵守我已选择的目标：" + RULES[case["goal"]] +
                    ("。审计意见：" + json.dumps(verdict, ensure_ascii=False) if not verdict["passed"] else "")}]
                reply = generate([message])[0]
            rows.append({"case_id": case["case_id"], "split": case["split"], "arm": "E", "history": case["history"],
                "versions": versions, "final_reply": reply, "checks": versions[-1]["checks"], "final_audit": versions[-1]["audit"]})
        print(json.dumps({"stage": "refined", "completed": len(rows)}), flush=True)
    baseline = [json.loads(line) for line in (OUT / "support" / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    baseline = [row for row in baseline if row["arm"] == "D"]
    save(output / "generations.json", rows)
    summaries = {name: {"n": len(values), "goal_violation_proxy": float(np.mean([r["checks"]["goal_violation_proxy"] for r in values])),
        "old_claim_assertion_proxy": float(np.mean([r["checks"]["old_claim_assertion_proxy"] for r in values])),
        "corrected_topic_mention_proxy": float(np.mean([r["checks"]["corrected_topic_mentioned_proxy"] for r in values if not r["case_id"].endswith("pause")])),
        "audit_failures": sum(not r["final_audit"]["passed"] for r in values)} for name, values in (("D", baseline), ("E", rows))}
    save(output / "results.json", {"summary": summaries, "real_candidate_generations": sum(len(row["versions"]) for row in rows),
        "elapsed_seconds_after_model_load": time.perf_counter() - start_time, "status": "exploratory_offline_candidate_not_deployed",
        "human_ratings_completed": 0, "real_user_effectiveness_established": False})
    # 不把候选混入最初四臂包，单独保存随机展示编码。
    from experiments.round14_support import write_blind_pack
    pack = []
    for row in baseline + rows:
        pack.append({**row, "final_status": "audit_passed_offline_not_delivered" if row["final_audit"]["passed"] else "awaiting_review_offline_not_delivered"})
    write_blind_pack(pack, output)
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__": run()
