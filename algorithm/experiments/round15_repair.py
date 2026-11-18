"""对具体回应后的重复核对和暂停追加行动进行离线契约修复，保留原版供评审。"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
import numpy as np
from experiments.round14_common import ROOT,save,freeze,sha256
from experiments.round14_support import MODEL,audit,automatic_checks,write_blind_pack
from experiments.round15_support import new_cases,messages
OUT=ROOT/"round15/support_repair"


def extra_checks(case: dict,reply: str) -> dict[str,bool]:
    """扩展表面代理，捕捉已知关切反问和暂停后的行动；仍不等同语义人评。"""
    return {"reconfirm_known_concern_proxy":case["goal"]=="understand" and bool(re.search(r"你.{0,35}(?:关心|担心|想|需要).{0,35}[吗么？?]|对吗|是这样吗",reply)),
        "pause_adds_action_proxy":case["goal"]=="pause" and bool(re.search(r"先.{0,12}(?:联系|检查|尝试|找|写|完成)|(?:联系|检查|回忆).{0,8}吧",reply))}


def run() -> None:
    """只修复可观察契约，不以评测关键词决定重写；暂停使用显式收束模板。"""
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/"round15/support/generations.jsonl"
    baseline=[json.loads(x) for x in source.read_text(encoding="utf-8").splitlines()]
    baseline=[r for r in baseline if r["arm"]=="F" and r["split"]=="new_authored_challenge"]
    cases={c["case_id"]:c for c in new_cases()}
    freeze(OUT/"protocol.json",{"code_sha256":sha256(Path(__file__)),"input_sha256":sha256(source),"status":"exploratory repair after F inspection; no new holdout",
        "pause":"explicit acknowledgement template; not a model-generated efficacy claim","rewrite":"only repeat-known-concern or unchanged audit/goal proxy; no topic label used"})
    torch.manual_seed(42);torch.set_num_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,device_map={"":"cuda:0"},dtype=torch.bfloat16,attn_implementation="sdpa").eval()
    rows=[]
    for original in baseline:
        case=cases[original["case_id"]]; reply=original["final_reply"]; versions=[]
        if case["goal"]=="pause":reply="好的，今天先到这里。"
        else:
            for attempt in range(2):
                if not any(extra_checks(case,reply).values()) and not automatic_checks(case,reply)["goal_violation_proxy"] and audit(case,reply)["passed"]:break
                prompt=messages(case,"F")+[{"role":"assistant","content":reply},{"role":"user","content":
                    "我已经说明当前关切，不需要再问我是不是这个意思。请用陈述句具体回应这件事，梳理已知的事实和下一项仍未知的信息，不要为了凑句子而提问或添加经历。"}]
                text=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
                encoded=tokenizer(text,return_tensors="pt").to("cuda")
                with torch.inference_mode():tokens=model.generate(**encoded,do_sample=False,max_new_tokens=192,pad_token_id=tokenizer.eos_token_id)
                reply=tokenizer.decode(tokens[0,encoded["input_ids"].shape[1]:],skip_special_tokens=True)
                versions.append(reply)
        verdict=audit(case,reply)
        rows.append({**original,"arm":"G","final_reply":reply,"final_audit":verdict,"checks":automatic_checks(case,reply),"extra_checks":extra_checks(case,reply),
            "repair_versions":versions,"original_F":original["final_reply"],"template_pause":case["goal"]=="pause",
            "final_status":"audit_passed_offline_not_delivered" if verdict["passed"] else "awaiting_review_offline_not_delivered"})
    summary={}
    for name,values in (("F",baseline),("G",rows)):
        active=[r for r in values if not r["case_id"].endswith("pause")]
        summary[name]={"n":len(values),"topic_mention_proxy":float(np.mean([r["checks"]["corrected_topic_mentioned_proxy"] for r in active])),
            "goal_violation_proxy":float(np.mean([r["checks"]["goal_violation_proxy"] for r in values])),
            "extra_violation_proxy":sum(any(extra_checks(cases[r["case_id"]],r["final_reply"]).values()) for r in values),"audit_failures":sum(not r["final_audit"]["passed"] for r in values)}
    save(OUT/"generations.json",rows);save(OUT/"results.json",{"summary":summary,"model_rewrites":sum(len(r["repair_versions"]) for r in rows),"template_pauses":10,"human_raters":0})
    write_blind_pack(baseline+rows,OUT)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":run()
