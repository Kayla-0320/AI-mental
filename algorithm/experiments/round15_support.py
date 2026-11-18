"""具体回应与目标兼顾的匹配生成实验；场景均为编写数据，不替代独立人评。"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any
import numpy as np
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round14_support import MODEL,audit,automatic_checks,cases,GOALS,write_blind_pack
from experiments.round14_support_refine import refined_messages
from intervention.support_understanding import prepare_support,apply_feedback
from shared.dataclasses import SupportFeedback

OUT = ROOT/"round15/support"
INSTRUCTION = ("\n非暂停时，回复两句即可：第一句具体概括最新用户修正后的实际关切，明确提及其中的任务、物品或情境，"
    "不要只说好的、明白、理解或我在听，也不要再复述已被否认的解释。第二句执行当前目标："
    "倾听就温和接住这件具体事而不建议；梳理就针对这件事澄清，不复问已知信息；行动就给针对这件事的一个小步骤，"
    "不用征求意见的问题收尾。暂停时直接简短收束。不要照抄用户整句话，不编造原因。")
TOPICS = [
    ("手机没电", "我不是担心错过聊天，我只是担心电子车票打不开", "错过聊天", "车票|乘车"),
    ("图书馆关门", "我不是因为关门生气，我是想把借的书按时还回去", "生气", "还书|归还|借的书"),
    ("明天要出门", "我不是害怕出门，我只是还没找到身份证", "害怕", "身份证|证件"),
    ("打印店排队", "我不是嫌排队麻烦，我担心的是报名表来不及打印", "嫌排队", "报名表|打印"),
    ("钥匙不见了", "我不是怀疑别人拿走，我只是想联系管理员开储物柜", "别人拿走", "管理员|储物柜"),
    ("公交晚点", "我不是担心迟到被批评，我是还没通知一起排练的人", "被批评", "通知|排练"),
    ("电脑出故障", "我不是担心成绩，我只是想找回没有保存的照片", "成绩", "照片|找回"),
    ("快递没到", "我不是生快递员的气，我想弄清楚实验材料什么时候能到", "生气", "实验|材料|到货"),
    ("活动改期", "我不是觉得失望，我只是需要退掉已经订好的房间", "失望", "房间|退订|取消预订"),
    ("课程换教室", "我不是不想上课，我是需要告诉搭档新的集合地点", "不想上课", "搭档|地点|集合"),
]


def new_cases() -> list[dict[str, Any]]:
    """十类未参与旧轮编写场景，冻结语义检查标签但绝不送给生成器。"""
    result=[]
    for i,(event,correction,old,pattern) in enumerate(TOPICS):
        for goal,request in GOALS.items():
            identifier=f"N{i+1:02d}_{goal}"; first=f"今天{event}，有件事让我挂心。"
            history=[{"role":"user","id":"u1","content":first},{"role":"assistant","content":f"你主要在担心{old}。"},
                     {"role":"user","id":"u2","content":correction+"。"+request}]
            state=prepare_support(identifier,"u1",first,[])
            state=apply_feedback(state,SupportFeedback("f1",state.revision,"correct",state.items[0].id,correction))
            state=prepare_support(identifier,"u2",history[-1]["content"],history[:-1],asdict(state))
            result.append({"case_id":identifier,"split":"new_authored_challenge","origin":"engineer_authored_not_real_user",
                "goal":goal,"old_claim":old,"corrected_pattern":pattern,"history":history,"state":asdict(state)})
    return result


def messages(case: dict[str,Any],arm: str) -> list[dict[str,str]]:
    """两臂均只读可观察对话和修正；评分关键词不参与提示或重试。"""
    value=refined_messages(case)
    if arm=="F": value[0]["content"]+=INSTRUCTION
    elif arm!="E": raise ValueError("未知条件")
    return value


def run() -> None:
    """冻结后真实生成；两臂执行相同现有审计与表面目标复查。"""
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    OUT.mkdir(parents=True,exist_ok=True)
    challenge=new_cases(); development=cases()
    freeze(OUT/"protocol.json",{"seed":42,"model":str(MODEL),"instruction":INSTRUCTION,"new_cases":challenge,
        "generation":{"max_new_tokens":192,"do_sample":False,"batch":4,"max_retries":2},
        "selection":"one fixed candidate; old32 development/exposed; new40 authored challenge; not independent humans",
        "code_sha256":sha256(Path(__file__)),"primary":"joint surface goal adherence and corrected-topic mention; semantic appropriateness pending humans"})
    torch.manual_seed(42);torch.set_num_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side="left")
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,device_map={"":"cuda:0"},dtype=torch.bfloat16,attn_implementation="sdpa").eval()
    def generate(batch: list[list[dict[str,str]]]) -> list[str]:
        """不混入其他样本、参考答案或评分关键词。"""
        texts=[tokenizer.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in batch]
        encoded=tokenizer(texts,padding=True,return_tensors="pt").to("cuda")
        with torch.inference_mode():tokens=model.generate(**encoded,do_sample=False,max_new_tokens=192,pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(tokens[:,encoded["input_ids"].shape[1]:],skip_special_tokens=True)
    path=OUT/"generations.jsonl"
    rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    seen={(r["case_id"],r["arm"]) for r in rows}
    work=[(c,a) for c in challenge for a in ("E","F")]+[(dict(c,split="old_exposed_development"),"F") for c in development]
    work=[(c,a) for c,a in work if (c["case_id"],a) not in seen]
    for start in range(0,len(work),4):
        batch=work[start:start+4]; prompts=[messages(c,a) for c,a in batch]; replies=generate(prompts)
        for (case,arm),prompt,reply in zip(batch,prompts,replies,strict=True):
            versions=[]
            for attempt in range(3):
                check=automatic_checks(case,reply);verdict=audit(case,reply)
                versions.append({"reply":reply,"checks":check,"audit":verdict})
                if verdict["passed"] and not check["goal_violation_proxy"] or attempt==2:break
                # 无论E/F都使用同一条复查提醒；不以是否命中关切关键词为重试依据。
                prompt=prompt+[{"role":"assistant","content":reply},{"role":"user","content":"请遵守我本轮已说明的目标，重新写完整回复；不要超出提问和建议范围。"+
                    (json.dumps(verdict,ensure_ascii=False) if not verdict["passed"] else "")}]
                reply=generate([prompt])[0]
            row={"case_id":case["case_id"],"arm":arm,"split":case["split"],"history":case["history"],"versions":versions,
                "final_reply":reply,"checks":check,"final_audit":verdict,"final_status":"audit_passed_offline_not_delivered" if verdict["passed"] else "awaiting_review_offline_not_delivered"}
            rows.append(row)
            with path.open("a",encoding="utf-8") as stream:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        print(json.dumps({"completed":len(rows)}),flush=True)
    results={}
    for split in ("new_authored_challenge","old_exposed_development"):
        results[split]={}
        for arm in ("E","F"):
            subset=[r for r in rows if r["split"]==split and r["arm"]==arm]
            if not subset:continue
            active=[r for r in subset if not r["case_id"].endswith("pause")]
            results[split][arm]={"n":len(subset),"nonpause_n":len(active),"goal_violation_proxy":float(np.mean([r["checks"]["goal_violation_proxy"] for r in subset])),
                "topic_mention_proxy":float(np.mean([r["checks"]["corrected_topic_mentioned_proxy"] for r in active])),
                "joint_goal_and_topic_proxy":float(np.mean([not r["checks"]["goal_violation_proxy"] and r["checks"]["corrected_topic_mentioned_proxy"] for r in active])),
                "old_claim_proxy":float(np.mean([r["checks"]["old_claim_assertion_proxy"] for r in subset])),"audit_failures":sum(not r["final_audit"]["passed"] for r in subset)}
    save(OUT/"results.json",{"summary":results,"generations":sum(len(r["versions"]) for r in rows),"independent_human_ratings":0,"production_changed":False})
    write_blind_pack([r for r in rows if r["split"]=="new_authored_challenge"],OUT)
    print(json.dumps(results,ensure_ascii=False),flush=True)


if __name__=="__main__":run()
