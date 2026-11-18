"""明确支持目标、否定及引述的人工编写契约场景；不充当真人效果评价。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))

# 在实现修改之前冻结；development用于定位，challenge用于回归，不声称独立人评。
CASES=[
    ("development","我今天考试失利，只想被理解。","understand","reflect_check"),
    ("development","我今天有点难过，帮我分析一下原因。","understand","reflect_check"),
    ("development","我今天考试失利，只想说说，不过现在想知道怎么做。","act","collaborate_action"),
    ("development","今天考试失利，先听我说，再一起想办法。","listen","listen"),
    ("development","今天考试失利，不要帮我想办法，帮我梳理一下。","understand","reflect_check"),
    ("development","今天朋友说“帮我想办法”，我只是转述。","unspecified","listen"),
    ("development","今天朋友说他想知道怎么做。","unspecified","listen"),
    ("development","今天考试失利，我不是不想说，只是不知道从哪说。","unspecified","listen"),
    ("development","今天朋友说“先不聊了”，我还想理清这件事。","understand","reflect_check"),
    ("development","今天考试失利，我不想休息，想知道怎么做。","act","collaborate_action"),
    ("development","今天考试失利，先暂停一下。","unspecified","respect_pause"),
    ("development","今天考试失利，我想理清这件事。","understand","reflect_check"),
    ("development","今天考试失利，帮我想个办法。","act","collaborate_action"),
    ("development","今天考试失利，不用给建议。","listen","listen"),
    ("development","今天考试失利，我不想一起想办法。","unspecified","listen"),
    ("development","今天考试失利，想知道怎么做，但先不聊了。","act","respect_pause"),
    ("challenge","朋友今天没回消息，我希望你理解我。","understand","reflect_check"),
    ("challenge","朋友今天没回消息，我想理解自己为什么这么在意。","understand","reflect_check"),
    ("challenge","朋友今天没回消息，我希望你陪我梳理。","understand","reflect_check"),
    ("challenge","朋友今天没回消息，我想做点什么。","act","collaborate_action"),
    ("challenge","朋友今天没回消息，给我一个具体建议。","act","collaborate_action"),
    ("challenge","朋友今天没回消息，帮我出个主意。","act","collaborate_action"),
    ("challenge","朋友今天没回消息，我只想倾诉。","listen","listen"),
    ("challenge","朋友今天没回消息，我只想被倾听。","listen","listen"),
    ("challenge","朋友今天没回消息，我想理清，不过现在先听我说。","listen","listen"),
    ("challenge","朋友今天没回消息，先听我说，但是现在帮我理清。","understand","reflect_check"),
    ("challenge","朋友今天没回消息，先听我说，可是现在一起想办法。","act","collaborate_action"),
    ("challenge","朋友今天没回消息，想先理清，再帮我想个办法。","understand","reflect_check"),
    ("challenge","朋友今天没回消息，先只想说说，之后再想知道怎么做。","listen","listen"),
    ("challenge","朋友今天没回消息，别帮我分析，只想说说。","listen","listen"),
    ("challenge","朋友今天没回消息，我不想被理解。","unspecified","listen"),
    ("challenge","朋友今天没回消息，我不想知道怎么做。","unspecified","listen"),
    ("challenge","朋友今天没回消息，不用帮我梳理。","unspecified","listen"),
    ("challenge","朋友今天没回消息，我不是不想说。","unspecified","listen"),
    ("challenge","朋友今天没回消息，我没有说想休息。","unspecified","listen"),
    ("challenge","朋友今天没回消息，我不想休息。","unspecified","listen"),
    ("challenge","朋友今天说『我想休息』，我想理清我们的误会。","understand","reflect_check"),
    ("challenge","朋友今天说“先听我说”，我想知道怎么做。","act","collaborate_action"),
    ("challenge","老师今天说他想弄明白，我只是记录这句话。","unspecified","listen"),
    ("challenge","同学今天说想休息，我还想一起梳理。","understand","reflect_check"),
    ("challenge","朋友今天说他想知道怎么做，但我只想说说。","listen","listen"),
    ("challenge","朋友今天没回消息，想知道怎么做，但让我安静一会。","act","respect_pause"),
    ("challenge","朋友今天没回消息，今天先到这里。","unspecified","respect_pause"),
    ("challenge","朋友今天没回消息，不要再问。","unspecified","respect_pause"),
    ("challenge","朋友今天没回消息，先不聊了。","unspecified","respect_pause"),
    ("challenge","朋友今天没回消息，今天想休息一下。","unspecified","respect_pause"),
    ("challenge","朋友今天没回消息，我想弄明白自己在担心什么。","understand","reflect_check"),
    ("challenge","朋友今天没回消息，希望你先听我说。","listen","listen"),
]


def load_module(path: Path, name: str) -> Any:
    """加载已保存的实现，不修改模块或输入状态。"""
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader:raise ValueError("模块无法加载")
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded


def run(baseline: Path, current: Path, output: Path) -> dict[str,Any]:
    """同一组场景对比真实旧实现和当前实现，公开逐项失败。"""
    output.mkdir(parents=True,exist_ok=True)
    fixture_bytes=json.dumps(CASES,ensure_ascii=False).encode()
    protocol={"cases_sha256":hashlib.sha256(fixture_bytes).hexdigest(),"cases":CASES,
        "origin":"engineer-authored synthetic contract scenarios; no independent raters or participants",
        "metrics":"exact goal and plan agreement with authored expectations; not dialogue helpfulness",
        "baseline_sha256":hashlib.sha256(baseline.read_bytes()).hexdigest(),"prioritization":"current direct preference; quoted/third-party/negated/future intent is not immediate preference"}
    p=output/"preregister.json"
    if p.exists() and json.loads(p.read_text(encoding="utf-8"))!=json.loads(json.dumps(protocol)):raise ValueError("场景已冻结，不能改答案迎合实现")
    p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    results={}
    for name,path in [("baseline",baseline),("current",current)]:
        module=load_module(path,"support_"+name)
        rows=[]
        for split,message,goal,plan in CASES:
            state=module.prepare_support("contract-session","message-1",message,[])
            rows.append({"split":split,"message":message,"expected_goal":goal,"expected_plan":plan,"goal":state.goal,"plan":state.plan.action,
                         "goal_correct":state.goal==goal,"plan_correct":state.plan.action==plan})
        results[name]={"rows":rows,"summary":{split:{"n":len(selected),"goal_agreement":sum(r["goal_correct"] for r in selected)/len(selected),
            "plan_agreement":sum(r["plan_correct"] for r in selected)/len(selected),"both_correct":sum(r["goal_correct"] and r["plan_correct"] for r in selected)}
            for split in ["development","challenge"] if (selected:=[r for r in rows if r["split"]==split])}}
    report={"protocol":protocol,"results":results,"human_participants":0,"independent_raters":0,"efficacy_evidence":False,
            "current_sha256":hashlib.sha256(current.read_bytes()).hexdigest()}
    (output/"support_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({key:value["summary"] for key,value in results.items()},ensure_ascii=False))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["baseline","current","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.baseline,args.current,args.output)
