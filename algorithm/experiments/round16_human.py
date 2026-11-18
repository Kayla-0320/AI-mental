"""真人语义评分的独立采集与分歧复核；不调用模型给分，不填充人类判断。"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import shutil
from typing import Any
from experiments.round14_common import ROOT,save
from experiments.round14_human import read_csv,analyze_ratings,number
from experiments.round15_repair import extra_checks
from experiments.round15_support import new_cases

OUT=ROOT/"round16/human"
EXTRA=["action_matches_concern","reasks_known_information","unfounded_interpretation","evidence_user_quote","evidence_reply_quote","semantic_notes"]


def prepare() -> None:
    """复制已有F/G盲评并添加空白语义字段，机器排查列表与人评目录隔离。"""
    origin=ROOT/"round15/support_repair"
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in ("blind","private"):
        (OUT/folder).mkdir(exist_ok=True)
    for name in ("replies.json","rater_declarations.template.json"):
        shutil.copy2(origin/"blind"/name,OUT/"blind"/name)
    shutil.copy2(origin/"private/arm_key.json",OUT/"private/arm_key.json")
    for rater in ("rater_1","rater_2"):
        path=OUT/"blind"/(rater+".csv")
        if not path.exists():
            rows=read_csv(origin/"blind"/(rater+".csv"));fields=list(rows[0])+EXTRA
            with path.open("w",encoding="utf-8-sig",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    instructions="""# 独立真人语义评审

本包80条F/G匹配回复，场景为工程编写，不能代表真人效果。两名评审应独立于开发者/场景作者，先分别完成全部条目，完成前不讨论。不得以模型评分代填，也不得只评价机器标记的失败条目。

保留原有六项必填评分。新增action_matches_concern：给出的行动是否针对用户当前修正后的关切，1是、0否、NA无行动；reasks_known_information：是否重复询问已经明确的信息；unfounded_interpretation：是否把未表达的原因/动机/关联作为事实，后二者1是0否。暂停后追加任务也属于目标违约，不能因为没有问号就判合格。

填写逐字evidence_user_quote及evidence_reply_quote，后者必须在当前回复中。把推测说成事实、把已否认解释重新采用、要求再次确认已经说明的关切，均结合完整上下文判断。不因出现关键词或字数短自动给高分。NA不是得分0；空白不是已评。

完成后由评审本人填写rater_declarations.json的独立人评声明与时间。软件只能校验声明，不能证明真实身份或独立性。分析先保持原始独立评分，再输出分歧清单供第三位真实评审/讨论记录复核；不得用裁决替换原始一致性结果。

只向评审提供blind目录，不提供private条件映射、旧实验结果或机器问题清单。当前没有真实已完成评分。
"""
    (OUT/"blind/评分指南.md").write_text(instructions,encoding="utf-8")
    cases={c["case_id"]:c for c in new_cases()}
    rows=json.loads((origin/"generations.json").read_text(encoding="utf-8"));triage=[]
    for row in rows:
        flags={**extra_checks(cases[row["case_id"]],row["final_reply"]),"goal_surface_proxy":row["checks"]["goal_violation_proxy"]}
        if any(flags.values()):triage.append({"case_id":row["case_id"],"reply":row["final_reply"],"history":row["history"],"machine_flags_not_human_scores":flags})
    save(OUT/"private/machine_triage.json",triage)


def analyze(directory: Path=OUT) -> dict[str,Any]:
    """只接受带独立真人声明的完整语义字段，校验引用出处并生成空裁决表。"""
    base=analyze_ratings(directory/"blind",directory/"private/arm_key.json",directory/"base_ratings_analysis.json")
    public={r["display_id"]:r for r in json.loads((directory/"blind/replies.json").read_text(encoding="utf-8"))}
    completed=[]
    for path in sorted((directory/"blind").glob("rater_*.csv")):
        for row in read_csv(path):
            semantic_present=any(row.get(k,"").strip() for k in EXTRA)
            base_present=bool(row.get("fidelity_1_5","").strip())
            if not base_present:
                if semantic_present:raise ValueError("语义判断须与完整真实独立基础评分一同提交")
                continue
            if row.get("action_matches_concern") not in {"0","1","NA"}:raise ValueError("行动适切性须填0/1/NA")
            for key in ("reasks_known_information","unfounded_interpretation"):row[key]=str(int(number(row.get(key,""),0,1,True)))
            item=public[row["display_id"]];reply_quote=row.get("evidence_reply_quote","")
            user_quote=row.get("evidence_user_quote","")
            if not reply_quote or reply_quote not in item["reply"]:raise ValueError("回复证据必须逐字出自当前回复")
            if not user_quote or not any(user_quote in turn["content"] for turn in item["dialogue"] if turn["role"]=="user"):raise ValueError("用户证据必须逐字出自用户原话")
            completed.append(row)
    disagreements=[]
    for display in public:
        ratings=[r for r in completed if r["display_id"]==display]
        if len(ratings)>=2:
            differing=[k for k in EXTRA[:3] if len({r[k] for r in ratings})>1]
            if differing:disagreements.append({"display_id":display,"disputed_fields":";".join(differing),"human_adjudicator":"","final_decision":"","rationale":""})
    with (directory/"adjudication_pending.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["display_id","disputed_fields","human_adjudicator","final_decision","rationale"]);writer.writeheader();writer.writerows(disagreements)
    result={"status":"awaiting_real_independent_semantic_ratings" if not completed else "partial_or_completed_human_data",
        "completed_semantic_rows":len(completed),"rater_count":len({r["rater_id"] for r in completed}),"disagreements":len(disagreements),
        "mean_semantic_quality":None,"model_used_as_rater":False,"base_analysis":base,"clinical_effectiveness_established":False}
    arm_key={r["display_id"]:r["arm"] for r in json.loads((directory/"private/arm_key.json").read_text(encoding="utf-8"))}
    result["semantic_by_arm"]={}
    for arm in sorted(set(arm_key.values())):
        items=[r for r in completed if arm_key[r["display_id"]]==arm]
        actions=[r for r in items if r["action_matches_concern"]!="NA"]
        result["semantic_by_arm"][arm]={"rating_rows":len(items),"action_applicable_rows":len(actions),
            "action_matches_concern_rate":sum(r["action_matches_concern"]=="1" for r in actions)/len(actions) if actions else None,
            "reasks_known_information_rate":sum(r["reasks_known_information"]=="1" for r in items)/len(items) if items else None,
            "unfounded_interpretation_rate":sum(r["unfounded_interpretation"]=="1" for r in items)/len(items) if items else None}
    save(directory/"semantic_analysis.json",result);return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");args=parser.parse_args()
    if args.prepare:prepare()
    print(json.dumps(analyze(),ensure_ascii=False))
