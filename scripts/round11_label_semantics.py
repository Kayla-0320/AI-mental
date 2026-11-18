"""核对外部开发标注的聚合和语义分布，不重标、不删除难例、不查看test目标。"""
from __future__ import annotations

from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import ROOT,OUTPUT,digest,save


def main() -> None:
    """七类类别与连续极性各自保存，不能互相替换为另一任务金标。"""
    path=ROOT/"_国赛优化/round6/data/multimodal_annotations.jsonl"
    table: dict[str,Counter[str]]=defaultdict(Counter)
    sums=defaultdict(float)
    mismatches=[]
    invalid=[]
    count=0
    for line in path.read_text("utf-8").splitlines():
        item=json.loads(line)
        key=item["id"]
        if key.split("_")[0] in ("G00003","G00015"):
            continue
        row=item["annotation"]
        continuous=float(row["Continuous label_result"])
        votes=[float(vote["Continuous_label"]) for vote in row["data"].values()]
        if len(votes)!=5 or any(value not in (-2.,-1.,0.,1.,2.) for value in votes):
            invalid.append(key)
        if abs(sum(votes)/len(votes)-continuous)>1e-7:
            mismatches.append({"id":key,"published":continuous,"vote_average":sum(votes)/len(votes)})
        pole="negative" if continuous<0 else "positive" if continuous>0 else "neutral"
        table[row["emotion_result"]][pole]+=1
        sums[row["emotion_result"]]+=continuous
        count+=1
    report={"development_samples":count,"external_test_targets_used":False,
        "source_sha256":digest(path),"invalid_vote_values_or_counts":invalid,
        "published_mean_mismatches":mismatches,
        "category_vs_signed_polarity":{key:dict(value) for key,value in table.items()},
        "mean_continuous_by_category":{key:sums[key]/sum(value.values()) for key,value in table.items()},
        "official_polarity_definition":"-2 strongnegative,-1 weaknegative,0 neutral,1 weakpositive,2 strongpositive;mean five raters",
        "definition_source":"https://aclanthology.org/2026.findings-acl.440.pdf#page=4",
        "interpretation":"published categorical emotion and signed polarity differ substantially;cause not verified;not proof of coding reversal",
        "decision":"evaluate released continuous sign as preregistered;retain all samples;never remap polarity from categorical emotion",
        "additional_task":"categorical7 perception/forecast reported separately;do not merge metrics with polarity3"}
    save(OUTPUT/"corpus/label_semantics_audit.json",report)
    print(json.dumps({"n":count,"mean_mismatches":len(mismatches),"invalid_votes":len(invalid),
        "happy_polarity":dict(table["happy"])}),flush=True)


if __name__=="__main__":
    main()
