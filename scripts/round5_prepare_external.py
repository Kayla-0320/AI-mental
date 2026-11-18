"""准备官方公开ECISA文本，去重并隔离历史测试相似文本；保留外部验证不参与选型。"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np


def normalize(text: str) -> str:
    """只用于重合筛查；训练文本保留原始内容。"""
    return re.sub(r"[^\w\u4e00-\u9fff]","",text.casefold())


def read_xml(path: Path, split: str) -> list[dict]:
    """仅提取明确标注的句子，缺标签句子不能猜成中性。"""
    root=ET.fromstring(path.read_bytes());rows=[]
    mapping={"0":1,"1":2,"2":0}
    for doc in root.iter("Doc"):
        for sentence in doc.iter("Sentence"):
            label=sentence.get("label")
            if label not in mapping:continue
            text="".join(sentence.itertext()).strip()
            if not text:continue
            rows.append({"id":f"ecisa/{split}/{doc.get('ID')}/{sentence.get('ID')}","group":f"ecisa/{split}/{doc.get('ID')}",
                "text":text,"label":mapping[label],"source":"SMP2019_ECISA","source_label":label,"split":split,
                "label_kind":"human_implicit_sentiment_not_personal_mental_health"})
    return rows


def run(source: Path,bundle: Path,output: Path) -> dict:
    """冻结外部训练和验证，排除与CH-SIMS留出原文相同或长子串重合的样本。"""
    output.mkdir(parents=True,exist_ok=True)
    heldout=set();all_target=set()
    for split in ["train","selection","calibration","test"]:
        with np.load(bundle/(split+".npz"),allow_pickle=False) as saved:
            texts={normalize(str(t)) for t in saved["raw_text"]}
            all_target|=texts
            if split in ["calibration","test"]:heldout|=texts
    grams={text[i:i+8] for text in heldout for i in range(max(0,len(text)-7))}
    sets={"train":read_xml(source/"SMP2019_ECISA_Train.xml","train"),"external_valid":read_xml(source/"SMP2019_ECISA_Dev.xml","external_valid")}
    duplicates=Counter();seen=set();counts={}
    # 外部验证优先保留；任何相同句子不得跨训练/外部验证。
    for split in ["external_valid","train"]:
        kept=[]
        for row in sets[split]:
            key=normalize(row["text"])
            if not key or key in seen:duplicates[split+"_duplicate"]+=1;continue
            if key in all_target or any(key[i:i+8] in grams for i in range(max(0,len(key)-7))):
                duplicates[split+"_target_overlap"]+=1;continue
            seen.add(key);kept.append(row)
        sets[split]=kept
        (output/(split+".jsonl")).write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in kept)+'\n',encoding="utf-8")
        counts[split]=dict(Counter(row["label"] for row in kept))
    report={"source":"https://github.com/sxu-nlp/ECISA2021","license_scope":"official README: technology evaluation and academic research only; no commercial use without permission",
        "label_map":{"0_non_sentiment":1,"1_positive_implicit":2,"2_negative_implicit":0},"counts":counts,"removed":dict(duplicates),
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('SMP2019*.xml')},
        "external_validation_policy":"never used for CH-SIMS hyperparameter selection; ECISA train may be auxiliary training",
        "dedup_limit":"text exact/8-character overlap screening cannot prove no shared films/actors or pretraining contamination",
        "real_human_target_population":False}
    (output/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["source","bundle","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.source,args.bundle,args.output)
