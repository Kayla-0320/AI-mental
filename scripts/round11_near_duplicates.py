"""跨来源近似转录候选与真实音轨复核，不修改样本与目标。"""
from __future__ import annotations
from collections import defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys
from typing import Any
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
from experiments.round11_common import ROOT,OUTPUT,PREVIOUS,digest,save


def normalized(text: str) -> str:
    """只留文字和数字，不读取任何情感目标。"""
    return "".join(c.lower() for c in text if c.isalnum())


def main() -> None:
    """长度至少12、三字片段Jaccard至少.8；波形复核仅对这些候选。"""
    output=OUTPUT/"near_duplicate_audit.json"
    if output.exists():return
    records=[]
    for split in ("train","valid","test_inputs"):
        for line in (OUTPUT/"corpus"/f"{split}.jsonl").read_text("utf-8").splitlines():
            row=json.loads(line);records.append({"id":row["id"],"split":split,"dataset":"EmotionTalk","text":normalized(row["text"])})
    for split in ("train","selection","calibration","test"):
        with np.load(ROOT/f"_国赛优化/round5/text_features/{split}.npz",allow_pickle=False) as z:
            records.extend({"id":str(i),"split":split,"dataset":"CH-SIMS","text":normalized(str(t))} for i,t in zip(z["sample_ids"],z["raw_text"]))
    inverted: dict[str,list[int]]=defaultdict(list);grams=[];candidates=[]
    for i,row in enumerate(records):
        text=row["text"];g={text[j:j+3] for j in range(len(text)-2)} if len(text)>=12 else set();grams.append(g)
        seen=set(k for word in g for k in inverted[word])
        for k in seen:
            other=records[k]
            if row["dataset"]==other["dataset"] and row["split"]==other["split"]:continue
            if min(len(g),len(grams[k]))<.8*max(len(g),len(grams[k])):continue
            similarity=len(g&grams[k])/len(g|grams[k])
            if similarity>=.8:candidates.append((k,i,similarity))
        for word in g:inverted[word].append(i)
    from round11_sensevoice import decode
    rows=[]
    for a,b,sim in candidates:
        pair=[records[a],records[b]];waves=[]
        for row in pair:
            directory="emotiontalk_media/clips" if row["dataset"]=="EmotionTalk" else "ch_sims_v2/supervised"
            wave,duration,valid=decode(ROOT/f"data/public/{directory}/{row['id']}.mp4")
            waves.append((wave,duration,valid))
        left,right=waves;correlation=None
        if left[2] and right[2] and abs(left[1]-right[1])<=.15 and min(len(left[0]),len(right[0]))>=16000:
            # 比较相同时轴，容许100毫秒小平移；不作为普适说话人/媒体身份识别。
            x=left[0][::8].astype(float);y=right[0][::8].astype(float);corr=[]
            for offset in range(-200,201,10):
                startx=max(0,offset);starty=max(0,-offset);length=min(len(x)-startx,len(y)-starty)
                if length<2000:continue
                xx=x[startx:startx+length];yy=y[starty:starty+length]
                xx=xx-xx.mean();yy=yy-yy.mean()
                corr.append(abs(float(xx@yy/max(1e-12,np.linalg.norm(xx)*np.linalg.norm(yy)))))
            correlation=max(corr) if corr else None
        rows.append({"left":{k:pair[0][k] for k in ("id","split","dataset")},"right":{k:pair[1][k] for k in ("id","split","dataset")},
            "text_jaccard3":sim,"durations":[left[1],right[1]],"waveform_correlation":correlation,
            "possible_reencoded_media":correlation is not None and correlation>=.98})
    save(output,{"rule":"length>=12 normalized chars;trigram Jaccard>=.8 across split/dataset;audio correlation for duration-close candidates",
        "n_inputs":len(records),"candidate_count":len(rows),"possible_reencoded_pairs":sum(r["possible_reencoded_media"] for r in rows),
        "candidates":rows,"samples_removed":0,"labels_read":False,"code_sha256":digest(Path(__file__)),
        "limits":"candidate-based audit;short paraphrases,large edits and unknown upstream pretraining overlap remain unverified"})
    print(json.dumps({"near_duplicate_candidates":len(rows),"possible_reencoded":sum(r["possible_reencoded_media"] for r in rows)}),flush=True)


if __name__=="__main__":main()
