"""核验被时间戳修正恢复的原始音画向量，不允许把补零当成恢复观测。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
from experiments.round11_common import ROOT,OUTPUT,save


def main() -> None:
    """对所有47条恢复记录核验与最初真实媒体特征逐位相同。"""
    rows=[json.loads(line) for line in (OUTPUT/"input_audit/records.jsonl").read_text("utf-8").splitlines()]
    restored=[r for r in rows if not r["old_mask"][1] and r["new_mask"][1]]
    parts={}
    for split in ("train","selection"):
        with np.load(ROOT/f"_国赛优化/round5/face_features/{split}.npz",allow_pickle=False) as z:
            parts[split]={"sample_ids":z["sample_ids"].tolist(),"audio":z["audio"],"visual":z["visual"]}
    checks=[]
    for row in restored:
        part=parts[row["split"]];index=part["sample_ids"].index(row["id"])
        with np.load(ROOT/f"_国赛优化/round3/model/ch_sims_v2/sample_features/{row['id']}.npz",allow_pickle=False) as z:
            equal={key:bool(np.array_equal(part[key][index],z[key])) for key in ("audio","visual")}
            checks.append({"id":row["id"],"bitwise_equal_original_media_features":equal,
                "audio_norm":float(np.linalg.norm(part["audio"][index])),"visual_norm":float(np.linalg.norm(part["visual"][index]))})
        if not all(equal.values()) or checks[-1]["audio_norm"]==0:raise ValueError("恢复观测的原始向量不一致或为空")
    save(OUTPUT/"validation/restored_inputs_bitwise.json",{"n":len(checks),"checks":checks,"all_original_vectors_preserved":True,
        "labels_used":False},immutable=False)
    print(json.dumps({"restored_real_inputs_verified":len(checks),"bitwise_equal":True}),flush=True)


if __name__=="__main__":main()
