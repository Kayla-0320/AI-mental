"""从逐样本预测独立复算第四轮数字，不重新训练或改选模型。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from round3_longitudinal_predict import regression_metrics


def verify(output: Path) -> dict[str,Any]:
    """逐项核对模型、故障、回归预测与原始记录的一致性。"""
    checked=[]
    model=json.loads((output/"model/model_results.json").read_text(encoding="utf-8"))
    locked=json.loads((output/"model/selection_locked.json").read_text(encoding="utf-8"))
    if locked["winner"]!=model["winner"]:raise ValueError("选型被历史结果改写")
    for candidate,scores in model["historical_diagnostic"].items():
        with np.load(output/"model"/(candidate+"_predictions.npz"),allow_pickle=False) as saved:
            labels,logits=saved["labels"],saved["logits"]
            if len(set(saved["sample_ids"]))!=len(labels) or not np.isfinite(logits).all():raise ValueError("预测样本无效")
            pred=logits.argmax(1)
            assert abs(accuracy_score(labels,pred)-scores["raw"]["accuracy"])<1e-12
            assert abs(f1_score(labels,pred,labels=[0,1,2],average="macro",zero_division=0)-scores["raw"]["macro_f1"])<1e-12
        checked.append("classification/"+candidate)
    quality=json.loads((output/"quality/quality_results.json").read_text(encoding="utf-8"))
    for scenario,conditions in quality["historical_diagnostic"].items():
        ids=None
        for mode in ["ignore","oracle","automatic"]:
            with np.load(output/"quality"/(scenario+"_"+mode+".npz"),allow_pickle=False) as saved:
                if ids is not None:assert np.array_equal(ids,saved["sample_ids"])
                ids=saved["sample_ids"].copy()
                pred=saved["logits"].argmax(1)
                assert abs(f1_score(saved["labels"],pred,labels=[0,1,2],average="macro",zero_division=0)-conditions[mode]["macro_f1"])<1e-12
                assert abs(accuracy_score(saved["labels"],pred)-conditions[mode]["accuracy"])<1e-12
            checked.append("quality/"+scenario+"/"+mode)
    longitudinal=json.loads((output/"longitudinal/longitudinal_results.json").read_text(encoding="utf-8"))
    for target,result in longitudinal["targets"].items():
        with np.load(output/"longitudinal"/(target+"_predictions.npz"),allow_pickle=False) as saved:
            for candidate,score in result["candidates"].items():
                recomputed=regression_metrics(saved["truth"],saved[candidate])
                for metric in ["mae","rmse","r2","spearman"]:
                    actual,expected=recomputed[metric],score["historical_diagnostic"][metric]
                    assert actual is None and expected is None or actual is not None and expected is not None and abs(actual-expected)<1e-12
                checked.append("longitudinal/"+target+"/"+candidate)
    for component in ["model","quality","longitudinal"]:
        protocol=json.loads((output/component/"preregister.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((output/component/"source_snapshot.py").read_bytes()).hexdigest()==protocol["code_sha256"]
        checked.append("source_snapshot/"+component)
    agent=json.loads((output/"agent/support_results.json").read_text(encoding="utf-8"))
    for name,result in agent["results"].items():
        for split,summary in result["summary"].items():
            rows=[row for row in result["rows"] if row["split"]==split]
            assert sum(row["goal"]==row["expected_goal"] and row["plan"]==row["expected_plan"] for row in rows)==summary["both_correct"]
            checked.append("agent/"+name+"/"+split)
    assert hashlib.sha256((output/"agent/current_support_understanding.py").read_bytes()).hexdigest()==agent["current_sha256"]
    result={"status":"passed","checks":len(checked),"checked":checked,
            "method":"recomputed saved per-sample predictions; no retraining or test-based selection",
            "independent_human_review":False}
    folder=output/"verification";folder.mkdir(exist_ok=True)
    (folder/"recomputed.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"checks":len(checked)}))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    verify(parser.parse_args().output)
