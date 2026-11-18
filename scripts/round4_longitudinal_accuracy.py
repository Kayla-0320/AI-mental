"""在既有参与者划分中比较MAE目标模型；不赋予匿名条目临床含义。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any

for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from round3_longitudinal_predict import (load_rows, quarantine_duplicates, build_frame, split_participants,
    features, regression_metrics, bootstrap_mae_delta, G_ITEMS, H_ITEMS, _number, _sha256)


def item_features(rows: list[tuple], header: list[str], participants: list[str]) -> np.ndarray:
    """仅第一波逐条回答和年龄；不允许随访字段或身份编码作为特征。"""
    positions={name:i for i,name in enumerate(header)}
    lookup={f"{row[positions['cohort']]}-{row[positions['code']]}":row for row in rows}
    columns=[f"{name}_1" for name in G_ITEMS+H_ITEMS]
    result=[]
    for participant in participants:
        row=lookup[participant]
        result.append([*[_number(row[positions[name]]) for name in columns],float(row[positions["age"]])])
    matrix=np.asarray(result,dtype=float)
    if not np.isfinite(matrix).all(): raise ValueError("基线条目特征包含缺失值")
    return matrix


def run(source: Path, output: Path) -> dict[str,Any]:
    """仅按选择集MAE锁定模型；历史测试不参与选择。"""
    output.mkdir(parents=True,exist_ok=True)
    destination=output/"longitudinal_results.json"
    if destination.exists():raise ValueError("拒绝覆盖已有实验")
    header,rows=load_rows(source);kept,_=quarantine_duplicates(rows,header)
    frame=build_frame(kept,header);split=split_participants(frame["participants"])
    matrices={"sums_age":np.column_stack([frame[k] for k in ["wave1_g","wave1_h","age"]]),
              "items_age":item_features(kept,header,frame["participants"])}
    configs=[{"id":"persistence","kind":"persistence"},{"id":"train_median","kind":"median"}]
    configs += [{"id":f"shrink_{a}","kind":"shrink","alpha":a} for a in [.25,.5,.75,1.]]
    configs += [{"id":f"ridge_{a}","kind":"ridge","alpha":a} for a in [.1,1.,10.,100.]]
    configs += [{"id":f"hgb_{name}_{leaves}","kind":"hgb","features":name,"leaves":leaves}
                for name in matrices for leaves in [7,15]]
    protocol={"seed":42,"source_sha256":_sha256(source),"code_sha256":_sha256(Path(__file__)),"candidates":configs,
              "selection":"minimum selection MAE; train only fits models","split":"same participant-level 60/20/20 as round3",
              "historical_test":"already examined; diagnostic only", "construct":"uninterpreted g/h item sums, not clinical scales",
              "spearman":"average tied ranks; constant predictions undefined/null",
              "features":"first-wave items/sums and age only; no followup, date, identity, or full-course summaries",
              "split_participant_sha256":{name:__import__('hashlib').sha256('\n'.join(frame['participants'][i] for i in ix).encode()).hexdigest() for name,ix in split.items()}}
    (output/"preregister.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    result={}
    for target,baseline in [("wave2_g","wave1_g"),("wave2_h","wave1_h")]:
        tr,va,te=split["train"],split["selection"],split["test"]
        train_y,valid_y=frame[target][tr],frame[target][va]
        candidates={};predictors={}
        for config in configs:
            kind=config["kind"]
            if kind in {"median","persistence","shrink"}:
                center=float(np.median(train_y));change=float(np.median(train_y-frame[baseline][tr]))
                alpha=config.get("alpha",1.)
                def predict(ix: np.ndarray, kind: str=kind, center: float=center,change: float=change,alpha: float=alpha) -> np.ndarray:
                    if kind=="median":return np.full(len(ix),center)
                    if kind=="persistence":return frame[baseline][ix]
                    return alpha*(frame[baseline][ix]+change)+(1-alpha)*center
            else:
                x=(features(frame,np.arange(len(train_y)+len(valid_y)+len(te)),["wave1_g","wave1_h","age","race"])
                   if kind=="ridge" else matrices[config["features"]])
                model=(make_pipeline(StandardScaler(),Ridge(alpha=config["alpha"])) if kind=="ridge" else
                       HistGradientBoostingRegressor(loss="absolute_error",max_iter=150,max_leaf_nodes=config["leaves"],
                           min_samples_leaf=30,l2_regularization=1.,learning_rate=.05,early_stopping=False,random_state=42))
                model.fit(x[tr],train_y)
                def predict(ix: np.ndarray, model: Any=model,x: np.ndarray=x) -> np.ndarray:
                    return model.predict(x[ix])
            predictors[config["id"]]=predict
            candidates[config["id"]]={"selection":regression_metrics(valid_y,predict(va))}
        winner=min(candidates,key=lambda key:candidates[key]["selection"]["mae"])
        (output/(target+"_selection_locked.json")).write_text(json.dumps({"winner":winner,"selection":candidates},indent=2),encoding="utf-8")
        truth=frame[target][te]
        for name,predict in predictors.items():candidates[name]["historical_diagnostic"]=regression_metrics(truth,predict(te))
        predictions={name:predict(te) for name,predict in predictors.items()}
        result[target]={"winner":winner,"candidates":candidates,"selected_minus_persistence":{
            "delta_mae":candidates[winner]["historical_diagnostic"]["mae"]-candidates["persistence"]["historical_diagnostic"]["mae"],
            "ci95":bootstrap_mae_delta(truth,predictions[winner],predictions["persistence"])}}
        np.savez_compressed(output/(target+"_predictions.npz"),truth=truth,**predictions)
    report={"protocol":protocol,"targets":result,"deployed":False,"new_independent_validation":False}
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:{"winner":v["winner"],"selected":v["candidates"][v["winner"]]["historical_diagnostic"],"comparison":v["selected_minus_persistence"]} for k,v in result.items()},ensure_ascii=False),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
