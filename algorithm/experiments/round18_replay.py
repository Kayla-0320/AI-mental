"""仅依靠交付缓存与冻结模型只读回放，不需要重新下载音视频或大型编码器。"""
from __future__ import annotations
import json
from pathlib import Path
import pickle
from typing import Any
import numpy as np
from experiments.round14_common import sha256
from experiments.round10_fusion import folds
from experiments.round18_runtime import OUT,predict_neutral,predict_day,predict_quality_features
from experiments.round18_images import CONDITIONS
from experiments.round18_graph import validate,retrieve


def replay() -> dict[str,Any]:
    """复核每一行概率与点预测，只返回结果，不改动冻结文件或总清单。"""
    if (OUT/"manifest.json").exists():
        manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
        for name,item in manifest.items():
            path=(OUT/name).resolve()
            if not path.is_relative_to(OUT.resolve()) or sha256(path)!=item["sha256"]:raise ValueError("交付清单校验失败")
    x=np.load(OUT/"neutral/runtime_inputs.npz",allow_pickle=False);n=np.load(OUT/"neutral/predictions.npz",allow_pickle=False);errors=[]
    for fold,(_,va) in enumerate(folds({"labels":x["labels"],"source_groups":x["source_groups"]})):
        p=predict_neutral(x["bank"][va],x["mask"][va],x["quality"][va],OUT/f"neutral/fold{fold}");errors.append(float(np.max(np.abs(p-n["selected"][va]))))
    image=np.load(OUT/"images/reencoded.npz",allow_pickle=False)
    banks={c:np.zeros((len(image["labels"]),3,3)) for c in CONDITIONS}
    for fold,(_,va) in enumerate(folds({"labels":image["labels"],"source_groups":image["groups"]})):
        with (OUT/f"images/fold{fold}.pkl").open("rb") as stream:bundle=pickle.load(stream)
        for c in CONDITIONS:
            t=bundle["projectors"]["T"].transform(bundle["scalers"]["T"].transform(image["T"][va]))/np.sqrt(image["T"].shape[1])
            v=bundle["projectors"]["V"].transform(bundle["scalers"]["V"].transform(image[c][va]))/np.sqrt(image[c].shape[1])
            banks[c][va]=np.stack([h.predict_proba(m) for h,m in zip(bundle["heads"],(t,v,np.column_stack((t,v))),strict=True)],1)
    for c in CONDITIONS:
        saved=np.load(OUT/f"images/{c}_predictions.npz",allow_pickle=False);g=np.load(OUT/f"images/{c}_guard.npz",allow_pickle=False);b=banks[c]
        quality=b.mean(1);drop=image[c+"_quality"]<=.51;quality[drop]=b[drop,0]
        conservative=b.mean(1);conservative[g["drop"]]=b[g["drop"],0]
        for p,q in ((b[:,2],saved["ordinary_TV"]),(b.mean(1),saved["uniform"]),(quality,saved["quality"]),(conservative,g["selected"])):errors.append(float(np.max(np.abs(p-q))))
    rows=json.loads((OUT/"forecast/windows.json").read_text(encoding="utf-8"));p=[]
    for row in rows:
        actual=predict_day({"source_schema":"openesm:beck:sad:1_5","target_day":row["target_day"],"observations":[{"person":row["person"],"day":int(d),"value":1+4*v} for d,v in zip(row["history_days"],row["history"],strict=True)]})
        p.append(actual["normalized_prediction"])
    saved=np.load(OUT/"forecast/predictions.npz",allow_pickle=False);errors.append(float(np.max(np.abs(np.array(p)-saved["prediction"]))))
    qx=np.load(OUT/"quality_model/features.npz",allow_pickle=False);qy=np.load(OUT/"quality_model/test_predictions.npz",allow_pickle=False)
    errors.append(float(np.max(np.abs(predict_quality_features(qx["X"][960:],OUT/"quality_model")-qy["selected"]))))
    graph=json.loads((OUT/"knowledge_graph/graph.json").read_text(encoding="utf-8"));assert not validate(graph) and not retrieve(graph,"锚定当下","act")["facts"]
    assert max(errors)<1e-12
    return {"neutral_rows":len(n["truth"]),"RGB_classification_rows":5*len(image["labels"]),"quality_holdout_images":len(qy["truth"]),"forecast_windows":len(rows),"max_prediction_error":max(errors),"pending_graph_default_abstains":True,"raw_pixels_redecoded_in_this_cache_replay":False,"manifest_checked":(OUT/"manifest.json").exists()}


if __name__=="__main__":print(json.dumps(replay()),flush=True)
