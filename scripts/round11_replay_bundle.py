"""复核锁定工件并从冻结权重重放逐条概率，可选核对独立重训副本。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
from experiments.round11_common import OUTPUT,PREVIOUS,digest,save
from experiments.round11_final import load_data,predict
from experiments.round11_external_data import load_data as external_data


def check_lock(path: Path) -> int:
    """核验真实模型与选型文件，拒绝只检查标记文件是否存在。"""
    body=json.loads(path.read_text("utf-8"));count=0
    for row in body.get("model_files",[])+body.get("selection_files",[])+body.get("models",[]):
        if digest(Path(row["path"]))!=row["sha256"]:raise ValueError(f"锁摘要改变:{row['path']}")
        count+=1
    return count


def main() -> None:
    """历史重放不再选择候选，只核对保存概率及无标签预测一致性。"""
    parser=argparse.ArgumentParser();parser.add_argument("--retrained",type=Path);args=parser.parse_args()
    checks={"lock_hash_checks":sum(check_lock(OUTPUT/path) for path in
        ("forecast_final/evaluation_lock.json","external_m3/evaluation_lock.json","historical_diagnostic/evaluation_lock.json")),"predictions":[]}
    data=load_data(("test",),"7b")
    with (OUTPUT/"final_models/7b/model.pkl").open("rb") as handle:model=pickle.load(handle)
    clean={k:v for k,v in data.items() if k!="labels"};p=predict(model,clean)
    with np.load(OUTPUT/"historical_diagnostic/7b_predictions.npz",allow_pickle=False) as z:
        if not np.array_equal(data["sample_ids"],z["sample_ids"]):raise ValueError("历史ID发生变化")
        error=float(np.abs(p-z["probabilities"]).max())
    if error>1e-10:raise ValueError("历史冻结概率重放不符")
    modified=dict(data);modified["labels"]=(data["labels"]+1)%3
    label_error=float(np.abs(p-predict(model,modified)).max())
    if label_error!=0:raise ValueError("推理依赖金标")
    checks["predictions"].append({"task":"historical821 main","n":len(p[0]),"probability_max_error":error,"mutated_labels_max_error":label_error})
    if args.retrained:
        with args.retrained.open("rb") as handle:retrained=pickle.load(handle)
        q=predict(retrained,clean);delta=float(np.abs(q-p).max())
        checks["independent_refit"]={"path":str(args.retrained),"probability_max_error":delta,"argmax_disagreements":int(np.sum(q.argmax(2)!=p.argmax(2)))}
        if delta>1e-6:raise ValueError("独立重训与冻结权重概率误差超容差")
    data=external_data(("test_inputs",),"pts","7b");q=predict(model,data)
    with np.load(OUTPUT/"external_m3/test_inputs_locked_predictions.npz",allow_pickle=False) as z:
        error=float(np.abs(q-z["new"]).max())
        if not np.array_equal(data["sample_ids"],z["sample_ids"]):raise ValueError("外部ID发生变化")
    if error>1e-10:raise ValueError("外部概率重放不符")
    checks["predictions"].append({"task":"external1929 main","n":len(q[0]),"probability_max_error":error})
    manifest=json.loads((PREVIOUS/"artifact_manifest.json").read_text("utf-8"));failures=[]
    for row in manifest["artifacts"]:
        if digest(PREVIOUS/row["path"])!=row["sha256"]:failures.append(row["path"])
    if failures:raise ValueError("第十轮冻结工件改变")
    checks["round10_preservation"]={"n":len(manifest["artifacts"]),"failures":failures,"manifest_sha256":digest(PREVIOUS/"artifact_manifest.json")}
    save(OUTPUT/"validation/final_replay.json",checks,immutable=False)
    print(json.dumps(checks),flush=True)


if __name__=="__main__":main()
