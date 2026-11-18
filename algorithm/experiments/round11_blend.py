"""仅以来源OOF检验新旧证据的互补，原历史成绩不参与选型。"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from experiments.round11_common import OUTPUT,PREVIOUS,digest,save


def scores(y: np.ndarray,pred: np.ndarray,weights: np.ndarray | None = None) -> tuple[float,float]:
    """固定三类宏F1与准确率，支持来源等权样本权重。"""
    matrix=np.bincount(3*y+pred,weights=weights,minlength=9).reshape(3,3)
    denominator=matrix.sum(0)+matrix.sum(1)
    f1=np.divide(2*np.diag(matrix),denominator,out=np.zeros(3),where=denominator>0)
    return float(f1.mean()),float(np.trace(matrix)/matrix.sum())


def main() -> None:
    """五档混合权重与固定676点截距，分别保留纯旧/纯新控制。"""
    output=OUTPUT/"blend"
    if (output/"selection_locked.json").exists():
        raise ValueError("拒绝覆盖新旧证据融合锁")
    grid=np.round(np.arange(-.5,.750001,.05),8).tolist()
    save(output/"preregister.json",{"new_weights":[0.,.25,.5,.75,1.],"bias_grid":grid,
        "old":"frozen Round10 refined selected source OOF with its existing bias",
        "new":"Round11 family10 depth4 no reliability source OOF",
        "selection":".5 pooled macroF1+.5 source-balanced macroF1;bias norm and simpler pure model break ties",
        "history821_and_external_test_targets_used":False,"code_sha256":digest(Path(__file__))})
    with np.load(PREVIOUS/"refined_final/selected_oof.npz",allow_pickle=False) as z:
        old=z["probabilities"];ids=z["sample_ids"];y=z["labels"]
    with np.load(OUTPUT/"evidence_deterministic/family10/oof.npz",allow_pickle=False) as z:
        if not np.array_equal(ids,z["sample_ids"]) or not np.array_equal(y,z["labels"]):
            raise ValueError("新旧来源OOF身份不符")
        new=z["probabilities"][0];groups=z["source_groups"]
    _,inverse,count=np.unique(groups,return_inverse=True,return_counts=True)
    weight=1/count[inverse]
    rows=[]
    best=None
    for fraction in [0.,.25,.5,.75,1.]:
        p=(1-fraction)*old+fraction*new
        logits=np.log(np.maximum(p,1e-12))
        for negative,neutral in itertools.product(grid,grid):
            bias=[negative,neutral,0.]
            predicted=(logits+np.array(bias)).argmax(1)
            f1,accuracy=scores(y,predicted)
            grouped,_=scores(y,predicted,weight)
            row={"new_weight":fraction,"bias":bias,"macro_f1":f1,"accuracy":accuracy,
                 "source_balanced_f1":grouped,"selection_score":.5*(f1+grouped)}
            rows.append(row)
            key=(row["selection_score"],f1,-sum(abs(v) for v in bias),float(fraction in (0.,1.)))
            if best is None or key>best[0]:best=(key,row,p)
    selected=best[1]
    z=np.log(np.maximum(best[2],1e-12))+selected["bias"]
    probability=np.exp(z-z.max(1,keepdims=True));probability/=probability.sum(1,keepdims=True)
    save(output/"candidates.json",rows)
    save(output/"selection_locked.json",{"winner":selected,"candidate_count":len(rows),
        "controls":[max([r for r in rows if r["new_weight"]==v],key=lambda r:r["selection_score"]) for v in (0.,1.)],
        "fit_recipe":"reuse immutable Round10 model;new family trained on nonhistorical3579;five meta seeds42..46",
        "not_final_external_evaluation_lock":True})
    np.savez_compressed(output/"selected_oof.npz",probabilities=probability,sample_ids=ids,labels=y,source_groups=groups)
    print(json.dumps(selected),flush=True)


if __name__=="__main__":
    main()
