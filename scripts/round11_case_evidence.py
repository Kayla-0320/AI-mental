"""固定开发成功/回退案例的来源留出专家证据，不把人工参考标签当模型输入。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import pickle
import sys
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from experiments.round11_common import OUTPUT,ROOT,digest,save
from experiments.round11_features import load_data
from experiments.round11_evidence import expert_evidence
from experiments.round10_fusion import take


def main() -> None:
    """事先已有的纠正列表中首个T/A相反例、首个回退例，固定规则可复查。"""
    folder=OUTPUT/"cases";folder.mkdir(exist_ok=True)
    if (folder/"expert_evidence.json").exists():return
    audit=json.loads((OUTPUT/"development_conflict_audit_v2.json").read_text("utf-8"))
    success=next(r for r in audit["cases"]["corrected"] if r["T_A_V"][0]!=1 and r["T_A_V"][1]!=1 and r["T_A_V"][0]!=r["T_A_V"][1])
    chosen=[success,audit["cases"]["regressed"][0]]
    data=load_data("pts");ids=data["sample_ids"].tolist();rows=[]
    font=FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout="constrained")
    for case,ax in zip(chosen,axes):
        i=ids.index(case["id"])
        for fold in range(4):
            path=OUTPUT/f"evidence_deterministic/family10/fold{fold}"
            with np.load(path.with_suffix(".npz")) as z:
                if case["id"] not in z["valid_ids"]:continue
            with path.with_name(path.name+"_models.pkl").open("rb") as handle:bundle=pickle.load(handle)
            part=take(data,np.array([i]));part.pop("labels")
            if str(part["source_groups"][0]) in set(bundle["training_groups"]):raise ValueError("案例来源进入训练")
            e=expert_evidence(bundle["experts"],part)[0];prob=e[:12].reshape(4,3)
            rows.append({**case,"source_group":str(data["source_groups"][i]),"held_out_fold":fold,
                "expert_probability_M_T_A_V":prob.tolist(),"expert_expected_polarity":e[16:20].tolist(),
                "expert_available":e[24:28].astype(bool).tolist(),"media":str(ROOT/f"data/public/ch_sims_v2/supervised/{case['id']}.mp4"),
                "labels_only_for_reference_display":True,"not_causal_attribution":True})
            x=np.arange(4)
            for j,(name,color) in enumerate(zip(("负向","中性","正向"),("#456DA8","#E4AA38","#21A28F"))):
                bars=ax.bar(x+(j-1)*.24,prob[:,j],.24,label=name,color=color)
            ax.set_xticks(x,["整体M","文字T","语音A","视觉V"],fontproperties=font);ax.set_ylim(0,1)
            ax.set_title(("纠正案例" if case is success else "回退案例")+"："+case["id"],fontproperties=font)
            ax.set_ylabel("来源留出专家概率（不是因果贡献）",fontproperties=font);ax.legend(prop=font);ax.grid(axis="y",alpha=.2)
            break
    fig.savefig(OUTPUT/"figures/case_expert_evidence.png",dpi=220);plt.close(fig)
    save(folder/"expert_evidence.json",{"selection":"first T/A-opposite corrected from preexisting list + first regressed;not test success search",
        "rows":rows,"code_sha256":digest(Path(__file__)),"scope":"development explanatory evidence only"})
    print(json.dumps({"case_evidence":len(rows)}),flush=True)


if __name__=="__main__":main()
