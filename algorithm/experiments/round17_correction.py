"""预测专家替换的纠错收益而非盲目选择最自信专家，质量和情绪分歧保持分离。"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round13_train import ch_data
from experiments.round10_fusion import folds,take
from experiments.round16_moe import gate_features
from experiments.round17_quality import route,score

OUT=ROOT/"round17/quality"


def features(bank: np.ndarray,mask: np.ndarray,quality: np.ndarray) -> np.ndarray:
    """概率分歧只作为决策特征，不把它转换为原始图像质量标签。"""
    return np.column_stack((gate_features(bank,mask),np.nan_to_num(quality,nan=1.),np.isfinite(quality)))


def decide(bank: np.ndarray,mask: np.ndarray,gain: np.ndarray,threshold: float,prior: np.ndarray) -> np.ndarray:
    """只有替代专家的估计净纠错收益超过门槛才替换，资格不足绝不参与。"""
    baseline=route(bank,mask,prior);code=mask.astype(int)@np.array([1,2,4]);allowed=np.column_stack([((code&p)==p) for p in (1,4,5)])
    change=gain.copy();change[~allowed]=-np.inf
    change[bank.argmax(2)==baseline.argmax(1)[:,None]]=-np.inf
    chosen=change.argmax(1);active=change[np.arange(len(mask)),chosen]>threshold
    result=baseline.copy();result[active]=bank[np.arange(len(mask))[active],chosen[active]]
    return result


def models_fit(x: np.ndarray,target: np.ndarray) -> list[Any]:
    """固定三个小回归器预测替换损益，训练数据只来自外层训练来源。"""
    result=[]
    for j in range(3):
        model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=3.,learning_rate=.05,random_state=42)
        result.append(model.fit(x,target[:,j]))
    return result


def predict_correction(bank: np.ndarray,mask: np.ndarray,quality: np.ndarray,folder: Path,prior: np.ndarray) -> np.ndarray:
    """加载冻结损益预测器；入口只有T/V专家概率、可用性和质量，不接受标签。"""
    if bank.shape!=(len(mask),3,3) or mask.shape!=(len(bank),3) or mask.dtype!=bool or quality.shape!=(len(bank),):raise ValueError("专家概率/资格形状无效")
    if mask[:,1].any():raise ValueError("不接受音频资格")
    if not np.isfinite(bank).all() or (bank<0).any() or not np.allclose(bank.sum(2),1.,atol=1e-6):raise ValueError("专家概率无效")
    policy=json.loads((folder/"policy.json").read_text(encoding="utf-8"))
    if sha256(folder/"models.pkl")!=policy["model_sha256"]:raise ValueError("纠错模型摘要不一致")
    with (folder/"models.pkl").open("rb") as stream:models=pickle.load(stream)
    x=features(bank,mask,quality);gain=np.column_stack([m.predict(x) for m in models])
    return decide(bank,mask,gain,policy["threshold"],prior)


def run() -> None:
    """内层元OOF确定保守切换门槛，所有专家、门控和门槛均不见外层来源。"""
    freeze(OUT/"correction_protocol.json",{"seed":42,"thresholds":[0.,.05,.1,.2,2.],"learner":"3 HGB regression,100iters,7leaves,min40,l2=3",
        "target":"expert correct minus uniform correct (-1,0,1), not a quality/mental-state label",
        "selection":"maximize inner meta-OOF F1 with accuracy>=uniform; threshold2 guarantees no switches",
        "scope":"exposed development-source diagnostic; fixed one candidate bank, no outer-test tuning","source_sha256":sha256(Path(__file__))})
    data=ch_data();data["mask"][:,1]=False;quality=json.loads((OUT/"raw_quality_v2.json").read_text(encoding="utf-8"))
    q=np.array([quality[str(i)]["score"] if quality[str(i)]["measured"] else np.nan for i in data["sample_ids"]])
    saved={c:np.load(ROOT/f"round16/moe/{c}_predictions.npz",allow_pickle=False) for c in ("TV","T_only","V_only")}
    bank=np.stack((saved["T_only"]["uniform"],saved["V_only"]["uniform"],saved["TV"]["ordinary_TV"]),1)
    output=np.zeros((len(q),3));locks=[]
    for fold,(tr,va) in enumerate(folds(data)):
        folder=OUT/f"correction_models/fold{fold}";folder.mkdir(parents=True,exist_ok=True)
        inner=np.load(ROOT/f"round13/neural/ch/fold{fold}/inner_oof.npz",allow_pickle=False)["missing_A"][:,[0,3,4],:]
        with (ROOT/f"round16/moe/export/fold{fold}/experts.pkl").open("rb") as stream:prior=pickle.load(stream)["prior"]
        train=take(data,tr);base=route(inner,train["mask"],prior);base_correct=(base.argmax(1)==train["labels"]).astype(float)
        target=(inner.argmax(2)==train["labels"][:,None]).astype(float)-base_correct[:,None]
        x=features(inner,train["mask"],q[tr]);oof=np.zeros_like(target)
        for a,b in folds(train,3):
            fitted=models_fit(x[a],target[a]);oof[b]=np.column_stack([m.predict(x[b]) for m in fitted])
        baseline_metric=score(train["labels"],base);scores=[]
        for threshold in (0.,.05,.1,.2,2.):
            p=decide(inner,train["mask"],oof,threshold,prior);metrics=score(train["labels"],p)
            scores.append({"threshold":threshold,**metrics,"switched":int(np.sum(p.argmax(1)!=base.argmax(1)))})
        eligible=[r for r in scores if r["accuracy"]>=baseline_metric["accuracy"]-1e-12]
        winner=max(eligible,key=lambda r:(r["macro_f1"],-r["switched"]))
        fitted=models_fit(x,target);gain=np.column_stack([m.predict(features(bank[va],data["mask"][va],q[va])) for m in fitted])
        output[va]=decide(bank[va],data["mask"][va],gain,winner["threshold"],prior)
        with (folder/"models.pkl").open("wb") as stream:pickle.dump(fitted,stream)
        save(folder/"policy.json",{"threshold":winner["threshold"],"model_sha256":sha256(folder/"models.pkl")})
        locks.append({"fold":fold,"winner":winner,"scores":scores,"train_groups":np.unique(train["source_groups"]).tolist(),"held_groups":np.unique(data["source_groups"][va]).tolist()})
        print(json.dumps({"correction_fold":fold,"threshold":winner["threshold"]}),flush=True)
    baseline=saved["TV"]["uniform"];y=data["labels"]
    result={"baseline":score(y,baseline),"correction":score(y,output),"switched":int(np.sum(output.argmax(1)!=baseline.argmax(1))),
        "rescued":int(np.sum((baseline.argmax(1)!=y)&(output.argmax(1)==y))),"introduced_errors":int(np.sum((baseline.argmax(1)==y)&(output.argmax(1)!=y)))}
    np.savez_compressed(OUT/"correction_predictions.npz",truth=y,groups=data["source_groups"],sample_ids=data["sample_ids"],selected=output,baseline=baseline)
    save(OUT/"correction_selection_locks.json",locks);save(OUT/"correction_results.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":run()
