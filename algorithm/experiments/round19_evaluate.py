"""同编码器、同容量和相同选型预算的知识消融，测试标签仅在全部锁型后读取。"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import pickle
from typing import Any
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,accuracy_score
from sklearn.svm import LinearSVC
from experiments.round14_common import freeze,save,sha256
from experiments.round19_prepare import OUT
from experiments.round16_statistics import macro

ARMS=("no_graph","definitions_only","true_graph","shuffled_graph")


def metrics(y: np.ndarray,score: np.ndarray) -> dict[str,Any]:
    """八类原标注一致性，不是回复帮助程度或临床有效性。"""
    pred=score.argmax(1)
    return {"macro_f1":float(f1_score(y,pred,labels=np.arange(8),average="macro",zero_division=0)),"accuracy":float(accuracy_score(y,pred)),"per_class_f1":f1_score(y,pred,labels=np.arange(8),average=None,zero_division=0).tolist(),"n":len(y)}


def matrix(lexical: sparse.spmatrix,z: Any,arm: str) -> sparse.csr_matrix:
    """各臂共享TFIDF和384维编码，最后八通道仅改变知识条件。"""
    return sparse.hstack((lexical,sparse.csr_matrix(z["embedding"]),sparse.csr_matrix(z[arm])),format="csr")


def cluster_intervals(y: np.ndarray,g: np.ndarray,a: np.ndarray,b: np.ndarray) -> dict[str,Any]:
    """按整段对话聚类，多个策略句不当作相互独立观测。"""
    ids=np.unique(g);ix=np.random.default_rng(42).integers(0,len(ids),(3000,len(ids)))
    def cm(p: np.ndarray) -> np.ndarray:
        return np.array([np.bincount(8*y[g==v]+p.argmax(1)[g==v],minlength=64).reshape(8,8) for v in ids])
    ac,bc=cm(a)[ix].sum(1),cm(b)[ix].sum(1);f1=macro(ac)-macro(bc)
    accuracy=np.trace(ac,axis1=1,axis2=2)/ac.sum((1,2))-np.trace(bc,axis1=1,axis2=2)/bc.sum((1,2))
    return {"dialogues":len(ids),"macro_f1_delta_ci95":np.quantile(f1,[.025,.975]).tolist(),"accuracy_delta_ci95":np.quantile(accuracy,[.025,.975]).tolist(),"scope":"exploratory paired dialogue bootstrap, not clinical confirmation"}


def run(tasks: tuple[str,...] = ("attribution","next_strategy"),lock_only: bool=False) -> None:
    """先完成两个任务所有臂的验证与导出，再一次性打开测试标签评价。"""
    root=OUT/"benchmark";freeze(root/"evaluation_protocol.json",{"source_sha256":sha256(Path(__file__)),"arms":ARMS,"C":[.1,1.],"class_weight":"balanced","selection":"validation macroF1","refit_train_validation":False,"all_test_open_after_all_locks":True})
    records={};models={};vectorizers={}
    for task in tasks:
        folder=root/task
        if (folder/"selection_lock.json").exists():
            cached=json.loads((folder/"selection_lock.json").read_text(encoding="utf-8"));assert sha256(folder/"vectorizer.pkl")==cached["vectorizer_sha256"]
            records[task]=cached["arms"];models[task]={}
            with (folder/"vectorizer.pkl").open("rb") as stream:vectorizers[task]=pickle.load(stream)
            for arm in ARMS:
                assert sha256(folder/(arm+".pkl"))==cached["arms"][arm]["model_sha256"]
                with (folder/(arm+".pkl")).open("rb") as stream:models[task][arm]=pickle.load(stream)
            continue
        train=json.loads((folder/"train.json").read_text(encoding="utf-8"));valid=json.loads((folder/"validation.json").read_text(encoding="utf-8"))
        tr=np.load(folder/"train.npz",allow_pickle=False);va=np.load(folder/"validation.npz",allow_pickle=False)
        vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=30000,min_df=2,sublinear_tf=True);lex=vectorizer.fit_transform([r["text"] for r in train]);vlex=vectorizer.transform([r["text"] for r in valid]);vectorizers[task]=vectorizer
        with (folder/"vectorizer.pkl").open("wb") as stream:pickle.dump(vectorizer,stream)
        records[task]={};models[task]={}
        for arm in ARMS:
            x=matrix(lex,tr,arm);vx=matrix(vlex,va,arm);choices=[]
            for c in (.1,1.):
                model=LinearSVC(C=c,class_weight="balanced",random_state=42,max_iter=3000,tol=1e-3).fit(x,tr["labels"])
                metric=metrics(va["labels"],model.decision_function(vx));choices.append((metric["macro_f1"],c,metric,model))
            _,c,m,model=max(choices,key=lambda r:r[0]);models[task][arm]=model
            file=folder/(arm+".pkl")
            with file.open("wb") as stream:pickle.dump(model,stream)
            records[task][arm]={"C":c,"validation":m,"candidates":[{"C":r[1],"metrics":r[2]} for r in choices],"model_sha256":sha256(file),"features":x.shape[1]}
            print(json.dumps({"locked_task":task,"arm":arm,"C":c,"validation_f1":m["macro_f1"]}),flush=True)
        freeze(folder/"selection_lock.json",{"arms":records[task],"vectorizer_sha256":sha256(folder/"vectorizer.pkl"),"test_labels_used":False,"graph_sha256":sha256(OUT/"knowledge_graph/graph.json")})
    if lock_only:return
    if set(tasks)!={"attribution","next_strategy"}:raise ValueError("两个任务全部锁型后才能读取测试结局")
    # 训练和选型到此结束，下面不再拟合任何图谱、权重、分类器或阈值。
    result={}
    for task in records:
        folder=root/task;rows=json.loads((folder/"test.json").read_text(encoding="utf-8"));test=np.load(folder/"test.npz",allow_pickle=False);lex=vectorizers[task].transform([r["text"] for r in rows]);scores={arm:models[task][arm].decision_function(matrix(lex,test,arm)) for arm in ARMS}
        result[task]={"test":{arm:metrics(test["labels"],p) for arm,p in scores.items()},"comparisons":{arm:cluster_intervals(test["labels"],test["dialogues"],scores["true_graph"],scores[arm]) for arm in ("no_graph","definitions_only","shuffled_graph")},"test_scope":"dialogue-disjoint, not participant/source-disjoint; labels are observed human strategy choices, not unique correct advice"}
        np.savez_compressed(folder/"predictions.npz",truth=test["labels"],groups=test["dialogues"],ids=test["ids"],**scores)
        save(folder/"results.json",result[task])
    save(root/"results.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--lock-only",choices=("attribution","next_strategy"));args=parser.parse_args()
    run((args.lock_only,),True) if args.lock_only else run()
