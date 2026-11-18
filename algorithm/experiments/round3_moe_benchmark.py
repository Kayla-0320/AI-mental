"""同一权重稠密参考与真实稀疏分派的算子计数、延迟及文字编码一致性。"""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
from typing import Any
for _k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ.setdefault(_k,"1")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from perception.knowledge_moe.features import FrozenCLIPEncoder,save_json,FeatureStandardizer
from perception.knowledge_moe.model import QualityAwareMoE
from perception.knowledge_moe.sparse import SparseQualityAwareMoE


def benchmark(bundle:Path,artifacts:Path,model_dir:Path,samples:Path) -> dict[str,Any]:
    """只用训练特征测量头部延迟，明确不包括编码器和网络。"""
    torch.manual_seed(42);torch.set_num_threads(1)
    meta=json.loads((artifacts/"sparse/manifest.json").read_text(encoding="utf-8"))
    sparse=SparseQualityAwareMoE(meta["dimensions"],len(meta["class_names"]),meta["hidden"],top_k=2).eval()
    dense=QualityAwareMoE(meta["dimensions"],len(meta["class_names"]),meta["hidden"],top_k=2).eval()
    state=torch.load(artifacts/"sparse/weights.pt",map_location="cpu",weights_only=True);sparse.load_state_dict(state);dense.load_state_dict(state)
    scaler=FeatureStandardizer();scaler.parameters=meta["standardizer"]
    with np.load(bundle/"train.npz",allow_pickle=False) as z:d={k:z[k] for k in z.files}
    names=list(meta["dimensions"]);normalized=scaler.transform({n:d[n] for n in names},d["modality_mask"])
    output={"kind":"same_weight_dense_top2_reference_vs_actual_sparse_top2","device":"cpu","threads":1,
            "timing_scope":"fusion_head_only_no_encoder_no_network","batches":{},"limitations":["小型融合头实际稀疏分派未必更快","不推断整体VLM延迟下降"]}
    for batch_size in [1,64,256]:
        x={n:torch.tensor(normalized[n][:batch_size]) for n in names};m=torch.tensor(d["modality_mask"][:batch_size]);q=torch.tensor(d["quality"][:batch_size])
        with torch.inference_mode():
            a,_,_=sparse(x,m,q);b,_,_=dense(x,m,q)
        if not torch.allclose(a,b,atol=1e-5):raise ValueError("稀疏分派与相同权重的稠密top2参考不一致")
        counts=list(sparse.last_dispatch_counts)
        timings={}
        for name,model in [("dense_reference",dense),("actual_sparse",sparse)]:
            times=[]
            with torch.inference_mode():
                for i in range(110):
                    start=time.perf_counter();model(x,m,q);elapsed=time.perf_counter()-start
                    if i>=10:times.append(elapsed*1000)
            timings[name]={"p50_ms":float(np.quantile(times,.5)),"p95_ms":float(np.quantile(times,.95)),"repetitions":100}
        output["batches"][str(batch_size)]={"equivalent":True,"sparse_expert_row_evaluations":sum(counts),
                                              "dense_expert_row_evaluations":batch_size*len(dense.expert_names),
                                              "per_expert_actual_rows":counts,"timings":timings}
    # 两种文字编码路径实际前向，避免仅凭接口名认定训练/实时特征一致。
    import cv2
    from PIL import Image
    sample_rows=json.loads(samples.read_text(encoding="utf-8"))["samples"]
    images=[];texts=[]
    for row in sample_rows:
        cap=cv2.VideoCapture(row["path"]);ok,frame=cap.read();cap.release()
        if not ok:raise ValueError("原片无法解码")
        images.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)));texts.append(row["annotation"]["text"])
    encoder=FrozenCLIPEncoder(model_dir,"cpu")
    joint=encoder.encode(images,texts)["text"];text=encoder.encode_text(texts)
    diff=float(np.max(np.abs(joint-text)))
    if not np.allclose(joint,text,atol=1e-5):raise ValueError("独立文字前向与训练图文前向不一致")
    output["text_encoding_equivalence"]={"real_samples":len(texts),"max_absolute_difference":diff,"atol":1e-5,
                                           "fingerprint":encoder.fingerprint,"norms":np.linalg.norm(text,axis=1).tolist()}
    save_json(artifacts/"sparse_latency_and_text_equivalence.json",output)
    print(json.dumps(output,ensure_ascii=False));return output


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["bundle","artifacts","model-dir","samples"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();benchmark(a.bundle,a.artifacts,a.model_dir,a.samples)
