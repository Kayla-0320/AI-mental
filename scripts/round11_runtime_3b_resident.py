"""实测3B解码器与感知编码器常驻的原媒体请求，保持冻结表示数值。"""
from __future__ import annotations
import gc
import json
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
import torch
from experiments.round11_common import ROOT,OUTPUT,digest,save
from experiments.round11_final import predict
from round11_runtime import RawRuntime
from round11_language_runtime import build,encode


def main() -> None:
    """固定同12片段，常驻路径若失败则保留失败而由隔离路径接续。"""
    folder=OUTPUT/"runtime_staged";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"3b.json").exists():return
    jobs=json.loads((OUTPUT/"runtime_benchmark/preregister.json").read_text("utf-8"))["IDs"]
    lookup={}
    for split in ("train","selection"):
        with np.load(ROOT/f"_国赛优化/round5/text_features/{split}.npz",allow_pickle=False) as z:lookup.update(zip(z["sample_ids"].astype(str),z["raw_text"].astype(str)))
    policy="co-resident3B decoder plus frozen perception encoders;all requests re-decode/re-encode raw media;three original output rows;no prior feature cache"
    save(folder/"3b_resident_plan.json",{"policy":policy,"IDs":jobs,"n":12,"selection":"same fixed cohort;no accuracy or timing-based case selection",
        "code_sha256":digest(Path(__file__)),"language_runtime_sha256":digest(Path(__file__).with_name("round11_language_runtime.py")),
        "equivalence_sha256":digest(OUTPUT/"validation/language_runtime_3b_equivalence.json")})
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();runtime=RawRuntime();gc.collect();torch.cuda.empty_cache()
    model,tokenizer,head,metadata=build("3b");gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();startup=time.perf_counter()-start
    with (OUTPUT/"final_models/3b/model.pkl").open("rb") as handle:bundle=pickle.load(handle)
    rows=[];probabilities=[]
    for key in jobs:
        start=time.perf_counter();_,base,data=runtime.infer(ROOT/f"data/public/ch_sims_v2/supervised/{key}.mp4",lookup[key])
        t=time.perf_counter();data.update(encode(model,tokenizer,head,[lookup[key]]));torch.cuda.synchronize();language_seconds=time.perf_counter()-t
        p=predict(bundle,data);elapsed=time.perf_counter()-start
        rows.append({"id":key,"total_seconds":elapsed,"raw_base_seconds":base["total"],"LLM_forward_seconds":language_seconds})
        probabilities.append(p);print(json.dumps({"mode":"3b_resident",**rows[-1]}),flush=True)
        gc.collect();torch.cuda.empty_cache()
    values=np.array([r["total_seconds"] for r in rows]);warm=values[1:]
    result={"policy":policy,"n_requests":len(rows),"model_startup_seconds":startup,"first_request_seconds":startup+float(values[0]),
        "warm_policy_p50_seconds":float(np.quantile(warm,.5)),"warm_policy_p95_seconds":float(np.quantile(warm,.95)),
        "peak_cuda_allocated_MiB":torch.cuda.max_memory_allocated()/1024**2,"rows":rows,"language_metadata":metadata,
        "limits":"12 clips on one shared laptop;OS cache not flushed;resident warm requests unlike7B isolated loading;official transcription assumed"}
    np.savez_compressed(folder/"3b_predictions.npz",sample_ids=np.array(jobs),probabilities=np.concatenate(probabilities,axis=1),seeds=np.arange(42,47))
    save(folder/"3b.json",result)


if __name__=="__main__":main()
