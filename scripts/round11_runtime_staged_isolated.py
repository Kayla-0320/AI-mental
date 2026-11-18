"""8GB设备隔离进程分阶段请求，实际计入初始化、媒体处理和IPC。"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
from experiments.round11_common import ROOT,OUTPUT,digest,save


def execute(kind: str,folder: Path,mode: str) -> dict[str,Any]:
    """父进程不加载Torch，上一阶段退出后才启动下一阶段。"""
    worker=Path(__file__).with_name("round11_runtime_worker.py")
    start=time.perf_counter()
    with (folder/f"{kind}.log").open("w",encoding="utf-8") as handle:
        result=subprocess.run([sys.executable,str(worker),kind,"--folder",str(folder),"--mode",mode],stdout=handle,stderr=subprocess.STDOUT)
    elapsed=time.perf_counter()-start
    if result.returncode:
        save(folder/f"{kind}_failure.json",{"returncode":result.returncode,"seconds":elapsed},immutable=False)
        raise RuntimeError(f"{mode}/{kind}隔离进程失败，退出码{result.returncode}，详见{folder}")
    metrics=json.loads((folder/f"{kind}_metrics.json").read_text("utf-8"));metrics["process_wall_seconds"]=elapsed
    return metrics


def main() -> None:
    """每请求重新提取当前媒体和文本，上一阶段结果只作本次请求IPC。"""
    folder=OUTPUT/"runtime_staged";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"results.json").exists():return
    jobs=json.loads((OUTPUT/"runtime_benchmark/preregister.json").read_text("utf-8"))["IDs"]
    lookup={}
    for split in ("train","selection"):
        with np.load(ROOT/f"_国赛优化/round5/text_features/{split}.npz",allow_pickle=False) as z:lookup.update(zip(z["sample_ids"].astype(str),z["raw_text"].astype(str)))
    policy="fresh base-encoder process from raw media;exit;fresh LLM process from current text;exit;CPU fusion process;all process/model loads and within-request NPZ IPC timed"
    plan={"modes":["7b","3b"],"IDs":jobs,"n_requests_each":len(jobs),"batch":1,"runtime_policy":policy,
        "mode_policy":{"7b":"isolated decoder plus3 output rows","3b":"co-resident attempted on same fixed cohort;isolated fallback only if residency cannot complete"},
        "warm_definition":"OS cache not flushed;fresh processes and GPU contexts each request;not a resident full-model SLA",
        "code_sha256":digest(Path(__file__)),"worker_sha256":digest(Path(__file__).with_name("round11_runtime_worker.py")),
        "base_runtime_sha256":digest(Path(__file__).with_name("round11_runtime.py")),
        "language_runtime_sha256":digest(Path(__file__).with_name("round11_language_runtime.py")),
        "amendment":"full-vocabulary7B loads exited natively before first complete request;decoder-only retains original body and3 exact frozen output rows;equivalence checked against frozen features;no model selection"}
    pre=folder/"preregister.json"
    if pre.exists():
        if json.loads(pre.read_text("utf-8"))!=plan:raise ValueError("隔离进程基准计划不符")
    else:save(pre,plan)
    results={}
    for mode in ("7b","3b"):
        complete=folder/f"{mode}.json"
        if complete.exists():results[mode]=json.loads(complete.read_text("utf-8"));continue
        rows=[];probabilities=[]
        for i,key in enumerate(jobs):
            request=folder/"requests"/f"{mode}_{i:02d}";request.mkdir(parents=True,exist_ok=True)
            save(request/"request.json",{"id":key,"text":lookup[key],"clip":str(ROOT/f"data/public/ch_sims_v2/supervised/{key}.mp4")},immutable=False)
            start=time.perf_counter();base=execute("base",request,mode);language=execute("llm",request,mode);fusion=execute("fusion",request,mode)
            elapsed=time.perf_counter()-start
            with np.load(request/"prediction.npz",allow_pickle=False) as z:p=z["probabilities"]
            rows.append({"id":key,"total_seconds":elapsed,"base_load_and_raw_inference_seconds":base["process_wall_seconds"],
                "LLM_load_and_inference_seconds":language["process_wall_seconds"],"fusion_process_seconds":fusion["process_wall_seconds"],
                "cuda_peak_allocated_MiB":max(base["cuda_peak_allocated_MiB"],language["cuda_peak_allocated_MiB"]),
                "cuda_peak_reserved_MiB":max(base["cuda_peak_reserved_MiB"],language["cuda_peak_reserved_MiB"]),
                "fresh_raw_features_sha256":digest(request/"base.npz")})
            probabilities.append(p);print(json.dumps({"mode":mode,**rows[-1]}),flush=True)
        values=np.array([r["total_seconds"] for r in rows]);warm=values[1:]
        result={"policy":policy,"n_requests":len(rows),"first_request_seconds":float(values[0]),"warm_policy_p50_seconds":float(np.quantile(warm,.5)),
            "warm_policy_p95_seconds":float(np.quantile(warm,.95)),"peak_cuda_allocated_MiB":max(r["cuda_peak_allocated_MiB"] for r in rows),"rows":rows,
            "limits":plan["warm_definition"]+";within-request IPC is freshly generated and timed;official transcription assumed;includes redundant light fusion"}
        np.savez_compressed(folder/f"{mode}_predictions.npz",sample_ids=np.array(jobs),probabilities=np.concatenate(probabilities,axis=1),seeds=np.arange(42,47))
        save(complete,result);results[mode]=result
    save(folder/"results.json",results)


if __name__=="__main__":main()
