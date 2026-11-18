"""从封存窗口的真实历史MP4和转录开始重放未来预测，而非输入人工情绪状态。"""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from experiments.round11_common import ROOT,OUTPUT,PREVIOUS,digest,save
from experiments import round11_observed_perception as perception
from experiments.round11_forecast import load,summary,linear_predict
from experiments.round11_sequence_inputs import index_windows,materialize
from experiments.round11_forecast_finish import ForecastGRU,prepare,infer
from round11_sensevoice import build_model,decode,encode


class ObservedRuntime:
    """常驻T+A感知器，输入只有当前转录与音轨。"""
    def __init__(self) -> None:
        torch.manual_seed(42);torch.set_num_threads(1)
        with (OUTPUT/"observed_perception/model.pkl").open("rb") as handle:self.model=pickle.load(handle)
        self.sense,self.kwargs=build_model()
        base=Path(r"D:\ai\models\encoders")
        self.bge_tokenizer=AutoTokenizer.from_pretrained(base/"bge-base-zh-v1.5",local_files_only=True)
        self.bge=AutoModel.from_pretrained(base/"bge-base-zh-v1.5",local_files_only=True).cuda().eval()
        self.aux_tokenizer=AutoTokenizer.from_pretrained(base/"chinese-roberta-wwm-ext",local_files_only=True)
        aux=AutoModel.from_pretrained(base/"chinese-roberta-wwm-ext",local_files_only=True)
        state=torch.load(PREVIOUS/"text/auxiliary_only.pt",map_location="cpu",weights_only=True)
        aux.load_state_dict({k.removeprefix("encoder."):v for k,v in state.items() if k.startswith("encoder.")},strict=True)
        self.aux=aux.cuda().eval();del state;gc.collect();torch.cuda.empty_cache()

    def observe(self,path: Path,text: str) -> tuple[np.ndarray,float]:
        """每条独立编码，无未来对话、标签或跨样本上下文。"""
        start=time.perf_counter();wave,_,valid=decode(path)
        if not valid:raise ValueError("原始历史音轨不可用，拒绝伪造识别观测")
        pooled,query,emo,_=encode(self.sense,self.kwargs,[wave])
        with torch.inference_mode():
            row=self.bge_tokenizer([text],padding=True,truncation=True,max_length=128,return_tensors="pt").to("cuda")
            with torch.autocast("cuda",dtype=torch.float16):hidden=self.bge(**row).last_hidden_state.float()
            bge=hidden[:,0].cpu().numpy()
            row=self.aux_tokenizer([text],padding=True,truncation=True,max_length=96,return_tensors="pt").to("cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):hidden=self.aux(**row).last_hidden_state.float()
            mask=row["attention_mask"].unsqueeze(-1);aux=((hidden*mask).sum(1)/mask.sum(1)).cpu().numpy()
        data={"sample_ids":np.array([path.stem]),"bge_cls":bge,"aux_mean":aux,"sensevoice":pooled,
            "sensevoice_query":query[:,1,:],"sensevoice_emotion":emo[:,1,:],"audio_available":np.array([True])}
        p=perception.predict(self.model,data)[0];torch.cuda.synchronize()
        return p,time.perf_counter()-start


def linear(model_folder: Path,winner: dict[str,Any],window: dict[str,Any],observations: dict[str,np.ndarray]) -> np.ndarray:
    """同一锁定线性阶段配方，仅替换为原始媒体重算的历史识别概率。"""
    cfg=winner["config"];w=dict(window);w["history_ids"]=w["prefix_ids"][-cfg["context"]:]
    if cfg["kind"]=="linear":
        with (model_folder/"candidates"/winner["name"]/"model.pkl").open("rb") as handle:bundle=pickle.load(handle)
        return linear_predict(bundle["fit"],summary([w],observations,cfg["drift"]))[0]
    if cfg["kind"]=="observed_last":return observations[w["history_ids"][-1]]
    if cfg["kind"]=="observed_mean":return np.stack([observations[k] for k in w["history_ids"]]).mean(0)
    value=observations[w["prefix_ids"][0]].copy()
    for key in w["prefix_ids"][1:]:value=.2*observations[key]+.8*value
    return value


def main() -> None:
    """固定第一个合格测试窗口，不按成功与否挑例子；不读取其金标。"""
    output=OUTPUT/"forecast_replay";output.mkdir(parents=True,exist_ok=True)
    if (output/"results.json").exists():return
    setting="h1_lead1.25";folder=OUTPUT/"forecast_final"/setting
    lock=json.loads((folder/"selection_locked.json").read_text("utf-8"))
    records,cached,_=load();windows,_=index_windows(records,20,1,1.25)
    window=next(w for w in windows if w["split"]=="test_inputs")
    history=window["prefix_ids"]
    rows={r["id"]:r for r in map(json.loads,(OUTPUT/"corpus/test_inputs.jsonl").read_text("utf-8").splitlines())}
    if window["target_id"] in history:raise ValueError("回放意外包含未来输入")
    plan={"target_id":window["target_id"],"past_only_ids":history,"rule":"first eligible test window,not success-selected",
        "setting":setting,"selection_sha256":digest(folder/"selection_locked.json"),"code_sha256":digest(Path(__file__)),
        "official_transcripts":True,"test_target_read":False}
    pre=output/"preregister.json"
    if pre.exists():
        if json.loads(pre.read_text("utf-8"))!=plan:raise ValueError("回放计划不符")
    else:save(pre,plan)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();runtime=ObservedRuntime();torch.cuda.synchronize();startup=time.perf_counter()-start
    observations={};timing=[]
    for key in history:
        p,elapsed=runtime.observe(ROOT/f"data/public/emotiontalk_media/clips/{key}.mp4",rows[key]["text"])
        observations[key]=p;timing.append({"id":key,"seconds":elapsed})
    observation_error=max(float(np.abs(observations[k]-cached[k]).max()) for k in history)
    # 各组件的上下文不同，用相同已完成历史构造各自窗口，不传未来时间作为特征。
    c=lock["gru"]["context"];selected_window=next(w for w in index_windows(records,c,1,1.25)[0] if w["target_id"]==window["target_id"])
    materialized=materialize([selected_window],observations,c)
    checkpoint=torch.load(Path(lock["gru"]["path"]),weights_only=False)
    model=ForecastGRU();model.load_state_dict(checkpoint["state"])
    x,lengths,_=prepare(materialized["sequence"],np.array([False]),checkpoint["scaling"])
    forecast_start=time.perf_counter();gru=infer(model,x,lengths)[0]
    oldfolder=OUTPUT/"forecast_linear"/setting
    p_linear=linear(oldfolder,lock["linear"],window,observations)
    mean=np.stack([observations[k] for k in window["history_ids"]]).mean(0)
    p=sum(w*q for w,q in zip(lock["vote"]["weights"],(p_linear,gru,mean)));forecast_seconds=time.perf_counter()-forecast_start
    with np.load(folder/"locked_predictions.npz",allow_pickle=False) as z:
        i=list(z["sample_ids"]).index(window["target_id"]);reference=z["selected"][i]
    error=float(np.abs(p-reference).max())
    if observation_error>1e-3 or error>1e-3:raise ValueError("原媒体重放与冻结预测偏差超容差")
    # 同一历史单条再算一次，验证请求顺序不会改变状态。
    again,elapsed=runtime.observe(ROOT/f"data/public/emotiontalk_media/clips/{history[-1]}.mp4",rows[history[-1]]["text"])
    repeat_error=float(np.abs(again-observations[history[-1]]).max())
    save(output/"results.json",{**plan,"history_size":len(history),"cutoff":window["cutoff"],"target_start":window["target_start"],
        "actual_lead_seconds":window["lead_seconds"],"probabilities":p.tolist(),"locked_probabilities":reference.tolist(),
        "observation_max_probability_error":observation_error,"forecast_max_probability_error":error,"repeat_observation_max_error":repeat_error,
        "model_startup_seconds":startup,"past_observation_timings":timing,"forecast_head_seconds":forecast_seconds,
        "warm_last_observation_seconds":elapsed,"cuda_peak_allocated_MiB":torch.cuda.max_memory_allocated()/1024**2,
        "limits":"single predetermined replay;official transcript assumed;does not establish latency p95 or clinical long-term forecasting"})
    print(json.dumps({"raw_forecast_replay":True,"target":window["target_id"],"history_size":len(history),"max_error":error}),flush=True)


if __name__=="__main__":main()
