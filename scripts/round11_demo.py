"""答辩可直接运行的原始片段推理入口，默认固定第一个外部输入而非挑成功例。"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import torch
from experiments.round11_common import ROOT,OUTPUT,save,digest
from round11_runtime import RawRuntime


def main() -> None:
    """原片段与当前转录进入模型；没有人工情绪标签或缓存情感向量输入。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("--clip",type=Path);parser.add_argument("--text")
    parser.add_argument("--mode",choices=("light","3b"),default="light")
    parser.add_argument("--output",type=Path,default=OUTPUT/"cases/live_raw_media_demo.json")
    args=parser.parse_args()
    if args.clip is None and args.text is None:
        row=json.loads((OUTPUT/"corpus/test_inputs.jsonl").read_text("utf-8").splitlines()[0])
        path=ROOT/f"data/public/emotiontalk_media/clips/{row['id']}.mp4";text=row["text"]
        selection="first external input;not selected by correctness;no test gold supplied"
    elif args.clip is not None and args.text is not None:
        path=args.clip;text=args.text;selection="explicit user-supplied media and transcription"
    else:parser.error("指定片段时需要同时提供--clip和--text")
    if not path.is_file():raise FileNotFoundError(path)
    if not text.strip():raise ValueError("此原媒体演示入口需要非空的当前转录")
    start=time.perf_counter();runtime=RawRuntime()
    if args.mode=="3b":
        from round11_language_runtime import build,encode
        from experiments.round11_final import predict
        language,tokenizer,head,_=build("3b")
        with (OUTPUT/"final_models/3b/model.pkl").open("rb") as handle:bundle=pickle.load(handle)
    torch.cuda.synchronize();startup=time.perf_counter()-start
    request_start=time.perf_counter();p,timings,data=runtime.infer(path,text)
    if args.mode=="3b":
        begin=time.perf_counter();data.update(encode(language,tokenizer,head,[text]));p=predict(bundle,data);torch.cuda.synchronize()
        timings["additional_3b_language_and_fusion"]=time.perf_counter()-begin;timings["total"]=time.perf_counter()-request_start
    ensemble=p[:,0].mean(0);classes=["负向","中性","正向"]
    result={"sample_id":path.stem,"transcription":text,"model":f"locked Round11 {args.mode};five meta-seed probability ensemble",
        "task":"current M3;not future M7","classes":classes,"predicted_label":classes[int(ensemble.argmax())],
        "ensemble_probabilities":ensemble.tolist(),"per_meta_seed_probabilities":p[:,0].tolist(),
        "available_T_A_V":data["modality_mask"][0].tolist(),"face_coverage":float(data["coverage"][0]),
        "model_load_seconds":startup,"raw_inference_seconds":timings,"selection":selection,
        "model_sha256":digest(OUTPUT/f"final_models/{args.mode}/model.pkl"),
        "scope":"functional raw-media demo;no gold used;not a new accuracy/generalization estimate;official/current transcription assumed"}
    save(args.output,result,immutable=False)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
