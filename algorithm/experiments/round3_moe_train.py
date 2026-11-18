"""先锁定全部验证选择，再一次性评测新来源组测试集的九臂真实VLM消融。"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from experiments.round2_moe_train import seed_all, load_split, tensors, forward, metrics, temperature_fit, grouped_delta_ci
from perception.knowledge_moe.features import FeatureStandardizer, FrozenCLIPEncoder, save_json, sha256_file
from perception.knowledge_moe.knowledge import KnowledgeIndex
from perception.knowledge_moe.model import QualityAwareMoE, ConcatenationHead
from perception.knowledge_moe.sparse import SparseQualityAwareMoE


def attested_graph(output: Path) -> Path:
    """仅记录用户背书，不编造专家身份或实际审核日期。"""
    seed = Path(__file__).resolve().parents[1] / "perception/knowledge_moe/knowledge_seed.json"
    graph = json.loads(seed.read_text(encoding="utf-8"))
    for node in graph["nodes"]:
        node.update(review_status="user_attested", reviewer=None, reviewed_at=None)
    payload = json.dumps(graph["nodes"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    graph["status"] = "user_attested_expert_review_not_independently_verified"
    graph["review_attestation"] = {"type": "user_confirmation_of_expert_review", "source_statement": "专家审核已经过关",
                                   "confirmation_date": "2026-11-15", "expert_name": None, "expert_review_date": None,
                                   "nodes_sha256": hashlib.sha256(payload).hexdigest(),
                                   "scope": "用户确认该知识内容已通过专家审核；未取得专业人员签署文件，不等同独立核验。"}
    path = output / "user_attested_knowledge.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != graph:
        raise ValueError("已冻结知识版本不同，不覆盖")
    save_json(path, graph)
    return path


def model_for(arm: str, dimensions: dict[str, int], classes: int, hidden: int, kd: int) -> nn.Module:
    """构造共享表征和训练预算的模型臂。"""
    if arm.startswith("sparse"):
        return SparseQualityAwareMoE(dimensions, classes, hidden, kd, top_k=2)
    return (QualityAwareMoE if arm.startswith("moe") else ConcatenationHead)(dimensions, classes, hidden, kd)


def text_only(batch: tuple) -> tuple:
    """沿用同一文本特征，只关闭声音和视觉；不让模态缺失混入其它校准条件。"""
    x,mask,q,kg,km,y=batch
    new=mask.clone();new[:]=False
    if "text" in x:
        j=list(x).index("text");new[:,j]=mask[:,j]
    return x,new,q,kg,km,y


def run(bundle: Path, output: Path, knowledge_encoder: Path) -> dict[str, Any]:
    """完成九臂×相同两项隐藏维搜索；测试文件不参与模型选择。"""
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    arms = ["text", "audio", "visual", "concat", "concat_knowledge", "moe", "moe_knowledge", "sparse", "sparse_knowledge"]
    protocol = {"schema_version": "round3_model_preregister.v1", "seed": 42, "arms": arms, "hidden_grid": [32, 64],
                "epochs_max": 80, "patience": 12, "lr": .001, "weight_decay": .001, "batch": 64,
                "selection": "macro_f1_then_nll_on_selection_only", "calibration": "one_temperature_NLL_on_calibration_only",
                "live_text_selection": "separate_text_only_macro_f1_then_nll_on_selection_only",
                "live_text_calibration": "separate_temperature_on_calibration_text_present_audio_visual_missing",
                "knowledge_geometry":"32_relation_hash_plus_512_frozen_chinese_clip_node_semantics_unit_balanced",
                "test_read_policy": "only_after_all_arm_choices_locked", "dataset_manifest_sha256": sha256_file(bundle / "metadata.json"),
                "frozen_split_sha256": meta["frozen_split_sha256"]}
    protocol_path = output / "preregister.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("预注册配置不同，不得覆盖")
    if (output / "experiment.json").exists():
        raise ValueError("本测试已评估，拒绝静默重跑或改配方")
    save_json(protocol_path, protocol)
    data = {s: load_split(bundle / f"{s}.npz") for s in ["train", "selection", "calibration"]}
    # 测试ID/来源组仅从预先冻结的划分核验，不在选择阶段加载测试特征或标签。
    plan = json.loads((bundle / "frozen_split.json").read_text(encoding="utf-8"))
    source_sets = {s: {r["source_group"] for r in plan["samples"] if r["split"] == s} for s in ["train", "selection", "calibration", "test"]}
    for i, a in enumerate(source_sets):
        for b in list(source_sets)[i+1:]:
            if source_sets[a] & source_sets[b]:
                raise ValueError("来源组交叉")
    graph_path = attested_graph(output)
    index = KnowledgeIndex(graph_path, allow_user_attested=True)
    encoder=FrozenCLIPEncoder(knowledge_encoder,"cpu")
    if encoder.fingerprint!=meta["live_text_encoder_fingerprint"]:
        raise ValueError("知识与数据编码器不同")
    node_ids=[e["id"] for e in index.entries]
    vectors=encoder.encode_text([e["text"] for e in index.entries])
    np.savez_compressed(output/"knowledge_vectors.npz",node_ids=np.array(node_ids),vectors=vectors)
    index.attach_semantic_vectors(dict(zip(node_ids,vectors)))
    del encoder
    report: dict[str, Any] = {"schema_version": "round3_vlm_ablation.v1", "protocol": protocol,
                              "task": meta["task"], "class_names": meta["class_names"], "arms": {},
                              "knowledge_attestation": "user_confirmation_not_independent_professional_signature",
                              "limitations": ["仅seed42和一个来源视频划分", "来源视频不等同演员或影片", "参数量不相同，搜索项数相同",
                                              "音频为真实音轨手工MFCC/韵律，不是预训练语音模型", "三帧视觉池化不是完整时序视频理解"]}
    chosen_models = {}
    for arm in arms:
        names = [arm] if arm in ("text", "audio", "visual") else ["text", "audio", "visual"]
        dims = {name: data["train"][name].shape[1] for name in names}
        cols = [["text", "audio", "visual"].index(n) for n in names]
        scaler = FeatureStandardizer().fit({n: data["train"][n] for n in names}, data["train"]["modality_mask"][:, cols])
        batches = {s: tensors(d, names, scaler, index) for s, d in data.items()}
        kd = index.dimensions if "knowledge" in arm else 0
        candidates = []
        start = time.monotonic()
        for hidden in protocol["hidden_grid"]:
            seed_all();model = model_for(arm, dims, len(meta["class_names"]), hidden, kd)
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
            y = batches["train"][-1];c = len(meta["class_names"])
            counts = torch.bincount(y, minlength=c).clamp(min=1)
            criterion = nn.CrossEntropyLoss(weight=len(y)/(c*counts.float()))
            best_key = (-1., -float("inf"));best_state = None;best_epoch = 0;stale = 0
            for epoch in range(1, protocol["epochs_max"]+1):
                model.train();order = torch.randperm(len(y))
                for lo in range(0, len(y), 64):
                    ix = order[lo:lo+64];logits, _, balance, valid = forward(model, batches["train"], ix, augment=True)
                    if not valid.any():
                        continue
                    loss = criterion(logits, y[ix][valid]) + .01*balance
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(), 5.);opt.step()
                model.eval()
                with torch.inference_mode():
                    logits, _, _, valid = forward(model, batches["selection"], torch.arange(len(data["selection"]["labels"])))
                score = metrics(data["selection"]["labels"][valid.numpy()], logits.numpy(), c)
                key = (score["macro_f1"], -score["nll"])
                if key > best_key:
                    best_key, best_state, best_epoch, stale = key, copy.deepcopy(model.state_dict()), epoch, 0
                else:
                    stale += 1
                    if stale >= protocol["patience"]:
                        break
            if best_state is None:
                raise ValueError("未得到有效模型")
            candidates.append({"hidden": hidden, "key": best_key, "weights": best_state, "epoch": best_epoch})
        best = max(candidates, key=lambda x:x["key"])
        model = model_for(arm, dims, len(meta["class_names"]), best["hidden"], kd)
        model.load_state_dict(best["weights"]);model.eval()
        with torch.inference_mode():
            logits, _, _, valid = forward(model, batches["calibration"], torch.arange(len(data["calibration"]["labels"])))
        temperature = temperature_fit(data["calibration"]["labels"][valid.numpy()], logits.numpy(), len(meta["class_names"]))
        text_calibration = {"status":"not_applicable"}
        text_selection = None
        if "text" in names:
            with torch.inference_mode():
                tl,_,_,tv=forward(model,text_only(batches["calibration"]),torch.arange(len(data["calibration"]["labels"])))
                sl,_,_,sv=forward(model,text_only(batches["selection"]),torch.arange(len(data["selection"]["labels"])))
            text_t=temperature_fit(data["calibration"]["labels"][tv.numpy()],tl.numpy(),len(meta["class_names"]))
            text_calibration={"status":"calibrated","fit_split":"calibration","n":int(tv.sum()),"temperature":text_t,
                              "condition":"text_present_audio_visual_missing"}
            text_selection=metrics(data["selection"]["labels"][sv.numpy()],sl.numpy(),len(meta["class_names"]))
        arm_dir = output / arm;arm_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), arm_dir / "weights.pt")
        architecture = "sparse_moe" if arm.startswith("sparse") else ("moe" if arm.startswith("moe") else "concat")
        manifest = {"schema_version": "knowledge_moe.v1", "task": meta["task"], "class_names": meta["class_names"],
                    "dimensions": dims, "hidden": best["hidden"], "architecture": architecture, "top_k": 2 if architecture == "sparse_moe" else 0,
                    "knowledge_dim": kd, "knowledge_sha256": index.sha256 if kd else None,
                    "knowledge_review_mode": index.review_mode if kd else "disabled", "checkpoint_sha256": sha256_file(arm_dir/"weights.pt"),
                    "knowledge_vectors_sha256":sha256_file(output/"knowledge_vectors.npz") if kd else None,
                    "knowledge_vector_encoder_fingerprint":meta["live_text_encoder_fingerprint"] if kd else None,
                    "encoder_fingerprints": {n:meta["encoder_fingerprints"][n] for n in names}, "live_text_encoder_fingerprint":meta["live_text_encoder_fingerprint"],
                    "standardizer": scaler.parameters, "temperature": temperature,
                    "calibration": {"status":"calibrated","fit_split":"calibration","n":int(valid.sum())},
                    "text_only_calibration":text_calibration,
                    "training_manifest_sha256": sha256_file(bundle/"metadata.json"), "split_protocol_sha256":sha256_file(bundle/"frozen_split.json"), "seed":42}
        save_json(arm_dir/"manifest.json", manifest)
        if kd:
            shutil.copyfile(graph_path, arm_dir/"knowledge.json")
            shutil.copyfile(output/"knowledge_vectors.npz",arm_dir/"knowledge_vectors.npz")
        report["arms"][arm] = {"hidden":best["hidden"], "best_epoch":best["epoch"], "selection_macro_f1":best["key"][0],
                                 "selection_nll":-best["key"][1], "parameters":sum(p.numel() for p in model.parameters()),
                                 "temperature":temperature,"training_seconds":time.monotonic()-start,
                                 "text_only_selection":text_selection,"text_only_calibration":text_calibration,
                                 "candidates":[{k:v for k,v in x.items() if k!="weights"} for x in candidates]}
        chosen_models[arm] = (model, scaler, names)
        print(json.dumps({"stage":"selection_locked","arm":arm,"hidden":best["hidden"],"selection_f1":best["key"][0]}),flush=True)
    winner = max(arms, key=lambda a:(report["arms"][a]["selection_macro_f1"],-report["arms"][a]["selection_nll"]))
    report["deployment_selection"] = {"arm":winner,"rule":"selection_macro_f1_then_nll","test_not_used":True}
    live_arms=[a for a in arms if report["arms"][a]["text_only_selection"] is not None]
    live_winner=max(live_arms,key=lambda a:(report["arms"][a]["text_only_selection"]["macro_f1"],-report["arms"][a]["text_only_selection"]["nll"]))
    report["live_text_deployment_selection"]={"arm":live_winner,"rule":"text_only_selection_macro_f1_then_nll","test_not_used":True}
    save_json(output/"selection_locked.json", report)
    # 以上全部选择完成后才读取测试特征/标签。
    test = load_split(bundle/"test.npz");predictions = {}
    for arm in arms:
        model, scaler, names = chosen_models[arm];batch = tensors(test,names,scaler,index)
        conditions = {}
        for scenario in ["clean","without_text","without_audio","without_visual","random_missing","noisy_visual_known_quality"]:
            with torch.inference_mode():
                logits,gates,_,valid = forward(model,batch,torch.arange(len(test["labels"])),scenario=scenario)
            y = test["labels"][valid.numpy()]
            result = metrics(y,logits.numpy(),len(meta["class_names"]),report["arms"][arm]["temperature"])
            result["coverage"] = float(valid.float().mean());result["gate_weights"] = gates.mean(dim=0).tolist() if len(gates) else []
            conditions[scenario] = result
            if scenario=="clean":
                report["arms"][arm]["test_raw"] = metrics(y,logits.numpy(),len(meta["class_names"]))
                predictions[arm] = logits.argmax(dim=1).numpy()
                np.savez_compressed(output/arm/"test_predictions.npz",logits=logits.numpy(),labels=y,sample_ids=test["sample_ids"][valid.numpy()])
        report["arms"][arm]["conditions"] = conditions
        if "text" in names:
            with torch.inference_mode():
                tl,_,_,tv=forward(model,text_only(batch),torch.arange(len(test["labels"])))
            report["arms"][arm]["text_only_test"]=metrics(test["labels"][tv.numpy()],tl.numpy(),len(meta["class_names"]),report["arms"][arm]["text_only_calibration"]["temperature"])
        print(json.dumps({"stage":"final_test","arm":arm,"f1":conditions["clean"]["macro_f1"],"ece":conditions["clean"]["ece"]}),flush=True)
    report["group_bootstrap_f1"] = {}
    for a,b in [("moe","concat"),("sparse","concat"),("sparse","moe"),("moe_knowledge","moe"),("sparse_knowledge","sparse")]:
        if len(predictions[a])==len(test["labels"])==len(predictions[b]):
            report["group_bootstrap_f1"][f"{a}_minus_{b}"] = {"difference":report["arms"][a]["conditions"]["clean"]["macro_f1"]-report["arms"][b]["conditions"]["clean"]["macro_f1"],
                                                                "ci95":grouped_delta_ci(test["labels"],predictions[a],predictions[b],test["source_groups"],len(meta["class_names"])),"replicates":1000}
    shutil.copytree(output/winner,output/"selected",dirs_exist_ok=True)
    shutil.copytree(output/live_winner,output/"selected_live_text",dirs_exist_ok=True)
    report["counts"] = {s:sum(r["split"]==s for r in plan["samples"]) for s in source_sets}
    report["groups"] = {s:len(gs) for s,gs in source_sets.items()}
    save_json(output/"experiment.json", report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--bundle",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--knowledge-encoder",type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.output,args.knowledge_encoder)
