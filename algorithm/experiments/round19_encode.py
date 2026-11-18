"""ESConv仅按既有人工策略标注建任务，冻结编码与知识通道不接触标签。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import numpy as np
from experiments.round14_common import freeze,save,sha256
from experiments.round19_prepare import OUT,DATA,MODEL
from experiments.round19_graph import STRATEGIES,validate


def normalized(text: str) -> str:
    """仅用于检查重复输入，不重写原对话或人工标签。"""
    return "".join(c for c in text.casefold() if c.isalnum())


def make_tasks(dialogues: list[dict[str,Any]]) -> tuple[dict[str,dict[str,list[dict]]],dict[str,Any]]:
    """同一对话只能在一组；下一策略输入严格止于目标回复之前。"""
    mapping={s[1]:i for i,s in enumerate(STRATEGIES)};unique={}
    for d in dialogues:
        raw=json.dumps([(u["speaker"],u["content"]) for u in d["dialog"]],ensure_ascii=False,separators=(",",":"))
        digest=hashlib.sha256(raw.encode()).hexdigest()
        if digest not in unique:unique[digest]=d
    groups=sorted(unique,key=lambda h:hashlib.sha256(("19|42|"+h).encode()).digest());n=len(groups);cuts=[int(n*.7),int(n*.85)]
    partitions={"train":groups[:cuts[0]],"validation":groups[cuts[0]:cuts[1]],"test":groups[cuts[1]:]};result={t:{s:[] for s in partitions} for t in ("attribution","next_strategy")};audit={"raw_dialogues":len(dialogues),"unique_dialogues":n,"partitions":partitions,"duplicate_inputs_removed":{}}
    for split,ids in partitions.items():
        for digest in ids:
            turns=unique[digest]["dialog"]
            for j,u in enumerate(turns):
                label=u.get("annotation",{}).get("strategy")
                if u["speaker"]!="supporter" or label not in mapping:continue
                common={"id":f"{digest}:{j}","dialogue":digest,"target_index":j,"label":mapping[label],"original_strategy":label}
                result["attribution"][split].append(dict(common,text=u["content"],input_max_index=j))
                if j and turns[j-1]["speaker"]=="seeker":
                    history=turns[max(0,j-6):j];text="\n".join(x["speaker"]+": "+x["content"] for x in history)[-1800:]
                    result["next_strategy"][split].append(dict(common,text=text,input_max_index=j-1))
    for task,parts in result.items():
        seen=set();removed={}
        for split,rows in parts.items():
            kept=[r for r in rows if normalized(r["text"]) and normalized(r["text"]) not in seen]
            removed[split]=len(rows)-len(kept);seen.update(normalized(r["text"]) for r in kept);parts[split]=kept
        audit["duplicate_inputs_removed"][task]=removed
    return result,audit


class FrozenEncoder:
    """按发布模型卡进行mask均值池化；不微调，不用它生成评审标签。"""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel,AutoTokenizer
        lock=json.loads((OUT/"encoder_lock.json").read_text(encoding="utf-8"))
        if any(sha256(MODEL/n)!=v for n,v in lock["files"].items()):raise ValueError("冻结编码器摘要不符")
        torch.set_num_threads(1);self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);self.model=AutoModel.from_pretrained(MODEL,local_files_only=True).to(self.device).eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)

    def encode(self,texts: list[str],*,context: bool=False) -> np.ndarray:
        """对上下文保留最后256token；回复与概念按通常右侧截断处理。"""
        import torch
        self.tokenizer.truncation_side="left" if context else "right";blocks=[]
        for start in range(0,len(texts),64):
            batch=self.tokenizer(texts[start:start+64],padding=True,truncation=True,max_length=256,return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output=self.model(**batch).last_hidden_state;mask=batch["attention_mask"].unsqueeze(-1).to(output.dtype)
                vectors=(output*mask).sum(1)/mask.sum(1).clamp(min=1);vectors=torch.nn.functional.normalize(vectors,p=2,dim=1)
            blocks.append(vectors.cpu().numpy())
        return np.concatenate(blocks).astype(np.float32)


def knowledge_channels(embedding: np.ndarray,nodes: np.ndarray,graph: dict[str,Any],node_ids: list[str]) -> dict[str,np.ndarray]:
    """标签不可进入图谱计算；乱图只置换facet关系，保持所有文本、维数和边度数。"""
    lookup={n:i for i,n in enumerate(node_ids)};strategies=["strategy_"+s[0] for s in STRATEGIES]
    pairs=[[e["object"] for e in graph["edges"] if e["subject"]==s and e["predicate"]=="has_facet"] for s in strategies]
    flat=sum(pairs,[]);permuted=np.array(flat)[np.random.default_rng(42).permutation(len(flat))]
    if any(len(p)!=2 for p in pairs):raise ValueError("策略facet结构不符")
    own=embedding@nodes[[lookup[s] for s in strategies]].T
    scores=embedding@nodes.T
    true=np.column_stack([scores[:,[lookup[p] for p in pair]].max(1) for pair in pairs])
    shuffled=np.column_stack([scores[:,[lookup[p] for p in permuted[2*i:2*i+2]]].max(1) for i in range(8)])
    return {"no_graph":np.zeros_like(own),"definitions_only":own,"true_graph":.75*own+.25*true,"shuffled_graph":.75*own+.25*shuffled}


def run() -> None:
    """先冻结图谱与划分，再构建相同的只读表征缓存；不计算测试性能。"""
    folder=OUT/"benchmark";folder.mkdir(parents=True,exist_ok=True)
    freeze(folder/"encoding_protocol.json",{"source_sha256":sha256(Path(__file__)),"graph_sha256":sha256(OUT/"knowledge_graph/graph.json"),"protocol_sha256":sha256(OUT/"research_protocol.json"),"context_max_tokens":256,"context_last_chars":1800,"labels_used_in_encoding":False})
    raw=json.loads((DATA/"ESConv.json").read_text(encoding="utf-8"));tasks,audit=make_tasks(raw);freeze(folder/"split_audit.json",audit)
    graph=json.loads((OUT/"knowledge_graph/graph.json").read_text(encoding="utf-8"));assert not validate(graph)
    ids=[n["id"] for n in graph["nodes"]];encoder=FrozenEncoder();node_vectors=encoder.encode([n["definition"] for n in graph["nodes"]])
    np.savez_compressed(folder/"graph_vectors.npz",node_ids=np.array(ids),vectors=node_vectors)
    for task,parts in tasks.items():
        path=folder/task;path.mkdir(exist_ok=True)
        for split,rows in parts.items():
            save(path/(split+".json"),rows);cache=path/(split+".npz")
            if cache.exists():continue
            embeddings=encoder.encode([r["text"] for r in rows],context=task=="next_strategy");channels=knowledge_channels(embeddings,node_vectors,graph,ids)
            np.savez_compressed(cache,ids=np.array([r["id"] for r in rows]),dialogues=np.array([r["dialogue"] for r in rows]),labels=np.array([r["label"] for r in rows]),embedding=embeddings,**channels)
            print(json.dumps({"encoded_task":task,"split":split,"rows":len(rows)}),flush=True)
    print(json.dumps({"dialogues":len(raw),"graph_nodes":len(ids)}),flush=True)


if __name__=="__main__":run()
