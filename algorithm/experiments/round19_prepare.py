"""第十九轮知识图谱来源固定、模型下载和工作范围封存，不读取评测标签。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import requests
from huggingface_hub import HfApi,hf_hub_download
from experiments.round14_common import ROOT,freeze,save,sha256

OUT=ROOT/"round19"
DATA=Path(r"D:\Develop\AIC\data\public\round19_sources")
MODEL=Path(r"D:\ai\models\encoders\all-MiniLM-L6-v2")
URLS={
    "UNICEF_TALK":"https://www.unicef.org/parenting/mental-health/how-to-talk-to-kids-mental-health",
    "UNICEF_11_13":"https://www.unicef.org/parenting/mental-health/conversation-starters-11-13-years",
    "ACL2021":"https://aclanthology.org/2021.acl-long.269/",
}


def run() -> None:
    """首次下载前固定分组、任务、比较臂和选择标准；真人评价始终留空。"""
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True);MODEL.mkdir(parents=True,exist_ok=True)
    freeze(OUT/"research_protocol.json",{"seed":42,"scope":"offline knowledge graph, no production safety or platform edits","tasks":["recognize observed supporter response strategy","predict next strategy from past context only"],
        "strategy_labels":"eight original ESConv annotations; not optimality, helpfulness, or new human ratings","split":"dialogue text hash ordered70/15/15; exact input duplicates removed across earlier/later splits; no participant-disjoint claim",
        "arms":["no_graph","definitions_only","true_graph","shuffled_graph"],"selection":"each arm same LinearSVC C[.1,1.] by validation macroF1; no heldout model/graph edits",
        "inputs":"same training-only word TFIDF + same frozen MiniLM text384;8 extra knowledge channels or8 zeros",
        "graph_channels":".75 own definition cosine+.25 max neighboring facet cosine; compare definitions and permuted facet edges",
        "knowledge":"public definitions and relationships only; dataset texts/labels not inserted into graph","inference":"research-only pending concepts explicit opt-in; source scope retained","source_sha256":sha256(Path(__file__)),
        "limitations":["English public conversations are not Chinese adolescent field effectiveness","frozen encoder pretraining overlap unknown","strategy attribution is not next-strategy selection","no generated or model substituted human ratings"]})
    records={}
    def fetch(item: tuple[str,str]) -> tuple[str,dict]:
        key,url=item;path=DATA/(key+".html")
        if not path.exists():
            r=requests.get(url,timeout=(15,60));r.raise_for_status();path.write_bytes(r.content)
        return key,{"url":url,"sha256":sha256(path),"local_cache":str(path),"bytes":path.stat().st_size,"redistribute_full_source":False}
    with ThreadPoolExecutor(max_workers=3) as pool:records.update(pool.map(fetch,URLS.items()))
    github=requests.get("https://api.github.com/repos/thu-coai/Emotional-Support-Conversation/commits/main",timeout=30);github.raise_for_status();revision=github.json()["sha"]
    lock=OUT/"source_lock.json"
    if lock.exists():revision=json.loads(lock.read_text(encoding="utf-8"))["ESCONV"]["revision"]
    for filename in ("ESConv.json","README.md"):
        path=DATA/filename
        if not path.exists():
            r=requests.get(f"https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/{revision}/{filename}",timeout=(15,90));r.raise_for_status();path.write_bytes(r.content)
    records["ESCONV"]={"revision":revision,"url":f"https://github.com/thu-coai/Emotional-Support-Conversation/tree/{revision}","sha256":sha256(DATA/"ESConv.json"),"license":"CC BY-NC4.0 author HF dataset card; academic research only in README","paper":"https://aclanthology.org/2021.acl-long.269/"}
    freeze(lock,records)
    api=HfApi();info=api.model_info("sentence-transformers/all-MiniLM-L6-v2");model_revision=info.sha
    if (OUT/"encoder_lock.json").exists():model_revision=json.loads((OUT/"encoder_lock.json").read_text(encoding="utf-8"))["revision"]
    files=["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.txt"]
    for filename in files:hf_hub_download("sentence-transformers/all-MiniLM-L6-v2",filename,revision=model_revision,local_dir=MODEL)
    freeze(OUT/"encoder_lock.json",{"repo":"sentence-transformers/all-MiniLM-L6-v2","revision":model_revision,"path":str(MODEL),"license":"Apache2.0","pool":"attention-mask mean then L2","max_tokens":256,"files":{f:sha256(MODEL/f) for f in files}})
    print(json.dumps({"sources":len(records),"ESConv_sha256":records["ESCONV"]["sha256"],"encoder":str(MODEL)}),flush=True)


if __name__=="__main__":run()
