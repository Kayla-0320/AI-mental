"""将第十轮完全相同的冻结文字流程应用于外部真实转录，不读取情绪标签。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiments.round11_common import ROOT,OUTPUT,digest,save

EXTERNAL=OUTPUT/"emotiontalk_features"


def inputs(path: Path) -> dict[str,np.ndarray]:
    """按既定ID顺序返回纯输入，兼容原冻结编码入口的文件名约定。"""
    split=path.stem
    if split not in ("train","valid","test_inputs"):
        raise ValueError("外部输入划分不符合协议")
    rows=[json.loads(line) for line in (OUTPUT/"corpus"/f"{split}.jsonl").read_text("utf-8").splitlines()]
    return {"sample_ids":np.array([row["id"] for row in rows]),
            "raw_text":np.array([row["text"] for row in rows])}


def main() -> None:
    """不修改原脚本与原输出，只重新绑定外部输入和新目标目录。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("kind",choices=("text","llm","llm3b"))
    parser.add_argument("--splits",nargs="+",default=["train","valid","test_inputs"])
    args=parser.parse_args()
    if args.kind=="text":
        from round5_text_features import encode
        import round10_representations as representation
        output=EXTERNAL/"bge_features"
        output.mkdir(parents=True,exist_ok=True)
        for split in args.splits:
            target=output/f"{split}.npz"
            if target.exists():
                continue
            data=inputs(Path(f"{split}.npz"))
            cls,mean=encode(data["raw_text"].tolist(),Path(r"D:\ai\models\encoders\bge-base-zh-v1.5"))
            np.savez_compressed(target,bge_cls=cls,bge_mean=mean,sample_ids=data["sample_ids"])
            print(json.dumps({"component":"bge","split":split,"n":len(cls)}),flush=True)
        representation.load_npz=inputs
        pending=[split for split in args.splits if not (EXTERNAL/"auxiliary_features"/f"{split}.npz").exists()]
        if pending:
            representation.auxiliary(pending,EXTERNAL/"auxiliary_features")
        originals=[Path(__file__).with_name("round5_text_features.py"),Path(__file__).with_name("round10_representations.py")]
    else:
        import round10_llm_encoder as llm
        llm.load_npz=inputs
        llm.OUTPUT=EXTERNAL
        component="llm_large_features" if args.kind=="llm" else "llm_features"
        pending=[split for split in args.splits if not (EXTERNAL/component/f"{split}.npz").exists()]
        if pending:
            sys.argv=["round10_llm_encoder.py",*(["--large"] if args.kind=="llm" else []),"--splits",*pending]
            llm.main()
        originals=[Path(__file__).with_name("round10_llm_encoder.py")]
    save(EXTERNAL/f"{args.kind}_encoder_binding.json",{
        "original_source_sha256":{path.name:digest(path) for path in originals},
        "input_file_sha256":{split:digest(OUTPUT/"corpus"/f"{split}.jsonl") for split in args.splits},
        "wrapper_sha256":digest(Path(__file__)),"target_supervision_used":False,
        "pipeline":"same pretrained weights,precision,templates and truncation as frozen Round10",
        "input_fields":["sample_ids","raw_text"]},immutable=False)


if __name__=="__main__":
    main()
