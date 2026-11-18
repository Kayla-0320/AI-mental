"""第十三轮公开数据接入：固定版本、完整性摘要、受限数组反序列化。"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import pickle
import tarfile
from typing import Any
import zipfile

import numpy as np
from huggingface_hub import hf_hub_download
from experiments.round11_common import digest

ROOT = Path(r"D:\Develop\AIC")
OUTPUT = ROOT / "_国赛优化/round13"
DATA = ROOT / "data/public/round13_sources"
MOSI_REVISION = "092c41f8843df675ab8a169e41419e74e6ae05da"
MELD_REVISION = "9abc51ee7903424ffb971297608aa6d3d0de3bfa"


def save(path: Path, value: Any, overwrite: bool = False) -> None:
    """原子保存工件，默认拒绝覆盖已完成结果。"""
    if path.exists() and not overwrite:
        raise ValueError(f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    temp.replace(path)


class ArrayUnpickler(pickle.Unpickler):
    """只允许数组和基本容器，拒绝下载数据中包含任意可调用对象。"""

    def find_class(self, module: str, name: str) -> Any:
        allowed = {("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode"),
                   ("collections", "OrderedDict"), ("collections", "defaultdict"),
                   ("builtins", "set"), ("builtins", "slice")}
        allowed |= {("builtins", n) for n in ("list", "tuple", "dict", "int", "float", "str")}
        allowed |= {(m, n) for m in ("numpy.core.multiarray", "numpy._core.multiarray") for n in ("_reconstruct", "scalar")}
        allowed |= {(m, "_frombuffer") for m in ("numpy.core.numeric", "numpy._core.numeric")}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"拒绝非数组反序列化对象: {module}.{name}")
        return super().find_class(module, name)


def load_pickle(path: Path) -> Any:
    """读取受限公开特征。"""
    with path.open("rb") as handle:
        return ArrayUnpickler(handle, encoding="latin1").load()


def acquire(kind: str) -> None:
    """只获取可公开访问的指定工件，不绕过需授权的资源。"""
    files = {"mosi": ("tamb2203579/CMU-MOSI", MOSI_REVISION, ["Raw.zip", "label.csv", "Processed/aligned_50.pkl"]),
             "meld": ("declare-lab/MELD", MELD_REVISION, ["MELD.Features.Models.tar.gz"])}
    repo, revision, names = files[kind]
    directory = DATA / kind
    for name in names:
        path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=directory))
        print(json.dumps(dict(dataset=kind, downloaded=name, bytes=path.stat().st_size)), flush=True)
    target = directory / "receipt.json"
    if not target.exists():
        save(target, dict(repository=repo, revision=revision,
            files=[dict(name=n, bytes=(directory/n).stat().st_size, sha256=digest(directory/n)) for n in names],
            original_sources={"mosi":"https://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/", "meld":"https://github.com/declare-lab/MELD"}[kind],
            use="local research; mirror does not establish new rights over original media"))


def inspect(kind: str) -> None:
    """只显示结构与元数据，测试目标不用于候选比较。"""
    if kind == "mosi":
        with zipfile.ZipFile(DATA / "mosi/Raw.zip") as z:
            print("raw archive", len(z.infolist()), [x.filename for x in z.infolist()[:8]])
        obj = load_pickle(DATA / "mosi/Processed/aligned_50.pkl")
        for split, values in obj.items():
            print(split, {k:(getattr(v,"shape",None) or type(v).__name__) for k,v in values.items()})
    else:
        with tarfile.open(DATA / "meld/MELD.Features.Models.tar.gz") as z:
            print([(m.name,m.size) for m in z.getmembers() if m.isfile()][:100])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "inspect"))
    parser.add_argument("kind", choices=("mosi", "meld"))
    args = parser.parse_args()
    acquire(args.kind) if args.action == "download" else inspect(args.kind)
