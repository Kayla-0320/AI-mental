"""固定公开归档的并行范围续传，最终完整性仍由发布者SHA256判定。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from huggingface_hub import HfFileSystem

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import digest, save


def download(remote: str, part: Path, expected_size: int, expected_hash: str, workers: int) -> None:
    """保留已有串行前缀；每个剩余范围单独提交，最后顺序组装。"""
    folder=part.parent/"round11_transfer_ranges"
    folder.mkdir(parents=True,exist_ok=True)
    plan_path=folder/"plan.json"
    if plan_path.exists():
        plan=json.loads(plan_path.read_text("utf-8"))
        if plan["expected_size"] != expected_size or plan["expected_hash"] != expected_hash:
            raise ValueError("范围传输来源锁冲突")
        initial=plan["initial_prefix_bytes"]
        # 组装中断只回退本工具追加的临时片段，不涉及原始工件。
        if part.stat().st_size < initial:
            raise ValueError("已有前缀被缩短")
        if part.stat().st_size > initial:
            if not part.resolve().is_relative_to(folder.parent.resolve()) or part.is_symlink():
                raise ValueError("临时前缀路径不在传输目录")
            with part.open("r+b") as handle:
                handle.truncate(initial)
    else:
        initial=part.stat().st_size if part.exists() else 0
        plan={"initial_prefix_bytes":initial,"expected_size":expected_size,
              "expected_hash":expected_hash,"range_bytes":64*1024*1024}
        save(plan_path,plan)
    bounds=[(lo,min(lo+plan["range_bytes"],expected_size)) for lo in range(initial,expected_size,plan["range_bytes"])]

    def fetch(bound: tuple[int,int]) -> Path:
        lo,hi=bound
        target=folder/f"{lo}-{hi}.bin"
        if target.exists() and target.stat().st_size==hi-lo:
            return target
        for attempt in range(5):
            try:
                fs=HfFileSystem()
                with fs.open(remote,"rb",block_size=4*1024*1024,cache_type="readahead") as source:
                    source.seek(lo)
                    values=source.read(hi-lo)
                if len(values)!=hi-lo:
                    raise ValueError("远端范围长度不符")
                temporary=target.with_suffix(".partial")
                temporary.write_bytes(values)
                temporary.replace(target)
                return target
            except Exception:
                if attempt==4:
                    raise
                time.sleep(min(2**attempt,8))
        raise ValueError("范围下载失败")

    last=time.monotonic()
    done=initial
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fetch,bound):bound for bound in bounds}
        for future in as_completed(futures):
            done+=future.result().stat().st_size
            if time.monotonic()-last>=20 or done==expected_size:
                print(json.dumps({"range_bytes_downloaded":done,"total":expected_size,"workers":workers}),flush=True)
                last=time.monotonic()
    with part.open("ab") as out:
        for lo,hi in bounds:
            with (folder/f"{lo}-{hi}.bin").open("rb") as source:
                shutil.copyfileobj(source,out,4*1024*1024)
    if part.stat().st_size!=expected_size or digest(part)!=expected_hash:
        raise ValueError("组装归档未通过发布者SHA256核验")
    save(folder/"verified.json",{**plan,"sha256_verified":True},immutable=False)
    # 只删除本工具生成并已被完整归档校验覆盖的逐范围临时文件。
    for lo,hi in bounds:
        target=folder/f"{lo}-{hi}.bin"
        if not target.resolve().is_relative_to(folder.resolve()) or target.is_symlink():
            raise ValueError("范围临时文件路径异常")
        target.unlink()
