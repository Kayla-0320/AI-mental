"""首次使用KonIQ-10k真实画质评分，只评冻结代理，不据新标签调参。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile
import zlib
import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from experiments.round14_common import ROOT,freeze,save,sha256
from experiments.round17_pixels import quality_from_frames
from experiments.round18_visual_guard import unusable

OUT=ROOT/"round18/natural_quality"
DATA=Path(r"D:\Develop\AIC\data\public\round18_sources\koniq")
API="https://darus.uni-stuttgart.de/api/access/datafile/"


def run() -> None:
    """通过作者归档分段下载ID哈希预选128图，分数读取发生在质量测量后。"""
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    original=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    freeze(OUT/"protocol_v3.json",{**original,"source_sha256":sha256(Path(__file__)),"original_protocol_sha256":sha256(OUT/"protocol.json"),"repair":"unmatched archive names quarantined; undefined constant-score correlation serialized as null; no replacement or policy retuning"})
    session=requests.Session()
    def get_range(url: str,start: int,end: int) -> bytes:
        r=session.get(url,headers={"Range":f"bytes={start}-{end}"},timeout=(15,45));r.raise_for_status()
        if r.status_code!=206 or not r.headers.get("Content-Range","").startswith(f"bytes {start}-"):raise ValueError("服务器未尊重分段下载")
        return r.content
    # 文件由作者DOI 10.18419/DARUS-2435的固定V1记录提供，临时签名URL不保存。
    meta=json.loads((ROOT/"round18/koniq_metadata.json").read_text(encoding="utf-8"))["data"]["latestVersion"]
    item=next(f["dataFile"] for f in meta["files"] if f["dataFile"]["id"]==72286);size=item["filesize"]
    response=session.get(API+"72286",headers={"Range":f"bytes={size-65558}-{size-1}"},timeout=(15,45));response.raise_for_status()
    if response.status_code!=206:raise ValueError("分段下载不可用")
    tail=response.content;url=response.url;at=tail.rfind(b"PK\x05\x06");end=tail[at:];fields=struct.unpack("<4s4H2IH",end[:22]);cd_size,cd_offset=fields[5:7]
    directory=get_range(url,cd_offset,cd_offset+cd_size-1);patched=end[:16]+struct.pack("<I",0)+end[20:]
    with zipfile.ZipFile(io.BytesIO(directory+patched)) as archive:entries=[i for i in archive.infolist() if i.filename.lower().endswith((".jpg",".png",".jpeg"))]
    chosen=sorted(entries,key=lambda i:hashlib.sha256(("18|42|"+Path(i.filename).name).encode()).digest())[:128]
    save(OUT/"selected_ids.json",[Path(i.filename).name for i in chosen])
    def worker(info: zipfile.ZipInfo) -> dict:
        name=Path(info.filename).name;path=DATA/name
        if not path.exists():
            content=get_range(url,info.header_offset,min(size-1,info.header_offset+30+len(info.filename.encode())+info.compress_size+4096))
            if content[:4]!=b"PK\x03\x04":raise ValueError("本地头不匹配")
            header=struct.unpack("<4s5H3I2H",content[:30]);start=30+header[-2]+header[-1];raw=content[start:start+info.compress_size]
            pixels=zlib.decompress(raw,-15) if info.compress_type==8 else raw
            if len(pixels)!=info.file_size or zlib.crc32(pixels)!=info.CRC:raise ValueError("图像CRC或长度不符")
            path.write_bytes(pixels)
        if zlib.crc32(path.read_bytes())!=info.CRC:raise ValueError("缓存图像CRC不符")
        with Image.open(path) as im:rgb=np.asarray(im.convert("RGB"))
        image=cv2.resize(rgb,(128,128))[None];q=quality_from_frames(image)
        return {"image_name":name,"quality":q.quality_score,"rejected_at_051":q.quality_score<=.51,"hard_rejected":unusable(image),"reasons":q.reason_codes,"sha256":sha256(path)}
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(worker,chosen))
    save(OUT/"measured_before_labels.json",rows)
    score_path=DATA/"scores.tab"
    if not score_path.exists():
        r=session.get(API+"72288",timeout=(15,60));r.raise_for_status();score_path.write_bytes(r.content)
    table=pd.read_csv(score_path,sep="\t");table["image_name"]=table.image_name.astype(str).str.strip('"')
    lookup=table.set_index("image_name");unmatched=[r["image_name"] for r in rows if r["image_name"] not in lookup.index]
    rows=[r for r in rows if r["image_name"] in lookup.index]
    mos=np.array([float(lookup.loc[r["image_name"],"MOS"]) for r in rows]);q=np.array([r["quality"] for r in rows]);low=mos<2.5
    pd.DataFrame([dict(r,MOS=float(y)) for r,y in zip(rows,mos,strict=True)]).to_csv(OUT/"evaluation.csv",index=False)
    result={"n":len(rows),"selected_count":128,"unmatched_names_quarantined_without_replacement":unmatched,"all_archive_images":len(entries),"dataset_doi":"10.18419/DARUS-2435","source_files":{"images_id":72286,"images_publisher_checksum":item.get("checksum"),"scores_id":72288,"downloaded_score_sha256":sha256(score_path)},
        "spearman_quality_MOS":float(spearmanr(q,mos).statistic) if np.std(q)>0 else None,"constant_proxy_correlation_undefined":bool(np.std(q)==0),"auc_low_MOS_below_2_5":float(roc_auc_score(low,-q)),"low_MOS_count":int(low.sum()),"rejected_at_051":int(np.sum(q<=.51)),"hard_rejected":int(sum(r["hard_rejected"] for r in rows)),
        "low_MOS_recall_at_051":float(np.mean(q[low]<=.51)),"higher_MOS_rejection_at_051":float(np.mean(q[~low]<=.51)),"parameters_tuned_on_MOS":False,"emotion_fusion_measured":False,"published_MOS_not_new_project_human_blind_review":True,
        "limitations":"single-image resized128 proxies; no paired text or emotion labels; no natural-video routing or project-human claim","terms":meta.get("termsOfUse")}
    save(OUT/"results.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":run()
