"""打包可审阅的报告/源码/锁定模型/逐条结果，不复制几十GB媒体与上游权重。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import OUTPUT,PREVIOUS,digest,save


def main() -> None:
    """创建可核对摘要的审阅压缩包，原始输入仍由其来源清单定位。"""
    audit=json.loads((OUTPUT/"COMPLETION-AUDIT.json").read_text("utf-8"))
    if audit["delivered_checks"]!=audit["total_checks"]:raise ValueError("未完成不可打最终包")
    selected=[]
    folders=("figures","exports","cases","final_models","historical_diagnostic","external_m3","observed_perception",
        "forecast_final","forecast_linear","forecast_replay","runtime_benchmark","runtime_staged","validation","source/final")
    for name in ("REVIEW-PACKAGE.md","第十一轮完整优化与独立验证报告.md","答辩创新点与演示讲稿.md","REPRODUCE.md","PROTOCOL.md",
        "COMPLETION-AUDIT.json","environment-final.json","environment-final-requirements.txt","artifact_manifest_final.json","near_duplicate_audit.json",
        "development_conflict_audit_v2.json","development_conflict_bootstrap.json","audio_development_paired_bootstrap.json",
        "quality_strata.json","quality_strata_plan.json","source/runtime_staged_same_process.py","source/runtime_worker_full_vocab.py"):
        selected.append(OUTPUT/name)
    for name in folders:
        selected.extend(p for p in (OUTPUT/name).rglob("*") if p.is_file() and p.suffix in (".json",".md",".csv",".npz",".pkl",".pt",".py",".ps1",".png",".svg",".xml",".txt"))
    for name in ("robustness/results.json","wave_stress/preregister.json","wave_stress/complete.json","neural/summary.json",
        "missing_route/selection_locked.json","input_audit/complete.json","emotiontalk_acquisition/complete.json",
        "corpus/label_semantics_audit.json","timestamps/audio_duration_match_audit.json","timestamps/forecast_audio_verified_geometry.json"):
        selected.append(OUTPUT/name)
    selected=sorted(set(selected))
    destination=OUTPUT/"第十一轮答辩审阅包.zip"
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for path in selected:
            archive.write(path,path.relative_to(OUTPUT))
        archive.write(PREVIOUS/"refined_final/results.json","references/round10_refined_results.json")
    checksum=digest(destination)
    destination.with_suffix(".zip.sha256").write_text(checksum+"  "+destination.name+"\n","utf-8")
    save(OUTPUT/"review_bundle.json",{"path":str(destination),"sha256":checksum,"bytes":destination.stat().st_size,
        "files":len(selected)+1,"limits":"review bundle includes saved learned models and results;raw media/pretrained encoders not redistributed;absolute provenance paths retained"},immutable=False)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError("压缩包CRC校验失败")
    print(json.dumps({"bundle":str(destination),"MiB":destination.stat().st_size/1024**2,"files":len(selected)+1}),flush=True)


if __name__=="__main__":main()
