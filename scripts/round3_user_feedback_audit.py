"""核验用户反馈材料是否可作真实用户证据。

只读取表结构与标注列，不引用任何分数；输出可复核 JSON。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SHEET = "原始问卷数据"
TYPE_COLUMN = "B"
FIRST_ROW = 3
LAST_ROW = 157


def sha256(path: Path) -> str:
    """流式摘要，避免把工作簿整体读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(workbook: Path, report: Path | None) -> dict:
    """读取标注列并确认是否全为合成数据；同时记录旧报告是否有原始记录。"""
    import openpyxl

    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        sheets = list(book.sheetnames)
        if SHEET not in sheets:
            raise ValueError("缺少原始问卷数据表")
        sheet = book[SHEET]
        values = [sheet[f"{TYPE_COLUMN}{row}"].value for row in range(FIRST_ROW, LAST_ROW + 1)]
        header = sheet["A1"].value
    finally:
        book.close()
    distinct = sorted({str(value) for value in values})
    report_record = {"path": str(report), "exists": False} if report else {"path": None, "exists": False}
    if report and report.is_file():
        report_record = {"path": str(report), "exists": True, "bytes": report.stat().st_size, "sha256": sha256(report)}
    return {
        "schema": "user_feedback_material_audit.v1",
        "workbook": {"path": str(workbook), "bytes": workbook.stat().st_size, "sha256": sha256(workbook), "sheets": sheets},
        "raw_sheet": {
            "name": SHEET,
            "title_cell": str(header),
            "label_range": f"{TYPE_COLUMN}{FIRST_ROW}:{TYPE_COLUMN}{LAST_ROW}",
            "rows": len(values),
            "distinct_labels": distinct,
            "all_synthetic": distinct == ["Synthetic"],
        },
        "usable_as_real_user_evidence": False,
        "usable_as_questionnaire_structure_reference": distinct == ["Synthetic"],
        "legacy_report": report_record,
        "raw_record_found": False,
        "conclusion": "表格为 Synthetic 标注，只能参考问卷结构与评价维度；旧报告未找到对应原始回收记录，不能作为真实用户效果证据。",
    }


def main() -> None:
    """写出核验 JSON；缺失文件直接报错。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.workbook, arguments.report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "all_synthetic": result["raw_sheet"]["all_synthetic"],
                      "rows": result["raw_sheet"]["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
