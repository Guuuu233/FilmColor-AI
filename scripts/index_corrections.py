"""Corrections list for the material index 索引.csv, which itself is not edited (decision 2026-10-21): the rows that
are not in the header's column order, with their fields read by position.

  index_corrections.py MATERIAL_ROOT OUT_JSON

Layouts seen on 2026-10-23 (all rows have 11 cells):
  header          编号, 文件名, 类型 (folder), 大小MB, 胶片/来源, 原文件名, 原路径, 说明, 需目视复核, 配对编号, 许可
  collected       类型 = 已采集; cells: 编号, 文件名, 已采集, 许可, 说明, 大小, 画幅, 胶片, 需目视复核, 许可 (again),
                  note; no original name, original path or pairing
  format_in_type  类型 holds the film format (e.g. 135全画幅(35mm)); the other cells as the header
A pairing counts only if 配对编号 is a list of material ids (letter + 3-4 digits). The folder of a row is taken from
where its file is on disk. Descriptions (说明) are not copied.
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ID_RE = re.compile(r"^[A-Z]\d{3,4}$")


def layout(row, tops):
    return "header" if row[2] in tops else "collected" if row[2] == "已采集" else "format_in_type"


def main(root, out_json):
    root = Path(root)
    tops = {p.name for p in root.iterdir() if p.is_dir()}
    where = {}
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            where.setdefault(p.name, []).append(p.relative_to(root).as_posix())
    raw = (root / "索引.csv").read_bytes()
    rows = list(csv.reader(open(root / "索引.csv", encoding="utf-8-sig")))[1:]
    out = []
    for r in rows:
        lay = layout(r, tops)
        if lay == "header":
            continue
        pairs = [x.strip() for x in re.split(r"[;,，；\s]+", r[9]) if x.strip()] if lay != "collected" else []
        out.append(dict(
            id=r[0], file=r[1], layout=lay, on_disk=where.get(r[1], []),
            film=r[7] if lay == "collected" else None, film_format=r[6] if lay == "collected" else r[2],
            size=r[5] if lay == "collected" else r[3],
            original_name=None if lay == "collected" else r[5] or None,
            original_path=None if lay == "collected" else r[6] or None,
            pairing=pairs if pairs and all(ID_RE.match(x) for x in pairs) else [],
            pairing_cell_as_recorded=r[9] or None,
            licence_as_recorded=r[3] if lay == "collected" else r[10],
            review=r[8]))
    res = dict(id="material_index_corrections", created="2026-10-23",
               index_sha256=hashlib.sha256(raw).hexdigest(), index_rows=len(rows),
               layouts=dict(Counter(layout(r, tops) for r in rows)), rows=out,
               note="licence_as_recorded is the index text; for rows whose source is a link it is not a verified "
                    "licence (T1b review 2026-10-22, item 4)")
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
