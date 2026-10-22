"""T1 roll-level data check (task list 2026-10-21 v2): one record per negative frame and one per roll, with how
sure we are of the roll, where the reference comes from and who graded it, whether the engine reads the frame,
dev / frozen, ageing and why a frame is left out.

  t1_roll_inventory.py OUT_DIR

Inputs (read only; nothing under the material or photo folder is opened):
  - experiments/s3_ext_audit/dataset_20261018/catalog.json   material folder, frames merged by audit_dataset_report.py
  - data/manifests/dataset_visual_review_20261018.json       by-eye exclusions
  - data/raw/public/blueneg/meta.json                        BlueNeg release metadata (roll_id, number_in_roll, film)
  - data/raw/{own,public}                                    raws kept in the repo (hashed, except the BlueNeg DNGs,
                                                             which are matched by release file name)
  - experiments/s3_photos_audit/match.json                   photo folder vs material, by content
  - data/manifests/s3_dev_frames.json, bn_rb_frames.json, frozen_test_v2_ds.json, experiments/s3_ml/{labels,
    user_labels,processing_status.json}                      which frames research has used / labelled
Frozen frames: identity, roll and counts only (the catalog never decoded them; nothing here decodes anything).

Roll levels:
  verified  one roll by physical / release evidence: own rolls (one scan folder per roll opened by a roll-label
            JPEG, S3_EXT_DATA_AUDIT §2); BlueNeg year-month + letter when every release roll_id in it agrees
            (one film stock, frame numbers never repeat and rise with the shooting date)
  inferred  a grouping that is probably one roll but not checked (folder of a loose batch with one camera, NLP
            frames scanned on the same day with no conflicting film stock)
  unknown   no roll information, or the grouping is known not to be one roll
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from filmcolor import frozen

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "experiments/s3_ext_audit/dataset_20261018/catalog.json"
MAN = REPO / "data/manifests"
READABLE = {"camera_raw", "camera_dng", "scanner_dng"}  # what filmcolor.app / pipeline open
OWN_ROLLS = {  # scan folder -> roll id (S3_EXT_DATA_AUDIT §2); the dsc number is checked against the folder below
    "Kodak G200（尼康数字化）-": "own-g200-20240320-dsc4889",
    "Eastman 5219(ecn-2次抛冲洗 尼康数字化)-": "own-e5219-20240323-dsc8202",
    "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-": "own-e5207-20240323-dsc8242",
    "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-3": "own-e5207-20240401-dsc6161",
    "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-2": "own-e5207-20240401-dsc6199",
    "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-1": "own-e5207-20240401-dsc6233",
    "APX 400(xtol次抛冲洗 尼康数字化)": "own-apx400-20240310-dsc6548",
    "Fuji Acros ll(xtol 尼康数字化)-": "own-acros2-20240331-dsc4899",
}
GRADER = {"own": "冲扫店调色师", "shop_r01": "冲扫店调色师", "blueneg": "当年冲印店（印相扫描）"}
DSC = re.compile(r"(\d{4,})")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def blueneg_rolls(meta):
    """Year-month + letter groups of the release metadata and whether the release roll_ids in each agree."""
    g = defaultdict(list)
    for m in meta:
        rid = m["roll_id"]
        key = f"ds-{rid[:6] + rid[-1]}" if rid[-1].isalpha() else f"ds-{rid}"
        g[key].append(m)
    out = {}
    for key, ms in g.items():
        films = {m["film_type"] for m in ms}
        nums = [int(m["number_in_roll"]) for m in ms if str(m["number_in_roll"]).isdigit()]
        dup = len(nums) - len(set(nums))
        by_date = defaultdict(list)
        for m in ms:
            if str(m["number_in_roll"]).isdigit():
                by_date[m["date"]].append(int(m["number_in_roll"]))
        spans = [(min(v), max(v)) for _, v in sorted(by_date.items())]
        rising = all(a[1] < b[0] for a, b in zip(spans, spans[1:]))
        letter = key[-1].isalpha()
        ids = sorted({m["roll_id"] for m in ms})
        if letter and len(films) == 1 and dup == 0 and rising:
            level = "verified"
            basis = (f"BlueNeg 官方 roll_id {ids[0]}" if len(ids) == 1 else
                     f"BlueNeg 官方 roll_id {len(ids)} 个（同字母跨 {len(by_date)} 天），胶片型号唯一，帧号按日期递增、不重复")
        elif letter:
            level = "inferred"
            why = [w for w, bad in (("胶片型号不唯一", len(films) > 1), (f"帧号重复 {dup} 个", dup > 0),
                                    ("帧号不随日期递增", not rising)) if bad]
            basis = "同字母合并，但" + "、".join(why)
        elif len(films) == 1 and dup == 0 and rising:
            level = "inferred"
            basis = f"官方 roll_id {ids} 没有卷字母；胶片型号唯一，帧号不重复（跨 {len(by_date)} 天）"
        else:
            level = "unknown"
            basis = f"官方 roll_id {ids} 没有卷字母" + (f"，帧号重复 {dup} 个，不止一卷" if dup else "，型号或帧号顺序不一致")
        out[key] = dict(level=level, basis=basis, release_roll_ids=ids, films=sorted(films), dates=len(by_date),
                        release_frames=len(ms), dup_numbers=dup)
    return out


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cat = json.loads(CATALOG.read_text())
    files = {f["rel"]: f for f in cat["files"]}
    vis = json.loads((MAN / "dataset_visual_review_20261018.json").read_text())
    nn = vis["not_a_negative"]
    not_neg = {**{i: "正像（确定）" for i in nn["positive_certain"]}, **{i: "正像（疑似）" for i in nn["positive_likely"]},
               **{i: "不是胶片：" + w for i, w in nn["not_film"].items()}}
    odd = set(nn["unclear_maybe_reversal_or_odd_colour"])
    versions = set(vis["other"]["same_scene_versions"])
    meta = json.loads((REPO / "data/raw/public/blueneg/meta.json").read_text())
    bmeta = {m["filename"]: m for m in meta}
    brolls = blueneg_rolls(meta)
    dev = json.loads((MAN / "s3_dev_frames.json").read_text())
    dev_by_sha = {f["raw_sha256"]: f for f in dev["frames"]}
    s3_bn = {f["stem"] for f in dev["frames"] if f["domain"] == "blueneg"}
    bn_rb = {f["stem"] for f in json.loads((MAN / "bn_rb_frames.json").read_text())["frames"]}
    fz = frozen.frozen_hashes()
    v2rule = json.loads((MAN / "frozen_test_v2_ds.json").read_text())["rule"]
    viewed_rolls = set(v2rule["viewed_dev_rolls"]) | set(v2rule["dev_manifest_rolls"])
    status = json.loads((REPO / "experiments/s3_ml/processing_status.json").read_text())
    design_excl = {(d, s) for d, v in status["design_exclusions"].items() if d != "reason" for s in v}
    failed = {f["id"] for f in status["pipeline_failures"]}
    p4 = {(p.parent.name, p.stem) for p in (REPO / "experiments/s3_ml/labels").glob("*/*.json")}
    user = {p.stem for p in (REPO / "experiments/s3_ml/user_labels").glob("*.json")}

    frames = []

    def research_use(domain, stem):
        """S3 design exclusion / pipeline failure / labels for a frame of the S3 dev manifest."""
        r = dict(in_s3_manifest=True, p4_label=(domain, stem) in p4,
                 user_label=domain == "ext_own" and stem in user)
        if (domain, stem) in design_excl:
            r["s3_status"] = "设计排除：" + status["design_exclusions"]["reason"][domain]
        elif f"{'ext' if domain == 'ext_own' else domain}:{stem}" in failed:
            r["s3_status"] = "处理失败：自动片基找不到候选"
        else:
            r["s3_status"] = "有标签" if r["p4_label"] else "清单内、无标签"
        return r

    # ---- material folder (catalog) ----
    own_dirs = defaultdict(list)
    for fr in cat["frames"]:
        mem = [files[r] for r in fr["files"]]
        ids = [m["id"] for m in mem if m["id"]]
        src = fr["source"]
        rec = dict(id=fr["frame"], where="material", source=src, kind=fr["kind"], files=fr["files"],
                   fmts=fr["fmts"], film=fr["film"], camera=fr["camera"], roll_label=fr["roll"],
                   partition="frozen" if fr["frozen"] else "dev", flags=[], exclude=[], ageing=[])
        # roll
        if src == "own":
            d = fr["roll"].split(":", 1)[1]
            rec["roll"] = OWN_ROLLS[d]
            own_dirs[d] += [m["orig"] for m in mem]
            nums = [int(DSC.search(m["orig"]).group(1)) for m in mem]
            rec.update(frame_no=min(nums), frame_order="翻拍文件号（与底片帧号一致与否未核实）")
        elif src == "blueneg":
            stem = fr["frame"].split(":", 1)[1]
            m = bmeta[stem]
            rec["roll"] = fr["roll"]
            rec.update(frame_no=int(m["number_in_roll"]) if str(m["number_in_roll"]).isdigit() else None,
                       frame_order="BlueNeg 官方 number_in_roll", release_roll_id=m["roll_id"],
                       release_testset=m["is_testset"])
            if m["partition"] == "blue-corrupted":
                rec["ageing"].append("蓝层变质（官方 partition）")
        else:
            rec["roll"] = fr["roll"]  # loose:<folder> or None
            rec.update(frame_no=None, frame_order=None)
        # reference evidence and graders (frozen frames: whether there is one, nothing else)
        lv = set(fr["pair_levels"])
        rec["ref_strength"] = ("strong" if lv & {"dev", "frozen", "official"} else
                               "medium" if lv & {"photos", "content"} else "weak" if "index" in lv else "none")
        g = set()
        if not fr["frozen"]:
            base = GRADER.get(src, "未知（群友或网络提供）")
            if lv & {"dev", "official", "content", "index"}:
                g.add(base)
            for m in mem:
                pr = m.get("photo_refs", {})
                if pr.get("user_lightroom"):
                    g.add("用户（Lightroom）")
                if pr.get("photo_folder_other"):
                    g.add("照片文件夹的其他成片（店家或冲印店）")
                d = dev_by_sha.get(m["sha256"])
                if d:
                    rec.update(research_use(d["domain"], d["stem"]))
                    if d["domain"] == "ext_own" and d["stem"] in user:
                        g.add("用户（Lightroom）")
        rec["ref_levels"] = sorted(lv)
        rec["graders"] = sorted(g)
        # engine
        rec["engine_readable"] = bool(set(fr["fmts"]) & READABLE)
        rec["readable_via"] = "material" if rec["engine_readable"] else None
        if src == "blueneg":
            dng = REPO / "data/raw/public/blueneg/negative-16bit" / bmeta[stem]["partition"] / f"{stem}.dng"
            if dng.exists():
                rec.update(engine_readable=True, readable_via="data/raw BlueNeg 官方 DNG")
            if not fr["frozen"] and stem in s3_bn:
                rec.update(research_use("blueneg", stem))
            if stem in bn_rb or rec.get("in_s3_manifest"):
                rec["in_research_manifest"] = True
        # exclusions and flags
        for i in ids:
            if i in not_neg:
                rec["exclude"].append(not_neg[i])
            if i in odd:
                rec["flags"].append("颜色可疑或可能是反转片")
            if i in versions:
                rec["flags"].append("同一画面的多个版本（A313～A318）")
        if src == "own" and fr["fmts"] == ["jpeg"] and len(mem) == 1 and \
                int(DSC.search(mem[0]["orig"]).group(1)) == int(rec["roll"].rsplit("dsc", 1)[1]):
            rec["exclude"].append("卷标帧（JPEG）")
        if fr["review"]:
            rec["flags"].append(f"索引“需目视复核”={fr['review']}")
        if rec.get("s3_status", "").startswith(("设计排除", "处理失败")):
            rec["flags"].append("S3 " + rec["s3_status"])
        frames.append(rec)

    # own-roll check: the roll id's dsc number is the first file of the folder and the first file is the JPEG label
    own_check = {}
    for d, origs in own_dirs.items():
        nums = sorted(int(DSC.search(o).group(1)) for o in origs)
        first = min(origs, key=lambda o: int(DSC.search(o).group(1)))
        own_check[OWN_ROLLS[d]] = dict(folder=d, first=first, files=len(origs), first_no=nums[0], last_no=nums[-1],
                                       missing_numbers=nums[-1] - nums[0] + 1 - len(set(nums)),
                                       id_matches_first=int(OWN_ROLLS[d].rsplit("dsc", 1)[1]) == nums[0],
                                       first_is_jpeg_label=first.lower().endswith(".jpg"))

    # ---- raws kept in the repo ----
    mat_sha = {f["sha256"]: f["rel"] for f in cat["files"]}
    extra = []
    um = list(csv.DictReader(open(MAN / "ultramax400_r01.csv")))
    for row in um:
        p = REPO / "data/raw" / row["file"]
        h = sha256(p)
        assert h == row["sha256"], p
        stem = p.stem
        extra.append(dict(id=f"shop_r01:{stem}", where="data/raw", source="shop_r01", kind="colour_neg",
                          files=[f"data/raw/{row['file']}"], fmts=["camera_raw"], film=row["film_stock"],
                          camera=row["camera_model"], roll_label="ultramax400_r01", roll="shop_r01",
                          frame_no=int(DSC.search(stem).group(1)), frame_order="翻拍文件号（与底片帧号一致与否未核实）",
                          partition="dev", ref_strength="strong" if row["reference_files"] else "none",
                          ref_levels=["registered"] if row["reference_files"] else [],
                          graders=[GRADER["shop_r01"]] if row["reference_files"] else [],
                          engine_readable=True, readable_via="data/raw", flags=[], exclude=[], ageing=[],
                          dup_of_material=mat_sha.get(h), sha256=h, **research_use("shop", stem)))
    label = REPO / "data/raw/own/ultramax400_r01/DSC_3820.JPG"
    extra.append(dict(id="shop_r01:DSC_3820", where="data/raw", source="shop_r01", kind="colour_neg",
                      files=["data/raw/own/ultramax400_r01/DSC_3820.JPG"], fmts=["jpeg"], film="Kodak UltraMax 400",
                      camera="NIKON Z 7_2", roll_label="ultramax400_r01", roll="shop_r01", frame_no=3820,
                      frame_order="翻拍文件号", partition="dev", ref_strength="none", ref_levels=[], graders=[],
                      engine_readable=False, readable_via=None, flags=[], exclude=["卷标帧（JPEG）"], ageing=[],
                      dup_of_material=mat_sha.get(sha256(label)), in_s3_manifest=False))
    nlp_refs = {p.name.split("-positive")[0] for p in (REPO / "data/reference/nlp_default").glob("*-positive.tif")}
    nlp_rows = list(csv.DictReader(open(MAN / "nlp_official.csv")))
    for row in nlp_rows:
        p = REPO / "data/raw" / row["file"]
        h = sha256(p)
        assert h == row["sha256"], p
        stem = p.stem
        bw = ("nlp", stem) in design_excl
        extra.append(dict(id=f"nlp:{stem}", where="data/raw", source="nlp_official", kind="bw_neg" if bw else "colour_neg",
                          files=[f"data/raw/{row['file']}"], fmts=["camera_raw"], film=row["film_stock"],
                          camera=row["camera_model"], roll_label=row["roll_id"], roll=f"nlp-{row['roll_id'][5:]}",
                          frame_no=int(DSC.search(stem).group(1)), frame_order="翻拍文件号",
                          partition="dev", ref_strength="strong" if stem in nlp_refs else "none",
                          ref_levels=["registered"] if stem in nlp_refs else [],
                          graders=["NLP 默认输出（用户在 LR 里操作，不是人工调色）"] if stem in nlp_refs else [],
                          engine_readable=True, readable_via="data/raw", flags=[], exclude=[], ageing=[],
                          dup_of_material=mat_sha.get(h), sha256=h, **research_use("nlp", stem)))
    for raw, refs in (("public/nlp_forum_2023/DSC_5889.NEF", 2), ("public/nlp_forum_9652/DSC03060.ARW", 1),
                      ("public/fabio_sirna/fabio-sirna-data-driven-approach-to-film-inversion/"
                       "2026-04-03T101828001-DC-S5M2.rw2", 0)):
        p = REPO / "data/raw" / raw
        h = sha256(p)
        extra.append(dict(id=f"{raw.split('/')[1]}:{p.stem}", where="data/raw", source=raw.split("/")[1],
                          kind="colour_neg", files=[f"data/raw/{raw}"], fmts=["camera_raw"], film=None, camera=None,
                          roll_label=None, roll=None, frame_no=None, frame_order=None, partition="dev",
                          ref_strength="weak" if refs else "none", ref_levels=["提供方附带"] if refs else [],
                          graders=["未知（论坛提供，含 NLP 输出或冲印扫描）"] if refs else [],
                          engine_readable=True, readable_via="data/raw", flags=[], exclude=[], ageing=[],
                          dup_of_material=mat_sha.get(h), sha256=h, in_s3_manifest=False))
    for r in extra:
        if r.get("sha256") in fz:
            r["partition"] = "frozen"
    by_file = {x: r for r in frames for x in r["files"]}
    for r in extra:
        m = by_file.get(r.get("dup_of_material"))
        if m is not None:
            frames.remove(m)
            r.update(where="material+data/raw", material_id=m["id"], files=m["files"] + r["files"],
                     ref_levels=sorted(set(r["ref_levels"]) | set(m["ref_levels"])))
            if r["ref_strength"] == "none" and m["ref_strength"] != "none":
                r.update(ref_strength=m["ref_strength"], graders=m["graders"])
    frames += extra

    # ---- roll level per frame ----
    nlp_films = defaultdict(set)
    for row in nlp_rows:
        if row["film_stock"] != "unknown":
            nlp_films[row["roll_id"]].add(row["film_stock"])
    cams = defaultdict(set)
    for r in frames:
        if r["roll"]:
            cams[r["roll"]].add(r["camera"])
    for r in frames:
        k = r["roll"]
        if r["source"] in ("own", "shop_r01"):
            chk = own_check.get(k)
            ok = chk is None or (chk["id_matches_first"] and chk["first_is_jpeg_label"])
            r["roll_level"] = "verified" if ok else "inferred"
            r["roll_basis"] = "一卷一个翻拍目录，首个文件是卷标 JPEG（S3_EXT_DATA_AUDIT §2）"
        elif r["source"] == "blueneg":
            b = brolls[k]
            r["roll_level"], r["roll_basis"] = b["level"], b["basis"]
        elif r["source"] == "nlp_official":
            ok = len(nlp_films[r["roll_label"]]) <= 1
            r["roll_level"] = "inferred" if ok else "unknown"
            r["roll_basis"] = "只有翻拍日期（同日翻拍）" + ("" if ok else "，同日胶片型号不止一种")
        elif k and k.startswith("loose:"):
            one = len(cams[k] - {None}) <= 1 and r["kind"] == "colour_neg" and len([x for x in frames if x["roll"] == k]) > 1
            r["roll_level"] = "inferred" if one else "unknown"
            r["roll_basis"] = ("同一个未入索引的目录、同一台相机（未核实是同一卷）" if one else
                               f"目录名分组，含 {len(cams[k] - {None})} 种相机或只有 1 帧，不能当作一卷")
            if not one:
                r["roll"] = None
        else:
            r["roll_level"] = "unknown"
            r["roll_basis"] = "馆藏单张底片，不成卷" if r["source"] == "auckland_museum" else "零散提供，没有卷信息"
        if r["source"] != "blueneg":
            r.setdefault("in_research_manifest", bool(r.get("in_s3_manifest")))
        r.setdefault("in_s3_manifest", False)
        r.setdefault("p4_label", False)
        r.setdefault("user_label", False)
        r["usable_negative"] = not r["exclude"]
        r["has_reference"] = r["ref_strength"] != "none"

    # ---- rolls ----
    rolls = defaultdict(list)
    for r in frames:
        if r["roll"]:
            rolls[r["roll"]].append(r)
    roll_rows = []
    for k, rs in sorted(rolls.items()):
        neg = [r for r in rs if r["usable_negative"]]
        kinds = Counter(r["kind"] for r in neg)
        kind = kinds.most_common(1)[0][0] if neg else rs[0]["kind"]
        # key-number count: frames of the roll's main kind only (nlp-20180728 holds 2 colour + 1 B&W)
        dev_ok = [r for r in neg if r["kind"] == kind and r["partition"] == "dev" and r["engine_readable"]
                  and r["has_reference"]]
        g = Counter(x for r in neg if r["partition"] == "dev" for x in r["graders"])
        used = any(r.get("in_research_manifest") for r in rs) or k in viewed_rolls
        roll_rows.append(dict(
            roll=k, level=rs[0]["roll_level"], basis=rs[0]["roll_basis"], source=rs[0]["source"],
            kind=kind, kinds=dict(kinds),
            films=sorted({str(r["film"]) for r in rs}), cameras=sorted({str(r["camera"]) for r in rs}),
            partition="frozen" if all(r["partition"] == "frozen" for r in rs) else
            "dev" if all(r["partition"] == "dev" for r in rs) else "mixed",
            frames=len(rs), negatives=len(neg), engine_readable=sum(r["engine_readable"] for r in neg),
            with_reference=sum(r["has_reference"] for r in neg), dev_readable_with_reference=len(dev_ok),
            graders=dict(g), two_graders=sum(len([x for x in r["graders"] if "用户" in x]) > 0 and
                                              any("冲扫店" in x for x in r["graders"]) for r in neg),
            p4_labels=sum(r["p4_label"] for r in rs), user_labels=sum(r["user_label"] for r in rs),
            ageing=sum(bool(r["ageing"]) for r in neg), used_in_research=used,
            frame_order=rs[0].get("frame_order"),
            **({"own_folder_check": own_check[k]} if k in own_check else {}),
            **({"release": {x: brolls[k][x] for x in ("release_roll_ids", "dates", "release_frames", "dup_numbers")}}
               if k in brolls else {})))

    # ---- photo folder ----
    pm = json.loads((REPO / "experiments/s3_photos_audit/match.json").read_text())
    ph = defaultdict(Counter)
    for p in pm["photos"]:
        parts = p["photo"].split("/")
        top = "（根目录的散文件）" if len(parts) == 1 else "/".join(parts[:2]) if parts[0] == "新疆" else parts[0]
        ph[top]["files"] += 1
        same = p.get("r", 0) >= pm["same_frame_r"] and p.get("margin", 0) >= 0.05 and p.get("best")
        ph[top]["same_frame_as_material"] += bool(same)
        ph[top]["user_edit"] += "编辑" in p["photo"]
        ph[top]["frozen_copy"] += bool(p.get("frozen_roll"))

    # ---- summary ----
    def cnt(sel):
        return len(sel)
    col = [r for r in frames if r["kind"] == "colour_neg" and r["usable_negative"]]
    bw = [r for r in frames if r["kind"] == "bw_neg" and r["usable_negative"]]
    mat_col = [r for r in col if r["where"].startswith("material")]
    key = [r for r in col if r["partition"] == "dev" and r["engine_readable"] and r["has_reference"]]
    lvl = lambda sel: dict(Counter(r["roll_level"] for r in sel))
    rolls_of = lambda sel, level: sorted({r["roll"] for r in sel if r["roll"] and r["roll_level"] == level})
    summary = dict(
        reconciliation=dict(
            material_colour_negatives=cnt(mat_col), material_colour_with_reference=sum(r["has_reference"] for r in mat_col),
            material_colour_with_reference_dev=sum(r["has_reference"] and r["partition"] == "dev" for r in mat_col),
            material_colour_with_reference_frozen=sum(r["has_reference"] and r["partition"] == "frozen" for r in mat_col),
            data_raw_colour_negatives_not_in_material=sum(r["where"] == "data/raw" and not r.get("dup_of_material")
                                                          for r in col),
            data_raw_duplicates_of_material=sum(r["where"] == "material+data/raw" for r in frames),
            all_colour_negatives=cnt(col), all_colour_with_reference=sum(r["has_reference"] for r in col),
            bw_negatives=cnt(bw), bw_with_reference=sum(r["has_reference"] for r in bw),
            blueneg_release_frames=len(meta), blueneg_release_frames_not_in_material=len(
                set(bmeta) - {r["id"].split(":", 1)[1] for r in frames if r["source"] == "blueneg"}),
            photo_folder_files=sum(c["files"] for c in ph.values()),
            photo_folder_same_frame_as_material=sum(c["same_frame_as_material"] for c in ph.values()),
            excluded=dict(Counter(x for r in frames for x in r["exclude"]))),
        key_number=dict(
            definition="彩负，不在排除之列，开发侧（非冻结），引擎现在能读（素材或 data/raw 的 RAW / DNG），至少 1 个参考",
            frames=len(key), by_roll_level=lvl(key), by_source=dict(Counter(r["source"] for r in key)),
            verified_rolls=rolls_of(key, "verified"), inferred_rolls=rolls_of(key, "inferred"),
            frames_without_roll=sum(r["roll"] is None for r in key),
            with_ageing=sum(bool(r["ageing"]) for r in key),
            with_two_graders=sum(any("用户" in g for g in r["graders"]) and any("冲扫店" in g for g in r["graders"])
                                 for r in key),
            used_in_research=sum(bool(r.get("in_research_manifest")) for r in key),
            s3_design_excluded_or_failed=sum(r.get("s3_status", "").startswith(("设计排除", "处理失败")) for r in key),
            in_untouched_verified_rolls=sum(not next(x["used_in_research"] for x in roll_rows if x["roll"] == r["roll"])
                                            for r in key if r["roll"] and r["roll_level"] == "verified")),
        blueneg_dev_with_reference=dict(Counter(
            "蓝层变质" if r["ageing"] else "蓝层完好、DNG 已下载" if r["engine_readable"] else "蓝层完好、DNG 未下载"
            for r in col if r["source"] == "blueneg" and r["partition"] == "dev" and r["has_reference"])),
        roll_labels=dict(
            colour_nonempty_labels=len({r["roll_label"] for r in col if r["roll_label"] and r["where"] == "material"}),
            note="非空卷标签只数素材目录里 catalog 的卷字段（含按目录名推测的分组），与 10-18 报告同一口径",
            colour_rolls_by_level={lv: len(rolls_of(col, lv)) for lv in ("verified", "inferred")},
            bw_nonempty_labels=len({r["roll_label"] for r in bw if r["roll_label"] and r["where"] == "material"}),
            bw_rolls_by_level={lv: len(rolls_of(bw, lv)) for lv in ("verified", "inferred")},
            frames_by_level_colour=lvl(col), frames_by_level_bw=lvl(bw)),
        photo_folders={k: dict(v) for k, v in sorted(ph.items())},
        blueneg_groups={k: v for k, v in sorted(brolls.items())},
    )
    lines = ["| 卷 | 核实 | 来源 | 片种 | 开发/冻结 | 负片 | 引擎可读 | 有参考 | 开发侧可读且有参考 | 两套调色 | P4 标签 | 用户标签 | 老化 | 研究用过 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in roll_rows:
        lines.append(f"| `{x['roll']}` | {x['level']} | {x['source']} | {'彩' if x['kind'] == 'colour_neg' else '黑白'} "
                     f"| {x['partition']} | {x['negatives']} | {x['engine_readable']} | {x['with_reference']} "
                     f"| {x['dev_readable_with_reference']} | {x['two_graders']} | {x['p4_labels']} | {x['user_labels']} "
                     f"| {x['ageing']} | {'是' if x['used_in_research'] else '否'} |")
    (out / "rolls.md").write_text("\n".join(lines) + "\n")
    for name, obj in (("frames.json", frames), ("rolls.json", roll_rows), ("summary.json", summary)):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
