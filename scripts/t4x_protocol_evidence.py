"""Record T4x input provenance without decoding original images."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4x"
COMMITS = {
    "t4o": "4cd76b7",
    "t4r": "fbd48a0",
    "t4s": "7c5107c",
    "t4t": "567aaf9",
    "t4u": "98ded3c",
    "t4v": "0d1d730",
    "t4w": "b3a7510",
}
TASK = Path("/Users/davidliu/Downloads/FilmColor_AI_任务书_T4x_主干快进合入与窄蓝救援有界封顶及过曝灰平负片恢复与中低正曝光阻尼扩展及索尼压缩RAW片基防夹顶_2026-11-03.md")
RESEARCH = Path("/Users/davidliu/Downloads/fresh_batches_t4w.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / "experiments/t4_auto_ui/t4w/input_protocol_sha256.json").read_text())
    history = {}
    for version, commit in COMMITS.items():
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit, f"experiments/t4_auto_ui/{version}"],
            cwd=ROOT,
            text=True,
        ).splitlines()
        for name in names:
            p = ROOT / name
            if p.suffix == ".json" and p.parent == ROOT / f"experiments/t4_auto_ui/{version}":
                data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
                assert data == p.read_bytes(), name
                history[name] = hashlib.sha256(data).hexdigest()
    for name, digest in old["frozen_manifest_sha256"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / "src/filmcolor/paths.py") == "fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913"
    research = json.loads(RESEARCH.read_text())
    rows = [row for batch in research["batches"].values() for row in batch]
    assert len(rows) == 50 and len({row["rel"] for row in rows}) == 50
    result = dict(
        historical_json_sha256=history,
        frozen_manifest_sha256=old["frozen_manifest_sha256"],
        task_sha256=sha(TASK),
        research_json_sha256=sha(RESEARCH),
        research_record_count=len(rows),
        research_claims_independently_verified=False,
        main_fast_forward=dict(
            before="0d1d7301339e011848cd7a2215ab0eaedffa7e4e",
            after=subprocess.check_output(["git", "rev-parse", "main"], cwd=ROOT, text=True).strip(),
        ),
        specification_decisions=dict(
            bounded_blue_rescue="ratio < 0.28 capped at 0.28 * d_hi_G with 0.85 convergence",
            sony_craw_base="exclude sp_max in (0.9985, 1.0) when remaining >= 100",
            thin_flat_neg="span_d_g < 0.60, q25 >= 0.62, iqr <= 0.145, pull ev up to -0.65 and blacks down to -0.18",
            ultra_flat_neg="range_stops >= 1.20, q25 >= 0.72, iqr <= 0.068, pull ev up to 0.55, blacks down to 0.30, contrast cap 1.75",
            overlift_mod_expansion="ev_min 0.70, ev_pivot 0.58, ev_ramp 0.55",
        ),
    )
    (OUT / "input_protocol_sha256.json").write_text(json.dumps(result, indent=2) + "\n")
    print(dict(historical_json_count=len(history), research_records=len(rows), frozen_manifests=len(old["frozen_manifest_sha256"])))


if __name__ == "__main__":
    main()
