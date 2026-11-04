"""Record T4z input provenance without decoding original images."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4z"
COMMITS = {
    "t4o": "4cd76b7",
    "t4r": "fbd48a0",
    "t4s": "7c5107c",
    "t4t": "567aaf9",
    "t4u": "98ded3c",
    "t4v": "0d1d730",
    "t4w": "b3a7510",
    "t4x": "f6340aa",
    "t4y": "4bd014e",
}
TASK = Path("/Users/davidliu/Downloads/FilmColor_AI_任务书_T4z_主干快进合入与夜景逆光双重提亮阻尼及灰平负片扩展与窗格遮挡白点桥接_2026-11-04.md")
RESEARCH = Path("/Users/davidliu/Downloads/fresh_batches_t4y.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / "experiments/t4_auto_ui/t4y/input_protocol_sha256.json").read_text())
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

    research_count = None
    research_sha = None
    if RESEARCH.is_file():
        research = json.loads(RESEARCH.read_text())
        rows = [row for batch in research["batches"].values() for row in batch]
        research_count = len(rows)
        research_sha = sha(RESEARCH)
        assert len(rows) == 41 and len({row["rel"] for row in rows}) == 41

    result = dict(
        historical_json_sha256=history,
        frozen_manifest_sha256=old["frozen_manifest_sha256"],
        task_sha256=sha(TASK),
        research_json_sha256=research_sha,
        research_record_count=research_count,
        research_claims_independently_verified=False,
        main_fast_forward=dict(
            before="f6340aa5fe053332a1d383702b3ad218f0c81624",
            after="4bd014ea956c7a569b0e2113ded6f054c6f1a34e",
        ),
        specification_decisions=dict(
            night_backlit_double_damp="overlift_night_bl_check with w_bl >= 0.65, bright_share < 0.090, ev >= 1.50, q25 >= 0.175",
            thin_flat_extension="thin_flat_check extended with span_max=0.72, iqr_max=0.20, iqr_pivot=0.20",
            bridged_holder_white_point="invert.block_white_point binary_dilation(hot,1) bridging mullions and _metering_mask drops holder pixels",
        ),
    )
    (OUT / "input_protocol_sha256.json").write_text(json.dumps(result, indent=2) + "\n")
    print(dict(historical_json_count=len(history), research_records=research_count, frozen_manifests=len(old["frozen_manifest_sha256"])))


if __name__ == "__main__":
    main()
