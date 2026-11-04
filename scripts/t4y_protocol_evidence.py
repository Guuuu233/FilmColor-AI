"""Record T4y input provenance without decoding original images."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4y"
COMMITS = {
    "t4o": "4cd76b7",
    "t4r": "fbd48a0",
    "t4s": "7c5107c",
    "t4t": "567aaf9",
    "t4u": "98ded3c",
    "t4v": "0d1d730",
    "t4w": "b3a7510",
    "t4x": "f6340aa",
}
TASK = Path("/Users/davidliu/Downloads/FilmColor_AI_任务书_T4y_主干快进合入与佳能索尼相机矩阵解封及短跨度亮暗部正曝光阻尼与扫描仪准饱和片基隔离_2026-11-04.md")
RESEARCH = Path("/Users/davidliu/Downloads/fresh_batches_t4x.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / "experiments/t4_auto_ui/t4x/input_protocol_sha256.json").read_text())
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
        assert len(rows) == 38 and len({row["rel"] for row in rows}) == 38

    result = dict(
        historical_json_sha256=history,
        frozen_manifest_sha256=old["frozen_manifest_sha256"],
        task_sha256=sha(TASK),
        research_json_sha256=research_sha,
        research_record_count=research_count,
        research_claims_independently_verified=False,
        main_fast_forward=dict(
            before="b3a751038958600a99f0fb9345852830b2de2d58",
            after="f6340aa5fe053332a1d383702b3ad218f0c81624",
        ),
        specification_decisions=dict(
            camera_matrix_unblock="MATRIX_DIAG (0.5, 2.4), MATRIX_OFF_DIAG (-1.6, 0.8) to admit Canon EOS & Sony A7/A7II",
            scanner_base_isolation="exclude_near_saturated=False for scanner path (scanner.py & app.py is_scan)",
            overlift_bright_check="short-span bright scenes (span_d_g < 0.76, share >= 0.20, ev >= 0.45, q25 in [0.475, 0.68]) damp ev & blacks",
        ),
    )
    (OUT / "input_protocol_sha256.json").write_text(json.dumps(result, indent=2) + "\n")
    print(dict(historical_json_count=len(history), research_records=research_count, frozen_manifests=len(old["frozen_manifest_sha256"])))


if __name__ == "__main__":
    main()
