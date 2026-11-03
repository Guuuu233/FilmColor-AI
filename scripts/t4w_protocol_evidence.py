"""Record T4w input provenance without decoding original images."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4w"
COMMITS = {"t4o": "4cd76b7", "t4r": "fbd48a0", "t4s": "7c5107c", "t4t": "567aaf9", "t4u": "98ded3c", "t4v": "0d1d730"}
TASK = Path("/Users/davidliu/Downloads/FilmColor_AI_任务书_T4w_主干快进合入与中等高曝光暗部过抬阻尼及过曝负片橙罩片基防误剔与窄蓝跨度阈值扩展_2026-11-03.md")
RESEARCH = Path("/Users/davidliu/Downloads/fresh_batches_t4v.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / "experiments/t4_auto_ui/t4v/input_protocol_sha256.json").read_text())
    history = {}
    for version, commit in COMMITS.items():
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit,
                                         f"experiments/t4_auto_ui/{version}"], cwd=ROOT, text=True).splitlines()
        for name in names:
            p = ROOT / name
            if p.suffix == ".json" and p.parent == ROOT / f"experiments/t4_auto_ui/{version}":
                data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
                assert data == p.read_bytes(), name
                history[name] = hashlib.sha256(data).hexdigest()
    for name, digest in old["frozen_manifest_sha256"].items():
        assert sha(ROOT / name) == digest
    assert sha(ROOT / "src/filmcolor/paths.py") == "fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913"
    research = json.loads(RESEARCH.read_text())
    rows = [row for batch in research["batches"].values() for row in batch]
    assert len(rows) == 62 and len({row["rel"] for row in rows}) == 62
    result = dict(historical_json_sha256=history, frozen_manifest_sha256=old["frozen_manifest_sha256"],
                  task_sha256=sha(TASK), research_json_sha256=sha(RESEARCH),
                  research_record_count=len(rows), research_claims_independently_verified=False,
                  main_fast_forward=dict(before="98ded3cc5a2aff9a82d5d0378eee7caa19b4cef7",
                                         after=subprocess.check_output(["git", "rev-parse", "main"], cwd=ROOT, text=True).strip()),
                  specification_decisions=dict(narrow_blue_rg_threshold=0.96,
                                               improvement_counts="measure; do not assume 26 or 28",
                                               moderate_branch="task elif without extra EV upper limit"))
    (OUT / "input_protocol_sha256.json").write_text(json.dumps(result, indent=2) + "\n")
    print(dict(historical_json_count=len(history), research_records=len(rows), frozen_manifests=len(old["frozen_manifest_sha256"])))


if __name__ == "__main__":
    main()
