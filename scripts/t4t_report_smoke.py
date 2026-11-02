"""Exercise the complete table builder with historical candidate rows in an isolated temporary output.
Does not create acceptance in the real output directory or claim that regressions finished.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4t_report_tables as report


def main():
    original = report.OUT
    with tempfile.TemporaryDirectory(prefix=".t4t_report_smoke_", dir=report.ROOT / "experiments/t4_auto_ui") as temp:
        report.OUT = Path(temp)
        for p in original.iterdir():
            if p.is_file() and p.suffix in (".json", ".log") and not p.name.startswith("candidate_"):
                shutil.copyfile(p, report.OUT / p.name)
        for part in ("a", "b", "cross"):
            record = report.committed_json("t4s", part)
            record["src_content_sha256"] = report.source_sha()
            (report.OUT / f"candidate_{part}.json").write_text(json.dumps(record))
            (report.OUT / f"candidate_{part}.log").write_text("1 peak memory footprint\n")
        report.main()
        result = json.loads((report.OUT / "acceptance.json").read_text())
        expected = {"A1040.TIF:green_q50_gt_0_05", "A239.ORF:tint_lt_0_22", "A1296:whole_image_any_gt_10_zero"}
        assert set(result["devices"]["failures"]) == expected
        assert not result["all_passed"]
        print("ISOLATED SMOKE ONLY: complete builder ran; all three known device failures retained")


if __name__ == "__main__":
    main()
