"""Exercise the report builder with explicitly substituted historical evaluation rows in temporary output.
This checks aggregation only; it does not produce formal regression evidence.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4u_report_tables as report


def main():
    original = report.OUT
    with tempfile.TemporaryDirectory(prefix=".t4u_report_smoke_", dir=report.ROOT / "experiments/t4_auto_ui") as temp:
        report.OUT = Path(temp)
        for p in original.iterdir():
            if p.is_file() and p.suffix in (".json", ".log") and not p.name.startswith("candidate_"):
                shutil.copyfile(p, report.OUT / p.name)
        for part in ("a", "b", "cross"):
            record = report.committed_json("t4t", part)
            record["src_content_sha256"] = report.source_sha()
            (report.OUT / f"candidate_{part}.json").write_text(json.dumps(record))
            (report.OUT / f"candidate_{part}.log").write_text("1 peak memory footprint\n")
        report.main()
        result = json.loads((report.OUT / "acceptance.json").read_text())
        assert result["all_passed"] and not result["devices"]["failures"]
        print("ISOLATED SMOKE ONLY: aggregation works; formal evaluation output was not touched")


if __name__ == "__main__":
    main()
