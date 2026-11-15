"""Copy memory-v2 step-2 per-case scores (without local paths) into docs/demo/sources and summarize them."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_demo import bootstrap  # noqa: E402


def main(source):
    rows = list(csv.DictReader(open(source)))
    kept = [{k: v for k, v in row.items() if not k.endswith("_video")} for row in rows]
    target = ROOT / "docs/demo/sources/memory-v2-step2-per-case.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kept[0].keys()))
        writer.writeheader()
        writer.writerows(kept)
    paired = [row for row in rows if row["arm"] == "memv2" and row["paired_difference"] not in ("", "None")]
    report = {}
    for name, keep in (("all", lambda r: True), ("held_out", lambda r: r["development"] == "False"),
                       ("gc", lambda r: r["split"] == "gc"), ("outsight", lambda r: r["split"] == "outsight")):
        delta = [float(r["paired_difference"]) for r in paired if keep(r)]
        report[name] = {"pairs": len(delta), "delta": bootstrap(delta), "wins": sum(d > 1e-9 for d in delta),
                        "losses": sum(d < -1e-9 for d in delta),
                        "stock": float(np.mean([float(r["stock_mean"]) for r in paired if keep(r)])),
                        "memv2": float(np.mean([float(r["memv2_mean"]) for r in paired if keep(r)]))}
        report[name]["ties"] = len(delta) - report[name]["wins"] - report[name]["losses"]
    (ROOT / "docs/demo/memv2-step2-summary.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
