"""Disable the failed optional DLPack addon and check complete-rollout parity."""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.modal_lingbot_runtime_benchmark import app, generate


@app.local_entrypoint(name="no-dlpack")
def main():
    output = ROOT / "reproduction/lingbot-runtime-benchmark"
    result = generate.remote("fa4", True)
    (output / "fa4-no-addon.json").write_text(json.dumps(result, indent=2) + "\n")
    old = json.loads((output / "fa4.json").read_text())["run"]
    new = result["run"]
    matches = [a["pixel_sha256"] == b["pixel_sha256"] for a, b in zip(old["chunks"], new["chunks"])]
    intervals = [new["chunks"][i]["elapsed_s"] - new["chunks"][i - 1]["elapsed_s"] for i in range(6, 10)]
    report = {"scope": "optional addon startup diagnostic; same FA4 math and checkpoint",
              "status": result["status"], "candidate_directory": result["result_directory"],
              "identical_chunk_pixels": matches, "all_pixels_identical": len(matches) == 10 and all(matches),
              "reference_first_chunk_elapsed_s": old["chunks"][0]["elapsed_s"],
              "candidate_first_chunk_elapsed_s": new["chunks"][0]["elapsed_s"],
              "candidate_warmed_intervals_s": intervals, "candidate_warmed_mean_s": statistics.mean(intervals),
              "candidate_peak_allocated_bytes": new["peak_allocated_bytes"],
              "caveat": "Independent containers; cold-start I/O and scheduling variability remain. Only attribute a reproducible reduction to addon suppression after further measurement."}
    (output / "no-addon-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if result["status"] != "completed" or new["hardware"] != old["hardware"] or not report["all_pixels_identical"]:
        raise ValueError("Optional addon diagnostic did not preserve all ten chunks; artifacts retained")
