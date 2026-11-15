"""Verify early retrieval-buffer release against the completed 32-chunk control."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.modal_lingbot_roundtrip import app, generate_h100


@app.local_entrypoint(name="read-lifetime")
def main():
    result = generate_h100.remote("turn_long_wait_return", 42, True, 18, "GC007",
                                  False, "virtual", 60, 4, "nearest", True)
    output = ROOT / "reproduction/lingbot-read-cache-lifetime"
    output.mkdir(exist_ok=True)
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    reference_path = ROOT / "reproduction/multiview-retrieval/nearest4/run.json"
    reference = json.loads(reference_path.read_text())
    old, new = reference["run"], result["run"]
    fields = ("source_revision", "checkpoint_revision", "task_id", "seed", "fixed_actions",
              "local_attn_size", "sink_size", "hardware", "text_encoder_offloaded_after_encoding",
              "turn_degrees_per_second", "max_chunks", "controls_sha256", "sampler_timesteps_index")
    matched = {field: old[field] == new[field] for field in fields}
    hashes = [a["pixel_sha256"] == b["pixel_sha256"] for a, b in zip(old["chunks"], new["chunks"])]
    released = [row["temporary_read_bytes_released_before_decode"] for row in new["memory"]["trace"]]
    report = {"scope": "GPU read-buffer lifetime optimization; not a quality change",
              "reference": str(reference_path.relative_to(ROOT)),
              "reference_directory": reference["result_directory"], "candidate_directory": result["result_directory"],
              "matched_settings": matched, "reference_status": old["status"], "candidate_status": new["status"],
              "reference_chunks": len(old["chunks"]), "candidate_chunks": len(new["chunks"]),
              "chunk_pixels_identical": hashes,
              "all_pixels_identical": all(hashes) and len(hashes) == 32 and len(new["chunks"]) == 32,
              "reference_peak_allocated_bytes": old["peak_allocated_bytes"],
              "candidate_peak_allocated_bytes": new["peak_allocated_bytes"],
              "max_temporary_gpu_read_bytes_released_before_decode": max(released),
              "reference_warm_suffix_s": old["chunks"][-1]["elapsed_s"] - old["chunks"][0]["elapsed_s"],
              "candidate_warm_suffix_s": new["chunks"][-1]["elapsed_s"] - new["chunks"][0]["elapsed_s"],
              "caveat": "One run per mode on separate H100 containers; timing variability remains. Protocol/player code changed to v6 between runs, but fixed controls bypass the player; generated raw pixels are checked directly."}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(matched.values()) or not report["all_pixels_identical"]:
        raise ValueError("Buffer lifetime parity did not pass; candidate artifacts preserved")
