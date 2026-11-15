"""Write reproduction/full-run-status.json: generated / scored / failed counts per arm and dimension.

    python scripts/full_run_status.py --label full1 [--loop 600]

Reads outputs/adaptive/<label>/status.json (refreshed by the scoring pipeline each cycle) and the
Gemini episode files. No Modal or Gemini calls.
"""
import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("gc", "if", "insight", "outsight")
EXPECTED = {"gc": 48, "if": 50, "insight": 30, "outsight": 43}


def build(label):
    base = ROOT / "outputs/adaptive" / label
    status = json.loads((base / "status.json").read_text()) if (base / "status.json").is_file() else {"arms": {}}
    out = {"label": label, "updated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "expected_per_arm_dimension": EXPECTED, "arms": {}}
    scored = {}
    episodes = base / "gemini/episodes"
    errors = 0
    if episodes.is_dir():
        for path in episodes.glob("*.json"):
            result = json.loads(path.read_text())
            if result.get("error") or (result.get("scores") or {}).get("official") is None:
                errors += 1
                continue
            arm, split = path.stem.split("-")[0], path.stem.split("-")[1]
            scored.setdefault(arm, {}).setdefault(split, []).append(float(result["scores"]["official"]))
    for arm in ("stock", "budget"):
        data = status["arms"].get(arm, {"completed": [], "failed": [], "completed_count": 0})
        generated, failed = {}, {}
        for row in data.get("completed", []):
            generated[row["split"]] = generated.get(row["split"], 0) + 1
        for row in data.get("failed", []):
            split = row.get("split") if isinstance(row, dict) else None
            if split:
                failed[split] = failed.get(split, 0) + 1
        out["arms"][arm] = {
            "generated": {s: generated.get(s, 0) for s in SPLITS}, "generated_total": sum(generated.values()),
            "failed": {s: failed.get(s, 0) for s in SPLITS}, "failed_total": sum(failed.values()),
            "scored": {s: len(scored.get(arm, {}).get(s, [])) for s in SPLITS},
            "scored_total": sum(len(v) for v in scored.get(arm, {}).values()),
            "throughput": data.get("throughput")}
    out["gemini_error_or_unscored_files"] = errors
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--loop", type=int, default=0, help="seconds between updates; 0 = once")
    args = parser.parse_args()
    while True:
        (ROOT / "reproduction/full-run-status.json").write_text(json.dumps(build(args.label), indent=1) + "\n")
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
