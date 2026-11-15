"""Compare the `replay` arm (same text, same fixed actions) with the stock full-run episode, chunk by chunk."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "worldsync-lingbot-results"


def main():
    status = json.loads((ROOT / "outputs/director-dev-status.json").read_text())
    cases = {(c["split"], c["task_id"]): c for c in json.loads((ROOT / "outputs/director-dev-cases.json").read_text())}
    out = []
    for row in status["arms"].get("replay", {}).get("completed", []):
        target = ROOT / "outputs/director-dev/episodes" / f"replay-{row['split']}-{row['task_id']}-seed42"
        target.mkdir(parents=True, exist_ok=True)
        if not (target / "run.json").is_file():
            subprocess.run(["modal", "volume", "get", VOLUME, f"{row['result_directory']}/run.json", str(target) + "/", "--force"],
                           capture_output=True, text=True)
        replay = json.loads((target / "run.json").read_text())
        stock = json.loads((Path(cases[(row["split"], row["task_id"])]["stock_video"]).parent / "run.json").read_text())
        a = [chunk["pixel_sha256"] for chunk in replay["chunks"]]
        b = [chunk["pixel_sha256"] for chunk in stock["chunks"]]
        matching = sum(x == y for x, y in zip(a, b))
        out.append({"case": f"{row['split']}:{row['task_id']}", "replay_chunks": len(a), "stock_chunks": len(b),
                    "identical_chunks": matching, "replay_gpu": (replay.get("hardware") or {}).get("gpu"),
                    "stock_gpu": (stock.get("hardware") or {}).get("gpu")})
    print(json.dumps(out, indent=1))
    (ROOT / "outputs/director-dev/replay-check.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
