"""List completed episodes without a usable Gemini score (read-only).

    python scripts/check_scoring_coverage.py --label full1
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="full1")
    args = parser.parse_args()
    base = ROOT / "outputs/adaptive" / args.label
    status = json.loads((base / "status.json").read_text())
    missing, errors, total = [], [], 0
    for arm, data in status["arms"].items():
        for row in data["completed"]:
            total += 1
            identifier = f"{arm}-{row['split']}-{row['task_id']}-seed{row['seed']}"
            path = base / "gemini/episodes" / f"{identifier}.json"
            if not path.is_file():
                missing.append(identifier)
                continue
            result = json.loads(path.read_text())
            if result.get("error") or (result.get("scores") or {}).get("official") is None:
                errors.append((identifier, str(result.get("error"))[:120]))
    print(json.dumps({"completed_rows": total, "missing_gemini_file": missing, "unscored": errors}, indent=1))


if __name__ == "__main__":
    main()
