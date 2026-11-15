"""Clean-room static director texts for every PlayWorld case (one LLM call per case; whitelisted inputs only).

    ANTHROPIC_BASE_URL=... ANTHROPIC_API_KEY=... python scripts/director_generate_static_all.py \
        --out reproduction/director-static-texts-all.json [--reuse reproduction/director-dev-texts-llm.json]

`--reuse` keeps the static texts already used in the development experiment so the frozen run uses the same
texts for those cases. Output has the PresetDirector layout (use with --director static --director-text-file).
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync.lingbot_director import DEFAULT_MODEL, Director  # noqa: E402

SPLITS = ("gc", "if", "insight", "outsight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    from anthropic import Anthropic
    client = Anthropic(timeout=120, max_retries=6)
    reused = json.loads(args.reuse.read_text())["cases"] if args.reuse else {}
    records = [(split, record) for split in SPLITS
               for record in json.loads((ROOT / f"external/PlayWorld/data/{split}/data.json").read_text())]

    def one(item):
        split, record = item
        key = f"{split}:{record['task_id']}"
        if key in reused:
            return key, {"static": reused[key]["static"], "source": "development experiment text"}
        director = Director(client, record, "static", model=args.model)
        text = director.initial()
        return key, {"static": text, "llm_error": bool(director.metadata()["errors"])}

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        cases = dict(pool.map(one, records))
    errors = sorted(key for key, entry in cases.items() if entry.get("llm_error"))
    payload = {"note": f"Static director texts written by {args.model} via worldsync.lingbot_director.Director from "
                       "image_caption, prompt and action_sequence_steps only (no images, rubric, questions or judge text). "
                       "Cases listed in llm_errors fell back to the plain caption.",
               "model": args.model, "llm_errors": errors, "cases": cases}
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(cases), "llm_errors": len(errors), "reused": sum("source" in e for e in cases.values())}))


if __name__ == "__main__":
    main()
