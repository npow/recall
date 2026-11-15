"""Clean-room director texts for fixed-action replays, written by the LLM director (no images, no rubric).

For a fixed commanded action list the director's inputs (caption, objective, reference plan, elapsed seconds,
commanded action, commanded camera state) do not depend on generated frames, so running it offline gives exactly
the texts an online run would produce. Output has the PresetDirector file layout.

    ANTHROPIC_BASE_URL=... ANTHROPIC_API_KEY=... python scripts/director_generate_texts.py --out reproduction/director-dev-texts-llm.json
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldsync.lingbot_controls import CameraPlan  # noqa: E402
from worldsync.lingbot_director import DEFAULT_MODEL, Director  # noqa: E402


def task_record(split, task_id):
    return next(r for r in json.loads((ROOT / f"external/PlayWorld/data/{split}/data.json").read_text()) if r["task_id"] == task_id)


def texts_for(client, case, model, interval):
    record = task_record(case["split"], case["task_id"])
    static = Director(client, record, "static", model=model)
    entry = {"static": static.initial(), "errors": static.metadata()["errors"]}
    if case["split"] in ("insight", "outsight"):
        director = Director(client, record, "unseen", model=model, interval=interval)
        schedule = [[0, director.initial()]]
        camera = CameraPlan()
        for index, action in enumerate(case["stock_actions"]):
            camera.advance(action)
            text, changed = director.for_chunk(index, action, camera.commanded_state(), max(0.0, index - 0.1875) if index else 0.0)
            if changed:
                schedule.append([index, text])
        entry["scheduled"] = schedule
        entry["errors"] += director.metadata()["errors"]
        entry["world_states"] = [call.get("world_state") for call in director.calls if "world_state" in call]
    return f"{case['split']}:{case['task_id']}", entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    from anthropic import Anthropic
    client = Anthropic(timeout=120, max_retries=5)
    cases = json.loads((ROOT / "outputs/director-dev-cases.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        results = dict(pool.map(lambda case: texts_for(client, case, args.model, args.interval), cases))
    errors = sum(entry["errors"] for entry in results.values())
    payload = {"note": f"Clean-room director texts written by {args.model} through worldsync.lingbot_director.Director "
                       "(whitelisted inputs only: image_caption, prompt, action_sequence_steps, elapsed seconds, commanded "
                       "action and commanded camera state of the fixed stock action list). No images, rubric or judge text.",
               "model": args.model, "interval_chunks": args.interval, "llm_errors": errors, "cases": results}
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(results), "llm_errors": errors,
                      "segments": {key: len(value.get("scheduled", [])) for key, value in results.items()}}))


if __name__ == "__main__":
    main()
