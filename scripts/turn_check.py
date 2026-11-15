"""Judge-independent turn check: did the generated video turn the way the plan says?

Measures realized yaw per chunk from the frames (scripts/measure_turn_rate.py) and compares
the yaw path to the reference plan's Left/Right holds (50 deg/s, the pacing the human plans
imply). Checks direction of the first turn, peak size, and return to the start when the plan
returns. Static-scene feature matching: reported as unreliable on low-texture clips.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_turn_rate import measure  # noqa: E402
from worldsync.reference_player import plan_steps  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "external/PlayWorld/data"
PLAN_RATE = 50.0


def task_record(task_id):
    for split in ("gc", "if", "insight", "outsight"):
        for record in json.loads((DATA / split / "data.json").read_text()):
            if record["task_id"] == task_id:
                return record
    raise KeyError(task_id)


def expected_path(task):
    """Signed expected yaw (right positive) after each plan step that turns."""
    total, path = 0.0, []
    for step in plan_steps(task):
        if step["action"] in ("LEFT", "RIGHT"):
            total += (1 if step["action"] == "RIGHT" else -1) * PLAN_RATE * step["seconds"]
            path.append(round(total, 1))
    return path


def check(directory):
    run = json.loads((directory / "run.json").read_text())
    task = task_record(run["task_id"])
    expected = expected_path(task)
    if not expected:
        return {"task_id": run["task_id"], "applicable": False}
    actions = [a["action"] for a in run["actions"]]
    rows, _ = measure(directory, actions)
    failed = sum(r["failed_pairs"] for r in rows)
    pairs = sum(max(r["frames"] // 3, 1) for r in rows)
    cumulative, total = [], 0.0
    for row in rows:
        total += row["realized_yaw_deg"]
        cumulative.append(round(total, 1))
    peak = max(cumulative, key=abs) if cumulative else 0.0
    first_target = expected[0]
    significant = next((c for c in cumulative if abs(c) > 20), None)
    direction_ok = significant is not None and (significant > 0) == (first_target > 0)
    extreme = max([abs(x) for x in expected] + [0])
    size_ok = extreme > 0 and 0.6 <= abs(peak) / extreme <= 1.6
    final_target = expected[-1]
    final = cumulative[-1] if cumulative else 0.0
    final_ok = abs(final - final_target) <= max(30.0, 0.25 * extreme)
    return {"task_id": run["task_id"], "applicable": True, "expected_yaw_path": expected, "realized_cumulative_yaw": cumulative,
        "direction_ok": direction_ok, "size_ok": size_ok, "final_ok": final_ok,
        "turn_check_pass": bool(direction_ok and size_ok and final_ok),
        "reliable": failed / max(pairs, 1) < 0.3, "failed_pair_fraction": round(failed / max(pairs, 1), 2),
        "commanded": actions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {d.name: check(d) for d in args.directories}
    for name, r in results.items():
        if not r["applicable"]:
            print(name, "no turn in plan")
            continue
        print(name, "pass" if r["turn_check_pass"] else "FAIL", "dir", r["direction_ok"], "size", r["size_ok"], "final", r["final_ok"],
              "reliable", r["reliable"], "expected", r["expected_yaw_path"], "realized", r["realized_cumulative_yaw"])
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
