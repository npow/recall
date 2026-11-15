"""Write jobs.jsonl for the near-field motion-realization sweep (stock 18/6, seed 42, H100-safe).

Scenes: gate-failure scenes first (GC017, GC015, GC024, GC007, GC005, GC027), controls (GC001, GC007), and
near-field scenes chosen by looking at the start images (GC011, GC012, GC028, GC014, GC023, GC025).
Priority order so a partial run is still usable: forward + backward on every scene, then turn in place,
then strafes (left/right) and the round trip on a subset.
Paths: fwd8 = WAIT + W x8, bwd8 = WAIT + S x8, left8/right8 = A/D x8, turn180 = RIGHT x3, round_trip.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldsync.lingbot_paths import PATHS  # noqa: E402

SCENES = ["GC017", "GC015", "GC024", "GC001", "GC007", "GC011", "GC012", "GC005", "GC027", "GC028", "GC014", "GC023", "GC025"]
ACTIONS = {
    "fwd8": ["WAIT"] + ["W"] * 8,
    "bwd8": ["WAIT"] + ["S"] * 8,
    "left8": ["WAIT"] + ["A"] * 8,
    "right8": ["WAIT"] + ["D"] * 8,
    "turn180": ["WAIT"] + ["RIGHT"] * 3,
    "round_trip": PATHS["round_trip"],
}
STRAFE_SCENES = {"GC017", "GC015", "GC024", "GC001", "GC007", "GC011"}
ROUND_TRIP_SCENES = {"GC017", "GC015", "GC001", "GC007", "GC024"}


def job(scene, name):
    return {"id": f"nf-{scene}-{name}", "task_id": scene, "split": "gc", "seed": 42,
            "actions": ACTIONS[name], "method": "stock"}


def main(out):
    jobs = []
    for scene in SCENES:                                  # tier 1: forward and backward everywhere
        jobs += [job(scene, "fwd8"), job(scene, "bwd8")]
    jobs += [job(scene, "turn180") for scene in SCENES]   # tier 2: rotation baseline
    for scene in SCENES:                                  # tier 3: strafes and round trips on a subset
        if scene in STRAFE_SCENES:
            jobs += [job(scene, "left8"), job(scene, "right8")]
        if scene in ROUND_TRIP_SCENES:
            jobs.append(job(scene, "round_trip"))
    Path(out).write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(len(jobs), "jobs ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reproduction/near-field-jobs.jsonl")
