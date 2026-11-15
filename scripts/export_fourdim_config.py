"""Write reproduction/four-dimension-config.json: exact flags, case selection and finished pilot episodes.

  python scripts/export_fourdim_config.py SEED_DIR OUTPUT.json
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from worldsync import fourdim  # noqa: E402
import modal_lingbot_fourdim as runner  # noqa: E402


def main():
    seed_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
    generated = json.loads((seed_dir / "generated-at-stop.json").read_text())
    cases = fourdim.select_cases(ROOT / "external/PlayWorld/data", 12)
    episodes_dir = seed_dir / "episodes"
    finished = []
    for g in sorted(generated, key=lambda g: g.get("identifier", "")):
        if g.get("returncode") != 0 or not g.get("identifier"):
            continue
        directory = episodes_dir / g["identifier"]
        finished.append({"directory": g["identifier"], "arm": g["arm"], "split": g["split"], "task": g["task"], "seed": g["seed"],
                         "has_rollout_mp4": (directory / "rollout.mp4").is_file(),
                         "has_opus_judge_json": (directory / "judge.json").is_file()})
    config = {
        "protocol": "worldsync four-dimension stock vs budget-memory pilot, seed 42",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "modal": {"profile": "nissan-pow", "app": "worldsync-lingbot-fourdim (stopped)", "gpu": "H200",
                  "results_volume": "worldsync-lingbot-results",
                  "volume_layout": "<directory>/{rollout.mp4,run.json,judge.json,controls/}; chunk-*.mp4 and observation-*.jpg are NOT copied",
                  "fetch": "modal volume get worldsync-lingbot-results <directory>/rollout.mp4 <local_dir>/"},
        "rollout_command": "python -m worldsync.lingbot_rollout --split SPLIT --task-id TASK --seed 42 --max-chunks N "
                           "COMMON ARM_FLAGS --output DIR",
        "common_flags": runner.COMMON, "arm_flags": runner.ARMS,
        "max_chunks_by_split": runner.MAX_CHUNKS,
        "player": "worldsync.reference_player.decide (Haiku 4.5, Player v6 with END challenge); insight cases: stationary WAIT, no player",
        "case_selection": {"salt": fourdim.SALT, "per_dimension": 12, "excluded": {f"{s}/{t}": why for (s, t), why in fourdim.EXCLUDED.items()},
                           "rule": "sort task ids per split by sha256(f'{salt}:{split}:{task}'), take first 12",
                           "cases": [{"split": s, "task": t} for s, t in cases]},
        "judges": {"primary": "released PlayWorld Gemini rubric via worldsync/gemini_vertex_judge.py (gemini-3.1-pro-preview, Vertex global, gcloud login)",
                   "secondary": "worldsync.judge (Opus) inside Modal"},
        "finished_pilot_episodes": finished,
        "not_generated_or_cancelled": "every case/arm not listed in finished_pilot_episodes (app stopped at the coordinator's request)",
    }
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(len(finished), "finished episodes;", len(cases), "cases")


if __name__ == "__main__":
    main()
