"""Two short coordinated POV experiments: reciprocal wave and joint attention."""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_two_agent_pov import image, weights, results
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")
app = modal.App("worldsync-extra-pov-demos")


def schedule(actor, scenario):
    from worldsync.pov_video import viewpoint_text, STATES
    base = viewpoint_text(actor, STATES[0])
    if scenario == "wave":
        actions = {
            "orange": ["The viewer lifts their right orange-sleeved hand and waves gently to the purple-clothed companion, keeping the basket steady in the left hand. The companion smiles.",
                       "The purple-clothed companion raises her right hand and waves back warmly. The viewer lowers their own hand and keeps holding the same basket."],
            "purple": ["The orange-clothed companion raises her right hand and waves to the viewer, keeping the basket steady in her left hand. The viewer smiles.",
                       "The viewer raises their own right purple-sleeved hand and waves back to the orange-clothed companion. She lowers her own hand and keeps the basket in her lap."]}
    else:
        actions = {
            "orange": ["The viewer points their right orange-sleeved index finger toward a duck swimming on the river on the right, holding the basket steady with the left hand. The purple-clothed companion watches the gesture.",
                       "The purple-clothed companion turns her head toward the indicated duck on the river. The viewer keeps pointing toward it. Both remain seated and the basket stays with the viewer."],
            "purple": ["The orange-clothed companion extends her right index finger toward a duck swimming on the river on the left, keeping the basket steady with her left hand. The viewer watches her pointing gesture.",
                       "The viewer turns slightly left toward the indicated duck swimming on the river. The orange-clothed companion continues pointing toward it and keeps the basket in her lap."]}
    return {"cases": {"insight:POV001": {"static": base+" "+actions[actor][0],
            "scheduled": [[0, base+" "+actions[actor][0]], [3, base+" "+actions[actor][1]]]}}}


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=900,
              max_containers=1, volumes={"/cache": weights, "/results": results})
def generate():
    import os
    import shutil
    import time
    from worldsync.lingbot_runtime import WarmRuntime
    from worldsync.lingbot_rollout import main as rollout
    os.environ["WORLDSYNC_LINGBOT_ATTENTION"] = "sdpa"
    weights.reload()
    raw = Path("/cache/14b-causal-fast")
    manifest = json.loads(Path("/work/reproduction/lingbot-weights.json").read_text())
    for row in manifest["files"]:
        if (raw / row["path"]).stat().st_size != row["bytes"]:
            raise ValueError("checkpoint inventory mismatch")
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(raw)
    identifier = f"extra-pov-{time.time_ns()}"
    runtime, output = WarmRuntime(), {}
    for scenario in ("wave", "look_at_river"):
        for actor, seed in zip(("orange", "purple"), (44, 45)):
            dest = Path("/work/outputs") / identifier / scenario / actor
            director = Path(f"/work/{scenario}-{actor}.json")
            director.write_text(json.dumps(schedule(actor, scenario)))
            try:
                run = rollout(["--data-root", f"/work/pov-inputs/{actor}", "--split", "insight", "--task-id", "POV001",
                               "--max-chunks", "6", "--seed", str(seed), "--fixed-actions", *(["WAIT"]*6),
                               "--director", "perchunk", "--director-text-file", str(director),
                               "--offload-text-encoder", "--output", str(dest)], runtime=runtime)
                if run["status"] != "completed" or len(run["chunks"]) != 6:
                    raise ValueError("incomplete render")
                output[f"{scenario}/{actor}"] = {"video": (dest/"rollout.mp4").read_bytes(), "run": run}
            finally:
                if dest.exists():
                    shutil.copytree(dest, Path("/results") / identifier / scenario / actor)
                    results.commit()
                runtime.between_jobs()
    return {"directory": identifier, "views": output}


@app.local_entrypoint()
def main():
    from worldsync.pov_video import tile
    result = generate.remote()
    root = ROOT/"outputs/extra-agent-demos"
    root.mkdir(parents=True, exist_ok=True)
    for key, outcome in result.pop("views").items():
        path = root/key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".mp4").write_bytes(outcome["video"])
        path.with_suffix(".json").write_text(json.dumps(outcome["run"], indent=2)+"\n")
    for scenario in ("wave", "look_at_river"):
        tile(root/scenario/"orange.mp4", root/scenario/"purple.mp4", root/scenario/"tiled.mp4")
    (root/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
