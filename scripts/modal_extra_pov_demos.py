"""Two short coordinated POV experiments: reciprocal wave and joint attention."""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_two_agent_pov import image, weights, results
    if (ROOT / "outputs/more-agent-pov/inputs").is_dir():
        image = image.add_local_dir(ROOT / "outputs/more-agent-pov/inputs", "/work/more-pov-inputs")
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")
app = modal.App("worldsync-extra-pov-demos")


def schedule(actor, scenario, empty_hands=False):
    from worldsync.pov_video import viewpoint_text, STATES
    base = viewpoint_text(actor, STATES[0])
    if empty_hands:
        companion = "purple floral kimono with a pale green obi and pink hair flower" if actor == "orange" else "orange floral kimono with a silver obi and pale green hair flower"
        side = "right" if actor == "orange" else "left"
        base = (f"A continuous seated first-person view beside the same sunny Japanese river, on the {side}. "
                f"Only the viewer's own {actor} floral sleeves and hands are visible below. "
                f"The same companion in a {companion} sits opposite. Both have empty hands. ")
        if scenario == "wave":
            actions = {
                "orange": ["The viewer raises the right orange-sleeved hand beside their face and waves with an open palm. The purple-clothed companion watches, smiling, her hands on her knees.", "The viewer lowers the right hand onto their knee. The purple-clothed companion raises her right hand beside her face and waves back with an open palm."],
                "purple": ["The orange-clothed companion raises her right hand beside her face and waves with an open palm. The viewer keeps both purple-sleeved hands on their knees.", "The orange-clothed companion lowers her right hand onto her knee. The viewer raises the right purple-sleeved hand beside their face and waves back with an open palm."]}
        elif scenario == "high_five":
            actions = {
                "orange": ["The viewer extends the right orange-sleeved arm, holding an open palm upright between the two people. The purple-clothed companion lifts her right palm toward it.", "The purple-clothed companion is pressing her right palm flat against the viewer's upright right palm in a gentle high five. Both smile, then lower their hands."],
                "purple": ["The orange-clothed companion extends her right arm, holding an open palm upright between the two people. The viewer lifts their own right purple-sleeved palm toward it.", "The viewer is pressing their right palm flat against the orange-clothed companion's upright right palm in a gentle high five. Both smile, then lower their hands."]}
        else:
            actions = {
                "orange": ["The viewer extends their right orange-sleeved index finger toward the river on the right, pointing steadily at one swimming duck. The purple-clothed companion watches the gesture.", "The purple-clothed companion turns her head toward the river on the right to follow the indicated duck. The viewer continues pointing toward it."],
                "purple": ["The orange-clothed companion extends her right index finger toward the river on the left, pointing steadily at one swimming duck. The viewer's own hands remain on their knees.", "The viewer turns slightly left toward the indicated duck on the river, while the orange-clothed companion continues pointing toward it."]}
        return {"cases": {"insight:POV001": {"static": base+actions[actor][0],
                "scheduled": [[0, base+actions[actor][0]], [3, base+actions[actor][1]]]}}}
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
              max_containers=3, volumes={"/cache": weights, "/results": results})
def generate(scenarios: list[str] | None = None, seed_offset: int = 0, empty_hands: bool = False):
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
    for scenario in (scenarios or ["wave", "look_at_river"]):
        for actor, seed in zip(("orange", "purple"), (44+seed_offset, 45+seed_offset)):
            dest = Path("/work/outputs") / identifier / scenario / actor
            director = Path(f"/work/{scenario}-{actor}.json")
            director.write_text(json.dumps(schedule(actor, scenario, empty_hands)))
            try:
                inputs = "/work/more-pov-inputs" if empty_hands else "/work/pov-inputs"
                run = rollout(["--data-root", f"{inputs}/{actor}", "--split", "insight", "--task-id", "POV001",
                               "--max-chunks", "6", "--seed", str(seed), "--fixed-actions", *(["WAIT"]*6),
                               "--director", "perchunk", "--director-text-file", str(director),
                               "--offload-text-encoder", "--output", str(dest)], runtime=runtime)
                if run["status"] != "completed" or len(run["chunks"]) != 6:
                    raise ValueError("incomplete render")
                output[f"{scenario}/{actor}"] = {"video": (dest/"rollout.mp4").read_bytes(), "run": run}
            finally:
                if dest.exists():
                    shutil.copytree(dest, Path("/results") / identifier / scenario / actor)
                    (Path("/results") / identifier / scenario / actor / "director-input.json").write_text(director.read_text())
                    results.commit()
                runtime.between_jobs()
    return {"directory": identifier, "views": output}


@app.local_entrypoint()
def main(parallel: bool = False):
    from worldsync.pov_video import tile
    scenarios = ("wave", "high_five", "look_at_river") if parallel else ("wave", "look_at_river")
    if parallel:
        calls = [generate.spawn([name], 10*(i+1), True) for i, name in enumerate(scenarios)]
        results_by_worker = [call.get() for call in calls]
        result = {"directories": [r["directory"] for r in results_by_worker],
                  "views": {key: value for r in results_by_worker for key,value in r["views"].items()}}
    else:
        result = generate.remote()
    root = ROOT/("outputs/more-agent-demos" if parallel else "outputs/extra-agent-demos")
    root.mkdir(parents=True, exist_ok=True)
    for key, outcome in result.pop("views").items():
        path = root/key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".mp4").write_bytes(outcome["video"])
        path.with_suffix(".json").write_text(json.dumps(outcome["run"], indent=2)+"\n")
    for scenario in scenarios:
        tile(root/scenario/"orange.mp4", root/scenario/"purple.mp4", root/scenario/"tiled.mp4")
    (root/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
