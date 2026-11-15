"""Actual cross-agent visual KV revisit. One static scene, common camera origin.

The B scout populates a right-hand view. A explores left then looks right and
reads B's KV. B then returns left and reads A's KV. All visits share one archive;
each visit starts a fresh live cache. This is serial exploration, not a live sim.
"""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if modal.is_local():
    from scripts.modal_lingbot_benchmark import image, weights, results
else:
    image = None
    weights = modal.Volume.from_name("worldsync-lingbot-weights")
    results = modal.Volume.from_name("worldsync-lingbot-results")
app = modal.App("worldsync-shared-kv-revisit")


@app.function(image=image, gpu="H200", cpu=8, memory=131072, timeout=1200,
              max_containers=2, volumes={"/cache": weights, "/results": results})
def generate(task_id: str = "GC007"):
    import os
    import shutil
    import time
    from worldsync.lingbot_runtime import WarmRuntime
    from worldsync.lingbot_rollout import main as rollout
    from worldsync.lingbot_shared_kv import SharedPoseArchive, SharedPoseKVMemory
    os.environ["WORLDSYNC_LINGBOT_ATTENTION"] = "sdpa"
    weights.reload()
    raw = Path("/cache/14b-causal-fast")
    manifest = json.loads(Path("/work/reproduction/lingbot-weights.json").read_text())
    for row in manifest["files"]:
        if (raw/row["path"]).stat().st_size != row["bytes"]:
            raise ValueError("checkpoint inventory mismatch")
    target = Path("/work/external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(raw)
    identifier = f"shared-kv-revisit-{task_id}-{time.time_ns()}"
    root = Path("/work/outputs")/identifier
    runtime, archive, output = WarmRuntime(), SharedPoseArchive(frames_per_actor=8), {}
    jobs = [("B-scout", "B", 42, ["WAIT", "RIGHT", "WAIT"]),
            ("A-visit", "A", 43, ["WAIT", "LEFT", "WAIT", "RIGHT", "RIGHT", "WAIT"]),
            ("B-visit", "B", 44, ["WAIT", "RIGHT", "WAIT", "LEFT", "LEFT", "WAIT"])]
    for name, actor, seed, actions in jobs:
        dest = root/name
        print(json.dumps({"event": "visit_start", "pass": name, "actor": actor,
                          "archive_frames": len(archive.frames)}), flush=True)
        try:
            run = rollout(["--split", "gc", "--task-id", task_id, "--seed", str(seed),
                           "--max-chunks", str(len(actions)), "--native-fixed-controls", "--fixed-actions", *actions,
                           "--offload-text-encoder", "--output", str(dest)], runtime=runtime,
                           memory_factory=lambda pipeline: SharedPoseKVMemory(pipeline, archive, actor, name))
            if run["status"] != "completed" or len(run["chunks"]) != len(actions):
                raise ValueError(f"incomplete visit: {name}")
            output[name] = {"run": run, "video": (dest/"rollout.mp4").read_bytes()}
        finally:
            if dest.exists():
                shutil.copytree(dest, Path("/results")/identifier/name)
                results.commit()
            runtime.between_jobs()
    proof = {}
    for name, peer, heading in [("A-visit", "B", 60), ("B-visit", "A", -60)]:
        run = output[name]["run"]
        rows = [r for r in run["memory"]["trace"] if r["attention_replaced"]
                and len(r["layers_injected"]) == 40 and r["chunk"] >= 4
                and r["peer_frames"] and all(f["actor"] == peer and f["pose_gap"][1] <= 20.001
                                             for f in r["peer_frames"])]
        if not rows:
            raise ValueError(f"no verified 40-layer peer KV injection on revisit: {name}")
        proof[name] = {"donor_agent": peer, "revisit_heading_degrees": heading, "chunks": rows,
                       "layers": 40, "injections_by_layer": run["memory"]["injections_by_layer"]}
    summary = {"directory": identifier, "task_id": task_id, "scope": "static common-origin rotation-only cross-agent revisit",
               "peer_visual_kv_injected_into_both": True, "proof": proof, "archive": archive.metadata(),
               "training": False, "generation_order": [j[0] for j in jobs],
               "limitation": "serial visits; injection verified, visual consistency must be reviewed"}
    (Path("/results")/identifier/"shared-kv-proof.json").write_text(json.dumps(summary, indent=2)+"\n")
    results.commit()
    return {"summary": summary, "visits": output}


@app.local_entrypoint()
def main(more: bool = False):
    from worldsync.pov_video import tile
    tasks = ("GC011", "GC026") if more else ("GC007",)
    calls = [(task, generate.spawn(task)) for task in tasks]
    for task, call in calls:
        result = call.get()
        root = ROOT/(f"outputs/shared-kv-revisit-{task}" if more else "outputs/shared-kv-revisit")
        root.mkdir(parents=True, exist_ok=True)
        for name, visit in result["visits"].items():
            (root/f"{name}.mp4").write_bytes(visit["video"])
            (root/f"{name}.json").write_text(json.dumps(visit["run"], indent=2)+"\n")
        (root/"shared-kv-proof.json").write_text(json.dumps(result["summary"], indent=2)+"\n")
        tile(root/"A-visit.mp4", root/"B-visit.mp4", root/"tiled.mp4",
             labels=("AGENT A - LOOKS WHERE B LOOKED", "AGENT B - LOOKS WHERE A LOOKED"))
        print(json.dumps({"task": task, "video": str(root/"tiled.mp4"), "proof": str(root/"shared-kv-proof.json")}))
