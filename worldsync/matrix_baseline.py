"""Stock Matrix-Game 3.0 with recorded headless I/O and a reconstructed player.

No generator or memory patch is applied. Missing paper-specific adapter settings
are explicit protocol deviations; this runner cannot unlock reproduction.
"""
import argparse
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from worldsync.credentials import load_local_credentials
from worldsync.judge import parse_json

SOURCE_REVISION = "51913b2bfa78a6d54de912b8348b1d318f4d36e2"
FPS = 17
KEYBOARD = {"W": [1, 0, 0, 0, 0, 0], "S": [0, 1, 0, 0, 0, 0],
            "A": [0, 0, 1, 0, 0, 0], "D": [0, 0, 0, 1, 0, 0]}
MOUSE = {"UP": [0.1, 0], "DOWN": [-0.1, 0], "LEFT": [0, -0.1], "RIGHT": [0, 0.1]}
OPERATIONS = {"KEEP", "STOP", "EXTEND", "CORRECT", "END"}
PROTOCOL_ID = "matrix-stock-reconstructed-haiku-17fps-v1"


class RolloutComplete(Exception):
    pass


def native_action(action):
    """Match upstream get_current_action exactly; invalid controls fail closed."""
    if action not in {*KEYBOARD, *MOUSE, "WAIT"}:
        raise ValueError(f"unsupported Matrix action: {action}")
    return {"keyboard": KEYBOARD.get(action, [0, 0, 0, 0, 0, 0]),
            "mouse": MOUSE.get(action, [0, 0])}


def reference_plan(task):
    """Retain the human plan verbatim; do not invent token-to-chunk equivalence."""
    return task.get("action_sequence_steps") or task.get("action_sequence") or task.get("action")


def player_prompt(task, history, maximum):
    return (
        "You are the Agent Player controlling an interactive world model. Pursue the goal using "
        "the shared human reference plan as guidance. Adapt to actual generated observations. "
        "The first image is the initial world; the second is the latest generated frame. "
        "KEEP continues the current reference action; STOP stops its execution and selects the "
        "next appropriate action; EXTEND repeats an action when more progress is needed; "
        "CORRECT selects a corrective action; END terminates when the objective is achieved. "
        "For all operations except END, explicitly select exactly one action: W, S, A, D, "
        "LEFT, RIGHT, UP, DOWN, WAIT. W/S move forward/back; A/D move sideways; arrows turn "
        "the camera; WAIT leaves the controls neutral while time passes. "
        "The next native chunk lasts 57 frames initially, then 40 frames at 17 FPS. "
        "You may end early. Never invent task completion because the scene changed identity. "
        "Return JSON only: {\"operation\":\"KEEP|STOP|EXTEND|CORRECT|END\", "
        "\"action\":\"one allowed action or null for END\", \"reason\":\"visual evidence\"}.\n"
        + json.dumps({"goal": task["prompt"], "reference_plan": reference_plan(task),
                      "executed_history": history, "max_interactions": maximum}, ensure_ascii=False)
    )


def decide(client, model, task, initial, latest, history, maximum):
    from PIL import Image
    content = [{"type": "text", "text": player_prompt(task, history, maximum)}]
    for path in (initial, latest):
        image = Image.open(path).convert("RGB")
        image.thumbnail((1024, 1024))
        data = BytesIO()
        image.save(data, format="JPEG", quality=85)
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                      "data": base64.b64encode(data.getvalue()).decode()}})
    response = client.messages.create(model=model, max_tokens=1500,
        messages=[{"role": "user", "content": content}])
    raw = "\n".join(block.text for block in response.content if block.type == "text")
    value = parse_json(raw)
    if value.get("operation") not in OPERATIONS or not isinstance(value.get("reason"), str):
        raise ValueError("invalid player decision")
    if value["operation"] != "END":
        native_action(value.get("action"))
    return value, {"response_id": response.id, "model": response.model,
                   "usage": response.usage.model_dump(), "raw_response": raw,
                   "prompt_sha256": hashlib.sha256(content[0]["text"].encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("external/Matrix-Game/Matrix-Game-3"))
    parser.add_argument("--checkpoint", type=Path, default=Path("external/Matrix-Game/checkpoints/Matrix-Game-3.0"))
    parser.add_argument("--data-root", type=Path, default=Path("external/PlayWorld/data"))
    parser.add_argument("--split", choices=["gc", "if", "insight", "outsight"], required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-interactions", type=int, default=40)
    parser.add_argument("--max-chunks", type=int, default=25, help="25 native chunks = 59.82 seconds; a reconstruction choice")
    parser.add_argument("--player-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--fixed-actions", nargs="+", help="Explicit transport smoke test only; not adaptive evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_interactions <= 40 or not 1 <= args.max_chunks <= 40:
        parser.error("chunk and interaction limits must be in [1,40]")
    source, checkpoint, data_root, output = [p.resolve() for p in (args.source, args.checkpoint, args.data_root, args.output)]
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; preserve the existing run and choose a new path")
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != SOURCE_REVISION:
        parser.error("Matrix source revision differs from the frozen reference")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "Matrix-Game-3"], cwd=source.parent, check=True, stdout=subprocess.DEVNULL)
    records = json.loads((data_root / args.split / "data.json").read_text())
    task = next((r for r in records if r["task_id"] == args.task_id), None)
    if task is None:
        parser.error("task not in selected split")
    initial = data_root / task["image_path"]
    native_args = SimpleNamespace(size="704*1280", output_dir=str(output), save_name="rollout",
        ckpt_dir=str(checkpoint), num_iterations=min(args.max_chunks, args.max_interactions),
        use_int8=True, verify_quant=False, vae_type="mg_lightvae", lightvae_pruning_rate=0.5,
        use_async_vae=False, compile_vae=False, fa_version="0")
    report = {"status": "prepared", "protocol_id": PROTOCOL_ID, "model": "Matrix-Game 3.0 stock",
        "source_revision": revision, "task_id": args.task_id, "split": args.split, "seed": args.seed,
        "task": task, "initial_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
        "fps": FPS, "native_args": vars(native_args), "sampling_steps": 3, "sample_shift": 5,
        "player_model": args.player_model, "max_interactions": args.max_interactions,
        "dry_run": args.dry_run, "fixed_actions": args.fixed_actions, "actions": [], "chunks": [],
        "published_score_reproduced": False,
        "deviations": ["Matrix paper adapter unpublished: reconstructed reference-guided player prompt and stopping",
            "Paper checkpoint revision and decoder/precision flags not documented: use pinned release and demo int8/LightVAE settings",
            "Single Thor GPU, PyTorch 2.15 CUDA 13.2, SDPA instead of demo multi-GPU FlashAttention3",
            "25-chunk cap chosen to approach 60 seconds; exact paper model-specific chunk budget unknown",
            "No Gemini judge calls; numerical agreement with published score remains unverified"]}
    report_path = output / "run.json"
    def save():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)
    save()
    if args.dry_run:
        print(json.dumps({"status": "prepared", "protocol_id": PROTOCOL_ID}))
        return
    if args.fixed_actions:
        for action in args.fixed_actions:
            native_action(action)
    else:
        from anthropic import Anthropic
        load_local_credentials()
        client = Anthropic(timeout=180, max_retries=2, default_headers={"Accept-Encoding": "identity"})
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    sys.path.insert(0, str(source))
    from pipeline import inference_interactive_pipeline as upstream
    from wan.configs import WAN_CONFIGS
    from utils.misc import set_seed
    set_seed(args.seed)
    latest = initial
    started = time.monotonic()

    def next_action():
        if len(report["actions"]) >= native_args.num_iterations:
            report["termination"] = "budget_exhausted"
            raise RolloutComplete
        if args.fixed_actions:
            index = len(report["actions"])
            if index >= len(args.fixed_actions):
                report["termination"] = "fixed_smoke_actions_exhausted"
                raise RolloutComplete
            decision, metadata = {"operation": "KEEP", "action": args.fixed_actions[index], "reason": "fixed transport smoke test"}, {}
        else:
            decision, metadata = decide(client, args.player_model, task, initial, latest,
                [{k: action[k] for k in ("operation", "action", "reason")} for action in report["actions"]], args.max_interactions)
        report["actions"].append({**decision, **metadata, "elapsed_s": time.monotonic() - started})
        save()
        if decision["operation"] == "END":
            report["termination"] = "agent_end"
            raise RolloutComplete
        return {key: torch.tensor(value) for key, value in native_action(decision["action"]).items()}

    def observe(video, filename, config, *unused, **kwargs):
        nonlocal latest
        if "_current_iteration_" not in filename:
            return
        index = len(report["chunks"])
        path = output / f"chunk-{index:03d}.mp4"
        # Receive pixels before upstream's key/mouse overlays. Preserve all frames.
        imageio.mimwrite(path, video, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
        latest = output / f"observation-{index:03d}.jpg"
        Image.fromarray(video[-1]).save(latest, quality=95)
        record = {"index": index, "path": path.name, "frames": len(video), "elapsed_s": time.monotonic() - started,
            "keyboard": config[0][0].tolist(), "mouse": config[1][0].tolist(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        report["chunks"].append(record)
        report["generated_frames"] = sum(chunk["frames"] for chunk in report["chunks"])
        report["generated_seconds"] = report["generated_frames"] / FPS
        save()
        print(json.dumps({"event": "chunk_saved", **record}), flush=True)
        if len(report["chunks"]) >= native_args.num_iterations:
            report["termination"] = "chunk_budget_exhausted"
            raise RolloutComplete

    previous_action, previous_video = upstream.get_current_action, upstream.process_video
    previous_cwd = Path.cwd()
    upstream.get_current_action, upstream.process_video = next_action, observe
    report["status"] = "running"
    report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0)}
    save()
    try:
        os.chdir(source)
        pipeline = upstream.MatrixGame3Pipeline(config=WAN_CONFIGS["matrix_game3"],
            checkpoint_dir=str(checkpoint), args=native_args, fa_version="0")
        pipeline.generate(task["image_caption"], Image.open(initial).convert("RGB"),
            max_area=704 * 1280, shift=5, num_inference_steps=3, guide_scale=5,
            seed=args.seed, use_base_model=False, args=native_args)
    except RolloutComplete:
        report["status"] = "completed" if report["chunks"] else "empty_rollout"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        os.chdir(previous_cwd)
        upstream.get_current_action, upstream.process_video = previous_action, previous_video
        report["elapsed_s"] = time.monotonic() - started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        save()
        if report["chunks"]:
            listing = output / "chunks.txt"
            listing.write_text("".join(f"file '{chunk['path']}'\n" for chunk in report["chunks"]))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", str(output / "rollout.mp4")], check=True)
    print(json.dumps({"status": report["status"], "chunks": len(report["chunks"]), "output": str(output)}))


if __name__ == "__main__":
    main()
