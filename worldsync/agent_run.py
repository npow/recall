"""Adaptive OpenAI player + ABot. Controller differs from the published PlayWorld setup."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageOps

from worldsync.player import KEY_MAP, OpenAIPlayer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("external/ABot-World"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=["stock", "window", "landmark"], default=["stock", "window", "landmark"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--player-model", help="default: gpt-6.1-sol (OpenAI), claude-opus-5-5 (Anthropic)")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-blocks", type=int, default=80, help="80 blocks is approximately 60 simulated seconds")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 40 or not 1 <= args.max_blocks <= 80:
        parser.error("max-steps must be 1..40 and max-blocks 1..80")
    repo, output = args.repo.resolve(), args.output.resolve()
    records = json.loads(args.dataset.read_text())
    task = next((r for r in records if r["task_id"] == args.task_id), None)
    if task is None:
        parser.error("task ID not found")
    image = (args.data_root / task["image_path"]).resolve()
    if not image.is_file() or not task.get("image_caption"):
        parser.error("task requires an existing image and image_caption")
    if "from worldsync.memory import memory_view" not in (repo / "wan/modules/causal_model.py").read_text():
        parser.error("apply ABot memory patch first")
    args.player_model = args.player_model or ("gpt-6.1-sol" if args.provider == "openai" else "claude-opus-5-5")
    if args.provider == "anthropic":
        from worldsync.player import AnthropicPlayer
        player = AnthropicPlayer(model=args.player_model)
    else:
        player = OpenAIPlayer(model=args.player_model)

    import imageio.v2 as imageio
    import torch
    from worldsync.session import ABotSession
    session = ABotSession(repo, args.height, args.width)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    for mode in args.modes:
        directory = output / f"seed-{args.seed}" / mode / args.task_id
        directory.mkdir(parents=True, exist_ok=False)
        initial_path = directory / "before_actions.jpg"
        current_path = initial_path
        ImageOps.fit(Image.open(image).convert("RGB"), (args.width, args.height), method=Image.Resampling.LANCZOS).save(initial_path)
        session.reset(image=image, caption=task["image_caption"], seed=args.seed, mode=mode, ref_dir=directory / "unused_refs")
        torch.cuda.reset_peak_memory_stats()
        result = {"status": "running", "task_id": args.task_id, "mode": mode,
                  "seed": args.seed, "player_model": args.player_model, "player_provider": args.provider,
                  "player_effort": "low" if args.provider == "openai" else "medium",
                  "objective": task["prompt"], "upstream_commit": commit,
                  "initial_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                  "controller_protocol": "custom objective-driven controller; not the published Claude harness",
                  "height": args.height, "width": args.width, "fps": 16,
                  "max_steps": args.max_steps, "max_blocks": args.max_blocks,
                  "decisions": [], "blocks": [], "agent_finished": False}
        result_path = directory / "result.json"
        result_path.write_text(json.dumps(result, indent=2))
        history = []
        start = time.perf_counter()
        try:
            with imageio.get_writer(str(directory / "rollout.mp4"), fps=16, codec="libx264", macro_block_size=1) as writer:
                for step in range(args.max_steps):
                    decision, metadata = player.decide(initial_path, current_path, task["prompt"], history, args.max_steps - step)
                    result["decisions"].append({**decision.model_dump(), **metadata})
                    print(f"{mode} {args.task_id} decision={step + 1} action={decision.action} blocks={decision.blocks}", flush=True)
                    if decision.action == "FINISH":
                        result["agent_finished"] = True
                        break
                    executed = 0
                    for _ in range(min(decision.blocks, args.max_blocks - len(result["blocks"]))):
                        tick = time.perf_counter()
                        keys = [] if decision.action == "WAIT" else [KEY_MAP[decision.action]]
                        frames = session.step(keys)
                        for frame in frames:
                            writer.append_data(frame)
                        current_path = directory / f"observation-{len(result['blocks']):03d}.jpg"
                        Image.fromarray(frames[-1]).save(current_path)
                        Image.fromarray(frames[-1]).save(directory / "after_actions.jpg")
                        result["blocks"].append({"action": decision.action, "native_keys": keys,
                                                 "decoded_frames": len(frames), "seconds": time.perf_counter() - tick})
                        executed += 1
                    history.append({"action": decision.action, "executed_blocks": executed})
                    result_path.write_text(json.dumps(result, indent=2))
                    if len(result["blocks"]) >= args.max_blocks:
                        break
            result["status"] = "completed"
            result["completion_reason"] = "agent_finish" if result["agent_finished"] else "budget_exhausted"
        except Exception as exc:
            result["status"] = "failed"
            result["error_type"] = type(exc).__name__
            raise
        finally:
            result["elapsed_seconds"] = time.perf_counter() - start
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
