"""Claude judging with PlayWorld's prompts and sampling. Not its published Gemini protocol."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from worldsync.credentials import load_local_credentials

REVISION = "eeaea12e050a90b815d11d030e943e7b3181da21"


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("judge must return a JSON object")
    return value


def evidence_content(prompt, artifacts, reference_image=None):
    content = [{"type": "text", "text": prompt}]

    def add_image(path):
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                       "data": base64.b64encode(path.read_bytes()).decode()}})

    if reference_image:
        content.append({"type": "text", "text": "REFERENCE IMAGE: initial scene/style context only; do not score it as a generated frame."})
        from PIL import Image
        from io import BytesIO
        buffer = BytesIO()
        Image.open(reference_image).convert("RGB").save(buffer, format="JPEG", quality=95)
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(buffer.getvalue()).decode()}})
    for stream in (artifacts.primary, artifacts.detail):
        spec = stream.spec
        content.append({"type": "text", "text": f"BEGIN {spec.name.upper()} EVIDENCE: {spec.fps:g} FPS, {spec.cell_width}x{spec.cell_height} cells in chronological {spec.grid_columns}x{spec.grid_rows} contact sheets."})
        for index, sheet in enumerate(stream.sheets, 1):
            first = (index - 1) * spec.frames_per_sheet
            last = min(first + spec.frames_per_sheet, len(stream.frames)) - 1
            content.append({"type": "text", "text": f"{spec.name.upper()} SHEET {index}/{len(stream.sheets)}; sample indices {first}-{last}; continue chronologically."})
            add_image(sheet)
        content.append({"type": "text", "text": f"END {spec.name.upper()} EVIDENCE."})
    return content


def effective_score(gate, rubric):
    if gate is not None and gate.get("verdict") == "fail":
        return 1.0
    if rubric is None:
        return None
    value = rubric.get("final_score_1_to_5")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 5:
        raise ValueError("judge score must be a number in [1, 5]")
    return float(value)


def metric_task_record(task):
    # The dataset uses spaces; the released gate dispatcher expects hyphens.
    # Keep every original field/rubric otherwise intact and record this bridge.
    return {**task, "task_id": task.get("source_task_id") or task["task_id"],
            "perspective": str(task.get("perspective") or "").strip().lower().replace(" ", "-")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playworld", type=Path, default=Path("external/PlayWorld"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--world-model", default="ABot-World-0")
    parser.add_argument("--mode", choices=["both", "gate", "score"], default="both")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = args.playworld.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected pinned PlayWorld revision {REVISION}")
    sys.path.insert(0, str(repo))
    from metrics.vqa.sampling import generate_dual_sampling, export_artifacts
    from metrics.vqa.rubric import build_scoring_context, build_scoring_prompt
    from metrics.vqa.instruction_gate import build_gate_context, build_gate_prompt, gate_kind
    task = next((r for r in json.loads(args.dataset.read_text()) if r["task_id"] == args.task_id), None)
    if task is None:
        parser.error("task not found")
    metric_task = metric_task_record(task)
    item = {"model": args.world_model, "task": metric_task["task_id"],
            "group": task.get("source_category") or task.get("category"),
            "view": metric_task.get("perspective"), "caption": task.get("image_caption"), "status": "generated"}
    result = {"task_id": args.task_id, "judge_provider": "anthropic", "judge_model": args.model,
              "playworld_revision": revision, "dry_run": args.dry_run, "mode": args.mode,
              "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
              "protocol_warning": "Claude judge and custom controller differ from the published PlayWorld protocol; do not compare directly with its leaderboard.",
              "gate": None, "rubric": None, "reported_score": None, "calls": []}
    result["input_normalization"] = {"original_perspective": task.get("perspective"), "metric_perspective": metric_task["perspective"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="worldsync-judge-") as directory:
        artifacts = generate_dual_sampling(args.video, Path(directory))
        result["sampling"] = artifacts.metadata()
        if args.sampling_output_dir:
            export_artifacts(artifacts, args.sampling_output_dir)
        if not args.dry_run:
            from anthropic import Anthropic
            load_local_credentials()
            client = Anthropic(timeout=180, max_retries=2, default_headers={"Accept-Encoding": "identity"})
        for mode in (["gate", "score"] if args.mode == "both" else [args.mode]):
            if mode == "gate" and gate_kind(metric_task) is None:
                result["gate"] = {"verdict": "not_applicable", "instruction_following_score": None}
                continue
            if mode == "score" and result["gate"] and result["gate"]["verdict"] == "fail":
                result["rubric_skipped"] = "trajectory failed; effective score is 1"
                break
            if mode == "gate":
                context = build_gate_context(item, metric_task, result["sampling"])
                prompt = build_gate_prompt(item, metric_task, result["sampling"])
            else:
                context = build_scoring_context(item, metric_task, len(artifacts.primary.frames), fps=10, sampling_metadata=result["sampling"])
                prompt = build_scoring_prompt(item, metric_task, len(artifacts.primary.frames), fps=10, sampling_metadata=result["sampling"])
            result[f"{mode}_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            context_path = args.output.with_name(args.output.stem + f"-{mode}-context.json")
            context_path.write_text(json.dumps(context, indent=2))
            if args.dry_run:
                continue
            response = client.messages.create(
                model=args.model, max_tokens=16384,
                system="Evaluate the supplied visual evidence using the provided rubric. Return only the requested JSON object, with no Markdown fences.",
                messages=[{"role": "user", "content": evidence_content(prompt, artifacts, args.reference_image if mode == "score" else None)}],
                thinking={"type": "adaptive"}, output_config={"effort": "medium"},
            )
            text = "\n".join(block.text for block in response.content if block.type == "text")
            parsed = parse_json(text)
            if mode == "gate" and parsed.get("verdict") not in {"pass", "partial", "fail", "not_applicable"}:
                raise ValueError("judge returned an invalid trajectory verdict")
            result["gate" if mode == "gate" else "rubric"] = parsed
            result["calls"].append({"mode": mode, "model": response.model, "response_id": response.id, "usage": response.usage.model_dump()})
            args.output.write_text(json.dumps(result, indent=2))
    result["reported_score"] = effective_score(result["gate"], result["rubric"])
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"task_id": args.task_id, "judge": args.model, "verdict": result["gate"].get("verdict") if result["gate"] else None, "score": result["reported_score"], "dry_run": args.dry_run}))


if __name__ == "__main__":
    main()
