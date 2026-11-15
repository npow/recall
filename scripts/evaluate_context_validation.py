"""Fetch and evaluate the frozen context pilot pairs, preserving every result."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def evaluate(case, output, judge, memory=False, input_prefix="memory-validation"):
    path = case.get("path", "turn_return")
    name = f"{case['task_id']}-seed{case['seed']}" + (f"-{path}" if memory else "")
    candidate = "retrieval" if memory else "context34"
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for mode in ("stock", candidate):
        filename = (f"{input_prefix}-{case['task_id']}-{mode}-{path}-seed{case['seed']}.json" if memory
                    else f"validation-{case['task_id']}-{mode}-seed{case['seed']}.json")
        source = ROOT / "outputs/lingbot-reproduction" / filename
        manifest = json.loads(source.read_text())
        if manifest["run"]["task_id"] != case["task_id"] or manifest["run"]["seed"] != case["seed"] or manifest["path"] != path:
            raise ValueError("Generation manifest does not belong to the declared case")
        manifests[mode] = manifest
        saved_manifest = directory / f"{mode}-run.json"
        if saved_manifest.exists() and json.loads(saved_manifest.read_text())["result_directory"] != manifest["result_directory"]:
            raise ValueError(f"{saved_manifest} belongs to a different generation; preserve it and choose a fresh artifact directory")
        shutil.copy2(source, saved_manifest)
        video = directory / f"{mode}.mp4"
        if not video.exists():
            subprocess.run(["modal", "volume", "get", "--profile", "nissan-pow",
                            "worldsync-lingbot-results", manifest["result_directory"] + "/rollout.mp4", str(video)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                        f"fps=1,scale=416:-1,tile=4x{(manifest['run']['max_chunks'] + 3) // 4}", "-frames:v", "1", str(directory / f"{mode}-contact.jpg")], check=True)
    comparison = directory / "comparison.json"
    existing = json.loads(comparison.read_text()) if comparison.exists() else {}
    for mode in ("stock", candidate):
        if existing.get(mode, {}).get("video_sha256") and existing[mode]["video_sha256"] != hashlib.sha256((directory / f"{mode}.mp4").read_bytes()).hexdigest():
            raise ValueError(f"Cached comparison refers to a different {mode} video")
    if not existing.get("feature_model") or (judge and not existing.get("visual_judge")):
        command = [sys.executable, str(ROOT / "scripts/compare_lingbot_roundtrip.py"), "--comparison", "retrieval" if memory else "context",
                   "--stock-run", str(directory / "stock-run.json"), "--candidate-run", str(directory / f"{candidate}-run.json"),
                   "--stock-video", str(directory / "stock.mp4"), "--candidate-video", str(directory / f"{candidate}.mp4"),
                   "--output", str(comparison), "--dino"]
        if judge:
            command.append("--judge")
        with (ROOT / "outputs/lingbot-reproduction" / f"context-evaluation-{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads(comparison.read_text())
    summary = {**case, "stock_dino": report["stock"]["endpoint_dino_cosine"],
               "candidate_dino": report[candidate]["endpoint_dino_cosine"],
               "stock_psnr_db": report["stock"]["endpoint_psnr_db"],
               "candidate_psnr_db": report[candidate]["endpoint_psnr_db"],
               "warmed_suffix_s": report["warmed_suffix_s"],
               "peak_allocated_bytes": {"stock": report["stock_peak_allocated_bytes"],
                                        candidate: report["candidate_peak_allocated_bytes"]}}
    if "visual_judge" in report:
        assessment = report["visual_judge"]
        preference = assessment["result"]["preference"]
        summary["visual_preference"] = assessment["label_mapping"].get(preference, preference)
        summary["visual_assessment"] = assessment["result"]
        summary["label_mapping"] = assessment["label_mapping"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--memory", action="store_true", help="Evaluate the frozen beyond-window memory validation instead of context34")
    parser.add_argument("--available", action="store_true", help="Evaluate only fully generated pairs while GPU jobs continue")
    args = parser.parse_args()
    plan_file = "memory-validation-plan.json" if args.memory else "context-validation-plan.json"
    plan = json.loads((ROOT / "reproduction" / plan_file).read_text())
    cases = plan["cases"]
    if args.available:
        def ready(case):
            for mode in ("stock", "retrieval" if args.memory else "context34"):
                filename = (f"memory-validation-{case['task_id']}-{mode}-{case['path']}-seed{case['seed']}.json" if args.memory
                            else f"validation-{case['task_id']}-{mode}-seed{case['seed']}.json")
                if not (ROOT / "outputs/lingbot-reproduction" / filename).exists():
                    return False
            return True
        cases = [case for case in cases if ready(case)]
    output = ROOT / "reproduction" / ("memory-validation" if args.memory else "context34-validation")
    output.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda case: evaluate(case, output, args.judge, args.memory), cases))
    summary = {"scope": plan["scope"], "complete_pairs": len(results), "declared_pairs": len(plan["cases"]),
               "settings": plan.get("settings_frozen", plan.get("candidate_frozen")), "results": results,
               "caveat": "Fixed-path memory mechanism validation, not an adaptive PlayWorld score or a claim about minute-long worlds."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
