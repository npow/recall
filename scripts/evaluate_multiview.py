"""Evaluate every declared view-count/diversity variant against four nearest views."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def label(variant):
    return f"{variant['selection']}{variant['retrieved_frames']}"


def manifest_path(case, variant):
    mode = "retrieval" if variant == {"retrieved_frames": 4, "selection": "nearest"} else f"retrieval-{variant['retrieved_frames']}-{variant['selection']}"
    return ROOT / "outputs/lingbot-reproduction" / f"multiview-{case['task_id']}-{mode}-{case['path']}-seed{case['seed']}.json"


def fetch(case, variant, output):
    source = manifest_path(case, variant)
    manifest = json.loads(source.read_text())
    if manifest["run"]["status"] != "completed":
        raise ValueError("Variant did not complete")
    directory = output / label(variant)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "run.json"
    if target.exists() and json.loads(target.read_text())["result_directory"] != manifest["result_directory"]:
        raise ValueError("Preserve existing artifacts before evaluating a different generation")
    shutil.copy2(source, target)
    video = directory / "rollout.mp4"
    if not video.exists():
        subprocess.run(["modal", "volume", "get", "--profile", "nissan-pow", "worldsync-lingbot-results",
                        manifest["result_directory"] + "/rollout.mp4", str(video)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                    "fps=1,scale=416:-1,tile=4x8", "-frames:v", "1", str(directory / "contact.jpg")], check=True)
    return directory, manifest


def compare(case, variant, output, judge):
    reference = output / "nearest4"
    directory, manifest = fetch(case, variant, output)
    comparison = directory / "comparison.json"
    previous = json.loads(comparison.read_text()) if comparison.exists() else {}
    if not previous.get("feature_model") or (judge and not previous.get("visual_judge")):
        command = [sys.executable, str(ROOT / "scripts/compare_lingbot_roundtrip.py"), "--comparison", "retrieval",
                   "--reference-label", "nearest4", "--candidate-label", label(variant),
                   "--stock-run", str(reference / "run.json"), "--candidate-run", str(directory / "run.json"),
                   "--stock-video", str(reference / "rollout.mp4"), "--candidate-video", str(directory / "rollout.mp4"),
                   "--output", str(comparison), "--dino"]
        if judge:
            command.append("--judge")
        with (ROOT / "outputs/lingbot-reproduction" / f"multiview-evaluation-{label(variant)}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "HF_HUB_OFFLINE": "1"})
    report = json.loads(comparison.read_text())
    memory = manifest["run"]["memory"]
    changed = [row for row in memory["trace"] if row["attention_replaced"]]
    result = {**variant, "status": "completed", "endpoint_dino": report[label(variant)]["endpoint_dino_cosine"],
              "reference_endpoint_dino": report["nearest4"]["endpoint_dino_cosine"],
              "actual_read_counts": sorted({len(row["retrieved_frame_ids"]) for row in changed}),
              "actual_recent_tail_counts": sorted({row["retained_tail_latent_frames"] for row in changed}),
              "warmed_suffix_s": report["warmed_suffix_s"],
              "peak_allocated_bytes": manifest["run"]["peak_allocated_bytes"]}
    if "visual_judge" in report:
        judge_report = report["visual_judge"]
        preference = judge_report["result"]["preference"]
        result["visual_preference"] = judge_report["label_mapping"].get(preference, preference)
        result["visual_assessment"] = judge_report["result"]
        result["label_mapping"] = judge_report["label_mapping"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available", action="store_true")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    plan = json.loads((ROOT / "reproduction/multiview-retrieval-plan.json").read_text())
    output = ROOT / "reproduction/multiview-retrieval"
    output.mkdir(exist_ok=True)
    reference = plan["variants"][0]
    if not manifest_path(plan["case"], reference).exists():
        print("Reference generation is not complete; no comparison made.")
        return
    fetch(plan["case"], reference, output)
    candidates = plan["variants"][1:]
    if args.available:
        candidates = [variant for variant in candidates if manifest_path(plan["case"], variant).exists()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda variant: compare(plan["case"], variant, output, args.judge), candidates))
    summary = {"scope": plan["scope"], "case": plan["case"], "reference": reference,
               "declared_comparisons": len(plan["variants"]) - 1, "completed_comparisons": len(results),
               "results": results, "caveat": "Development ablation at constant attention budget; eight historical views replace recent context. No held-out selection or leaderboard claim."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
