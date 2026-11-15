"""Fetch and evaluate the frozen context pilot pairs, preserving every result."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def evaluate(case, output, judge):
    name = f"{case['task_id']}-seed{case['seed']}"
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for mode in ("stock", "context34"):
        source = ROOT / "outputs/lingbot-reproduction" / f"validation-{case['task_id']}-{mode}-seed{case['seed']}.json"
        manifest = json.loads(source.read_text())
        manifests[mode] = manifest
        shutil.copy2(source, directory / f"{mode}-run.json")
        video = directory / f"{mode}.mp4"
        if not video.exists():
            subprocess.run(["modal", "volume", "get", "--profile", "nissan-pow",
                            "worldsync-lingbot-results", manifest["result_directory"] + "/rollout.mp4", str(video)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                        "fps=1,scale=416:-1,tile=4x2", "-frames:v", "1", str(directory / f"{mode}-contact.jpg")], check=True)
    comparison = directory / "comparison.json"
    existing = json.loads(comparison.read_text()) if comparison.exists() else {}
    if not existing.get("feature_model") or (judge and not existing.get("visual_judge")):
        command = [sys.executable, str(ROOT / "scripts/compare_lingbot_roundtrip.py"), "--comparison", "context",
                   "--stock-run", str(directory / "stock-run.json"), "--candidate-run", str(directory / "context34-run.json"),
                   "--stock-video", str(directory / "stock.mp4"), "--candidate-video", str(directory / "context34.mp4"),
                   "--output", str(comparison), "--dino"]
        if judge:
            command.append("--judge")
        with (directory / "evaluation.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads(comparison.read_text())
    summary = {**case, "stock_dino": report["stock"]["endpoint_dino_cosine"],
               "candidate_dino": report["context34"]["endpoint_dino_cosine"],
               "stock_psnr_db": report["stock"]["endpoint_psnr_db"],
               "candidate_psnr_db": report["context34"]["endpoint_psnr_db"],
               "warmed_suffix_s": report["warmed_suffix_s"],
               "peak_allocated_bytes": {"stock": report["stock_peak_allocated_bytes"],
                                        "context34": report["candidate_peak_allocated_bytes"]}}
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
    parser.add_argument("--available", action="store_true", help="Evaluate only fully generated pairs while GPU jobs continue")
    args = parser.parse_args()
    plan = json.loads((ROOT / "reproduction/context-validation-plan.json").read_text())
    cases = plan["cases"]
    if args.available:
        cases = [case for case in cases if all((ROOT / "outputs/lingbot-reproduction" /
                 f"validation-{case['task_id']}-{mode}-seed{case['seed']}.json").exists() for mode in ("stock", "context34"))]
    output = ROOT / "reproduction/context34-validation"
    output.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda case: evaluate(case, output, args.judge), cases))
    summary = {"scope": plan["scope"], "complete_pairs": len(results), "declared_pairs": len(plan["cases"]),
               "settings": plan["settings_frozen"], "results": results,
               "caveat": "Fixed-path memory mechanism validation, not an adaptive PlayWorld score or a claim about minute-long worlds."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
