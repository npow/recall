"""Complete-video comparisons for the predeclared protected-recent ablation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproduction/protected-recent-memory"


def fetch(label, manifest_path, directory_root=OUTPUT):
    manifest = json.loads(manifest_path.read_text())
    run = manifest["run"]
    if run["status"] != "completed" or len(run["chunks"]) != 32:
        raise ValueError(f"Incomplete variant{label}")
    directory = directory_root / label
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "run.json"
    if target.exists() and json.loads(target.read_text())["result_directory"] != manifest["result_directory"]:
        raise ValueError("Preserve prior artifacts before replacing a generation")
    shutil.copy2(manifest_path, target)
    video = directory / "rollout.mp4"
    if not video.exists():
        profile = manifest.get("workspace", "nissan-pow")
        client = str(ROOT / ".venv-modal-new/bin/modal") if profile == "krasnaya-zvesda" else "modal"
        subprocess.run([client, "volume", "get", "--profile", profile, "worldsync-lingbot-results",
                        manifest["result_directory"] + "/rollout.mp4", str(video)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                    "fps=1,scale=416:-1,tile=4x8", "-frames:v", "1", str(directory / "contact.jpg")], check=True)
    return manifest


def compare(pair, judge, directory_root=OUTPUT):
    reference, candidate = pair
    output = directory_root / f"{reference}-vs-{candidate}.json"
    previous = json.loads(output.read_text()) if output.exists() else {}
    if not previous.get("feature_model") or (judge and not previous.get("visual_judge")):
        command = [sys.executable, str(ROOT / "scripts/compare_lingbot_roundtrip.py"),
                   "--comparison", "memory-context", "--reference-label", reference,
                   "--candidate-label", candidate, "--output", str(output), "--dino"]
        for flag, label, file in (("stock-run", reference, "run.json"), ("candidate-run", candidate, "run.json"),
                                  ("stock-video", reference, "rollout.mp4"), ("candidate-video", candidate, "rollout.mp4")):
            command.extend(["--" + flag, str(directory_root / label / file)])
        if judge:
            command.append("--judge")
        log_path = ROOT / "outputs/lingbot-reproduction" / f"protected-recent-{reference}-vs-{candidate}.log"
        with log_path.open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, "HF_HUB_OFFLINE": "1"})
    report = json.loads(output.read_text())
    result = {"reference": reference, "candidate": candidate, "status": "completed",
              "reference_dino": report[reference]["endpoint_dino_cosine"],
              "candidate_dino": report[candidate]["endpoint_dino_cosine"],
              "reference_peak_bytes": report["reference_peak_allocated_bytes"],
              "candidate_peak_bytes": report["candidate_peak_allocated_bytes"],
              "warm_suffix_s": report["warmed_suffix_s"],
              "pixel_identical_prefix_chunks": report["pre_intervention_pixel_identical_chunks"]}
    if "visual_judge" in report:
        reviewed = report["visual_judge"]
        preference = reviewed["result"]["preference"]
        result.update(preference=reviewed["label_mapping"].get(preference, preference),
                      assessment=reviewed["result"], label_mapping=reviewed["label_mapping"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available", action="store_true")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    plan = json.loads((ROOT / "reproduction/protected-recent-memory-plan.json").read_text())
    reference = ROOT / plan["reference"]
    parity = reference.parent / "comparison.json"
    if not reference.exists() or not parity.exists():
        print("Waiting for the full32-chunk control parity check")
        return
    check = json.loads(parity.read_text())
    if not check["all_pixels_identical"] or not all(check["matched_settings"].values()):
        raise ValueError("Read-buffer lifetime control did not preserve the reference")
    sources = {"reference18_read4": reference}
    for label, mode in (("context22", "context22"), ("retrieval_context22", "retrieval-context22"),
                        ("retrieval8_context22", "retrieval-8-nearest-context22")):
        sources[label] = ROOT / "outputs/lingbot-reproduction" / f"protected-recent-{mode}.json"
    if not args.available and not all(path.exists() for path in sources.values()):
        raise ValueError("Some declared variants are incomplete; use --available for provisional comparisons")
    manifests = {label: fetch(label, path) for label, path in sources.items() if path.exists()}
    pairs = [pair for pair in plan["comparisons"] if all(label in manifests for label in pair)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda pair: compare(pair, args.judge), pairs))
    summary = {"scope": plan["scope"], "declared_variants": len(sources),
               "completed_variants": len(manifests), "declared_comparisons": len(plan["comparisons"]),
               "complete": len(results) == len(plan["comparisons"]), "comparisons": results,
               "artifacts": {label: {"result_directory": manifest["result_directory"],
                                     "video_sha256": hashlib.sha256((OUTPUT / label / "rollout.mp4").read_bytes()).hexdigest()}
                             for label, manifest in manifests.items()},
               "caveat": "Development-only fixed trajectory. Larger attention capacity and retrieval eligibility differ from18-frame reference. No held-out gain or leaderboard claim."}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
