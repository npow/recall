"""Run only the frozen protected-recent validation pairs, in parallel on H100."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.modal_lingbot_roundtrip import app, generate_h100


@app.local_entrypoint(name="protected-validation")
def main(plan_name: str = "protected-recent-validation-plan.json", workspace: str = "nissan-pow"):
    plan = json.loads((ROOT / "reproduction" / plan_name).read_text())
    if workspace != "nissan-pow":
        verified = ROOT / "outputs/lingbot-reproduction" / f"weights-verified-{workspace}.json"
        for attempt in range(120):
            if verified.exists():
                check = json.loads(verified.read_text())
                if check["status"] != "verified" or check["revision"] != "5c33dd40b213598c418fd25bff30fdbd23fd38a7":
                    raise ValueError("Checkpoint verification failed")
                break
            print("Waiting for CPU checkpoint verification; no GPU allocated", flush=True)
            time.sleep(15)
        else:
            raise RuntimeError("Checkpoint verification remains incomplete")
    prefix = plan.get("output_prefix", "protected-validation")
    tasks = json.loads((ROOT / "external/PlayWorld/data/gc/data.json").read_text())
    jobs, paths = [], []
    for case in plan["cases"]:
        task = next(row for row in tasks if row["task_id"] == case["task_id"])
        image = ROOT / "external/PlayWorld/data" / task["image_path"]
        if (hashlib.sha256(image.read_bytes()).hexdigest() != case["image_sha256"]
                or hashlib.sha256(task["image_caption"].encode()).hexdigest() != case["caption_sha256"]):
            raise ValueError("Source input changed since semantic audit")
        for variant in plan["variants"]:
            output = ROOT / "outputs/lingbot-reproduction" / f"{prefix}-{case['task_id']}-seed{case['seed']}-{variant['label']}.json"
            if output.exists() and json.loads(output.read_text())["run"]["status"] == "completed":
                print(f"Reusing completed {output.name}", flush=True)
                continue
            jobs.append((case["path"], case["seed"], True, variant["cache_frames"], case["task_id"],
                         False, "virtual", plan.get("turn_degrees_per_second", 60), variant["retrieved_frames"], "nearest", True))
            paths.append(output)
    failure_path = ROOT / f"outputs/lingbot-reproduction/{prefix}-failures.json"
    failures = json.loads(failure_path.read_text()) if failure_path.exists() else []
    for index, result in enumerate(generate_h100.starmap(jobs, return_exceptions=True)):
        if isinstance(result, BaseException):
            failures.append({"job": jobs[index], "error": str(result)})
            failure_path.write_text(json.dumps(failures, indent=2) + "\n")
            continue
        result["workspace"] = workspace
        paths[index].write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "run"}, indent=2), flush=True)
