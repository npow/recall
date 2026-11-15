"""Secondary-workspace joint versus separately normalized history attention."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.modal_lingbot_roundtrip import app, generate_h100


@app.local_entrypoint(name="history-gate")
def main(workspace: str = "krasnaya-zvesda"):
    plan = json.loads((ROOT / "reproduction/separate-history-attention-plan.json").read_text())
    verified = ROOT / "outputs/lingbot-reproduction" / f"weights-verified-{workspace}.json"
    for attempt in range(120):
        if workspace == "nissan-pow":
            break
        if verified.exists():
            report = json.loads(verified.read_text())
            if report["status"] != "verified" or report["revision"] != "5c33dd40b213598c418fd25bff30fdbd23fd38a7":
                raise ValueError("Secondary checkpoint verification did not pass")
            break
        print("Waiting for the running CPU checkpoint verifier; no GPU allocated", flush=True)
        time.sleep(15)
    else:
        raise RuntimeError("Secondary checkpoint verification is still incomplete")
    case = plan["case"]
    pending, destinations = [], []
    for variant in plan["variants"]:
        output = ROOT / "outputs/lingbot-reproduction" / f"history-attention-{variant['label']}.json"
        if output.exists() and json.loads(output.read_text())["run"]["status"] == "completed":
            print(f"Reusing completed {variant['label']}", flush=True)
            continue
        pending.append((case["path"], case["seed"], True, 22, case["task_id"], False,
                        "virtual", 60, 8, "nearest", True, variant["history_weight"]))
        destinations.append(output)
    failure_file = ROOT / "outputs/lingbot-reproduction/history-attention-failures.json"
    failures = json.loads(failure_file.read_text()) if failure_file.exists() else []
    for index, result in enumerate(generate_h100.starmap(pending, return_exceptions=True)):
        if isinstance(result, BaseException):
            failures.append({"job": pending[index], "error": str(result)})
            failure_file.write_text(json.dumps(failures, indent=2) + "\n")
            continue
        result["workspace"] = workspace
        destinations[index].write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "run"}, indent=2), flush=True)
