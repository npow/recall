"""PlayWorld's released Gemini rubric verifier, run through Vertex AI with gcloud credentials.

Prompts, dual-scale sampling (10 FPS 384x216 5x5 primary + 0.5 FPS 800x450 2x2 detail),
rubric, instruction gate and aggregation are the released PlayWorld code, imported
unchanged. Only the transport differs from `metrics.vqa.score`:

* Vertex AI endpoint (`genai.Client(vertexai=True, ...)`) with refreshing Google
  credentials instead of an AI-Studio API key. The Files API is not used; evidence is
  sent inline.
* Model id `gemini-3.1-pro-preview` (plain `gemini-3.1-pro` is 404 on the global
  Vertex endpoint for our project). This may not be the exact checkpoint behind the
  paper's table.
* Flow mirrors `worldsync.judge` (our Claude judge): gate first, then the rubric only
  when the gate does not fail. The released aggregator scores a failed gate as 1.0, so
  the rubric call would not change the aggregate.

Secrets are never printed or persisted: tokens stay in process memory.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_LOCATION = "global"
DEFAULT_PLAYWORLD = Path("external/PlayWorld")
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
TOKEN_LIFETIME_S = 45 * 60  # gcloud access tokens last about an hour
RETRYABLE = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
             "DEADLINE_EXCEEDED", "INTERNAL", "timed out", "Connection")


def load_playworld(root):
    root = Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from metrics.vqa import score as released_score
    from metrics.vqa import rubric, instruction_gate, sampling
    return released_score, rubric, instruction_gate, sampling


def gcloud_credentials_class():
    from google.auth import credentials as google_credentials

    class GcloudAccessTokenCredentials(google_credentials.Credentials):
        """Refreshes by shelling out to `gcloud auth print-access-token`."""

        def __init__(self, account=None, runner=subprocess.check_output, clock=time.time):
            super().__init__()
            self.account = account
            self._runner = runner
            self._clock = clock

        def refresh(self, request):
            command = ["gcloud", "auth", "print-access-token"]
            if self.account:
                command.append(f"--account={self.account}")
            token = self._runner(command, text=True, stderr=subprocess.DEVNULL).strip()
            if not token:
                raise RuntimeError("gcloud returned an empty access token; run `gcloud auth login`")
            self.token = token
            expiry = datetime.datetime.fromtimestamp(self._clock() + TOKEN_LIFETIME_S, datetime.timezone.utc)
            self.expiry = expiry.replace(tzinfo=None)  # google-auth compares against naive UTC

    return GcloudAccessTokenCredentials


def resolve_project(project=None, runner=subprocess.check_output):
    if project:
        return project
    for variable in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        if os.environ.get(variable):
            return os.environ[variable]
    value = runner(["gcloud", "config", "get-value", "project"], text=True, stderr=subprocess.DEVNULL).strip()
    if not value:
        raise RuntimeError("No Google Cloud project: pass --project or run `gcloud config set project`")
    return value


def make_credentials(prefer="gcloud", account=None):
    """The user's gcloud login. `prefer='adc'` tries application-default credentials first."""
    order = ["adc", "gcloud"] if prefer == "adc" else ["gcloud"]
    errors = []
    for kind in order:
        try:
            if kind == "adc":
                import google.auth
                credentials, _ = google.auth.default(scopes=SCOPES)
            else:
                credentials = gcloud_credentials_class()(account=account)
            from google.auth.transport.requests import Request
            credentials.refresh(Request())
            return credentials
        except Exception as error:  # fall through to the next credential source
            errors.append(f"{kind}: {type(error).__name__}")
    raise RuntimeError("No usable Google credentials (" + "; ".join(errors) + ")")


def create_vertex_client(project=None, location=DEFAULT_LOCATION, credentials=None, prefer="gcloud", account=None):
    from google import genai
    credentials = credentials or make_credentials(prefer, account)
    return genai.Client(vertexai=True, project=resolve_project(project), location=location, credentials=credentials)


def generate_with_retries(client, model, contents, config, retries=5, base_delay=4.0, sleep=time.sleep):
    last = None
    for attempt in range(retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as error:
            text = f"{type(error).__name__}: {error}"
            if attempt == retries or not any(token in text for token in RETRYABLE):
                raise
            last = error
            sleep(min(120.0, base_delay * (2 ** attempt)))
    raise last  # pragma: no cover


def unwrap_json_object(parsed):
    """Gemini occasionally wraps the requested JSON object in a one-element list."""
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        return parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError("judge must return a single JSON object")
    return parsed


def metric_task_record(task):
    """Same bridge as worldsync.judge: the gate dispatcher expects hyphenated perspectives."""
    return {**task, "task_id": task.get("source_task_id") or task["task_id"],
            "perspective": str(task.get("perspective") or "").strip().lower().replace(" ", "-")}


def cache_path(cache_dir, video_sha, task_id, mode, model, prompt_sha):
    key = hashlib.sha256(f"{video_sha}|{task_id}|{mode}|{model}|{prompt_sha}".encode()).hexdigest()[:24]
    return Path(cache_dir) / f"{task_id}-{mode}-{key}.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_scores(gate, rubric):
    """Released aggregation for one episode: Fail = 1, GC uses the weighted categories."""
    if gate is not None and gate.get("verdict") == "fail":
        return {"official": 1.0, "final_score": 1.0}
    if rubric is None:
        return {"official": None, "final_score": None}
    from metrics.aggregate import normalized_score  # released code
    key = "/GC" if str(rubric.get("task_id", "")).startswith("GC") else "/OE"
    official = normalized_score(key, rubric)
    final = rubric.get("final_score_1_to_5", rubric.get("score"))
    return {"official": official, "final_score": float(final) if isinstance(final, (int, float)) else None}


def judge_episode(client, playworld, dataset_task, video, reference_image=None, world_model="LingBot-World-Infinity",
                  model=DEFAULT_MODEL, cache_dir=None, dry_run=False, mode="both", retries=5):
    released_score, rubric_module, gate_module, sampling = load_playworld(playworld)
    from google.genai import types

    metric_task = metric_task_record(dataset_task)
    item = {"model": world_model, "task": metric_task["task_id"],
            "group": dataset_task.get("source_category") or dataset_task.get("category"),
            "view": metric_task.get("perspective"), "caption": dataset_task.get("image_caption"), "status": "generated"}
    video = Path(video)
    video_sha = sha256_file(video)
    result = {"task_id": dataset_task["task_id"], "judge_provider": "google-vertex", "judge_model": model,
              "endpoint": "vertex-ai-global", "video_sha256": video_sha, "dry_run": dry_run,
              "protocol_warning": "Vertex endpoint and model id differ from the paper's Gemini 3.1 Pro setup; controller, adapter and checkpoint also differ.",
              "gate": None, "rubric": None, "calls": []}
    with tempfile.TemporaryDirectory(prefix="worldsync-gemini-") as directory:
        artifacts = sampling.generate_dual_sampling(video, Path(directory))
        result["sampling"] = artifacts.metadata()
        for step in (["gate", "score"] if mode == "both" else [mode]):
            if step == "gate" and gate_module.gate_kind(metric_task) is None:
                result["gate"] = {"verdict": "not_applicable", "instruction_following_score": None}
                continue
            if step == "score" and result["gate"] and result["gate"].get("verdict") == "fail":
                result["rubric_skipped"] = "trajectory failed; aggregate score is 1"
                break
            if step == "gate":
                prompt = gate_module.build_gate_prompt(item, metric_task, result["sampling"])
                reference = None
            else:
                prompt = rubric_module.build_scoring_prompt(item, metric_task, len(artifacts.primary.frames), fps=10,
                                                            sampling_metadata=result["sampling"])
                reference = Path(reference_image) if reference_image else None
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            result[f"{step}_prompt_sha256"] = prompt_sha
            cached = cache_path(cache_dir, video_sha, dataset_task["task_id"], step, model, prompt_sha) if cache_dir else None
            if cached and cached.is_file():
                parsed = unwrap_json_object(json.loads(cached.read_text()))
                result["calls"].append({"mode": step, "cached": True})
            elif dry_run:
                continue
            else:
                parts = released_score.build_evidence_parts(prompt, artifacts, reference)
                response = generate_with_retries(
                    client, model, types.Content(role="user", parts=parts),
                    types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1), retries=retries)
                parsed = unwrap_json_object(released_score.json_from_response(response.text))
                usage = getattr(response, "usage_metadata", None)
                result["calls"].append({"mode": step, "cached": False,
                                        "prompt_tokens": getattr(usage, "prompt_token_count", None),
                                        "output_tokens": getattr(usage, "candidates_token_count", None)})
                if cached:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
            if step == "gate":
                if parsed.get("verdict") not in {"pass", "partial", "fail", "not_applicable"}:
                    raise ValueError("judge returned an invalid trajectory verdict")
                result["gate"] = parsed
            else:
                parsed.setdefault("task_id", dataset_task["task_id"])
                result["rubric"] = parsed
    if not dry_run:
        result["scores"] = effective_scores(result["gate"], result["rubric"])
    return result


def load_dataset_task(playworld, split, task_id):
    records = json.loads((Path(playworld) / "data" / split / "data.json").read_text())
    for record in records:
        if record["task_id"] == task_id:
            return record
    raise KeyError(f"{task_id} not in {split}")


def run_manifest(manifest, output_dir, playworld=DEFAULT_PLAYWORLD, model=DEFAULT_MODEL, workers=3, dry_run=False,
                 project=None, location=DEFAULT_LOCATION, prefer="gcloud", limit=None):
    """Manifest: jsonl of {episode_id, split, task_id, video, reference_image?, world_model?}."""
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    rows = rows[:limit] if limit else rows
    output_dir = Path(output_dir)
    (output_dir / "episodes").mkdir(parents=True, exist_ok=True)
    client = None if dry_run else create_vertex_client(project, location, prefer=prefer)
    lock = threading.Lock()
    summary = []

    def work(row):
        target = output_dir / "episodes" / f"{row['episode_id']}.json"
        if target.is_file() and not dry_run:
            previous = json.loads(target.read_text())
            if not previous.get("error"):  # errored episodes are retried; completed calls stay cached
                return previous
        task = load_dataset_task(playworld, row["split"], row["task_id"])
        try:
            result = judge_episode(client, playworld, task, row["video"], row.get("reference_image"),
                                   row.get("world_model", "LingBot-World-Infinity"), model,
                                   cache_dir=output_dir / "cache", dry_run=dry_run)
        except Exception as error:  # record and continue; the batch is resumable
            result = {"task_id": row["task_id"], "error": f"{type(error).__name__}: {str(error)[:300]}"}
        result["episode_id"] = row["episode_id"]
        if not dry_run:
            with lock:
                target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for result in pool.map(work, rows):
            summary.append(result)
            print(json.dumps({"episode": result["episode_id"], "task": result["task_id"],
                              "gate": (result.get("gate") or {}).get("verdict"),
                              "score": (result.get("scores") or {}).get("official"),
                              "error": result.get("error")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--playworld", type=Path, default=DEFAULT_PLAYWORLD)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--project")
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--credentials", choices=["gcloud", "adc"], default="gcloud")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Sample video and build prompts; no API call")
    args = parser.parse_args()
    run_manifest(args.manifest, args.output_dir, args.playworld, args.model, args.workers, args.dry_run,
                 args.project, args.location, args.credentials, args.limit)


if __name__ == "__main__":
    main()
