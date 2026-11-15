"""Compare a controlled native stock/retrieval pair; no leaderboard score."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stock_run", "candidate_run", "stock_video", "candidate_video", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--dino", action="store_true", help="Add pinned DINOv2-small endpoint feature similarity")
    args = parser.parse_args()
    stock = json.loads(args.stock_run.read_text())["run"]
    candidate = json.loads(args.candidate_run.read_text())["run"]
    fields = ("source_revision", "checkpoint_revision", "protocol_id", "task_id", "split", "controls_sha256", "seed", "fps", "max_chunks",
              "max_area", "chunk_size", "local_attn_size", "sink_size", "sampler_timesteps_index", "sample_shift", "fixed_actions")
    for field in fields:
        if stock[field] != candidate[field]:
            raise ValueError(f"Unmatched comparison setting: {field}")
    for run in (stock, candidate):
        if run["status"] != "completed" or len(run["chunks"]) != run["max_chunks"]:
            raise ValueError("Comparison requires two complete paths")
        if not run["native_fixed_controls"] or not all(a["within_tolerance"] for a in run["camera_embedding_audit"]):
            raise ValueError("Native camera conditioning audit failed")
    memory = candidate["memory"]
    modified = [item["chunk"] for item in memory["trace"] if item["attention_replaced"]]
    if not modified:
        raise ValueError("Candidate never retrieved memory; no intervention to compare")
    first = min(modified)
    identical_prefix = all(a["pixel_sha256"] == b["pixel_sha256"] for a, b in zip(stock["chunks"][:first], candidate["chunks"][:first]))
    if not identical_prefix:
        raise ValueError("Pre-intervention raw pixels differ; comparison is confounded")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = output.parent / (output.stem + "-artifacts")
    artifacts.mkdir(exist_ok=True)
    from PIL import Image
    import numpy as np
    endpoints = {}
    for label, video, run in (("stock", args.stock_video, stock), ("retrieval", args.candidate_video, candidate)):
        for part, frame in (("reference", 12), ("return", run["generated_frames"] - 1)):
            image = artifacts / f"{label}-{part}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{frame})",
                            "-frames:v", "1", str(image)], check=True)
            endpoints[label, part] = image
    report = {"scope": "single controlled development round trip; no PlayWorld score or SOTA claim",
        "matched_settings": {key: stock[key] for key in fields}, "native_camera_inputs_verified": True,
        "pre_intervention_pixel_identical_chunks": first, "modified_chunks": modified,
        "stock_wall_s": stock["elapsed_s"], "retrieval_wall_s": candidate["elapsed_s"],
        "stock_peak_allocated_bytes": stock["peak_allocated_bytes"],
        "retrieval_peak_allocated_bytes": candidate["peak_allocated_bytes"], "memory": memory,
        "caveat": "Commanded camera closure does not prove realized closure. Endpoint image similarity cannot establish world memory if the realized route fails."}
    for label in ("stock", "retrieval"):
        reference = np.asarray(Image.open(endpoints[label, "reference"])).astype(np.float32)
        returned = np.asarray(Image.open(endpoints[label, "return"])).astype(np.float32)
        mse = float(np.mean((reference - returned) ** 2))
        report[label] = {"endpoint_psnr_db": float(10 * np.log10(255 ** 2 / max(mse, 1e-12))),
                         "video_sha256": hashlib.sha256((args.stock_video if label == "stock" else args.candidate_video).read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.dino:
        import torch
        from urllib.request import Request, urlopen
        import requests
        from huggingface_hub import configure_http_backend
        from transformers import AutoImageProcessor, AutoModel
        model_name = "facebook/dinov2-small"
        with urlopen(Request(f"https://huggingface.co/api/models/{model_name}",
                             headers={"Accept-Encoding": "identity"}), timeout=60) as response:
            revision = json.load(response)["sha"]
        def backend():
            session = requests.Session()
            session.headers["Accept-Encoding"] = "identity"
            return session
        configure_http_backend(backend_factory=backend)
        processor = AutoImageProcessor.from_pretrained(model_name, revision=revision, use_fast=False)
        model = AutoModel.from_pretrained(model_name, revision=revision).eval()
        torch.set_num_threads(8)
        order = [(label, part) for label in ("stock", "retrieval") for part in ("reference", "return")]
        inputs = processor(images=[Image.open(endpoints[key]).convert("RGB") for key in order], return_tensors="pt")
        with torch.no_grad():
            features = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0], dim=-1)
        for i, label in enumerate(("stock", "retrieval")):
            report[label]["endpoint_dino_cosine"] = float(features[i * 2] @ features[i * 2 + 1])
        report["feature_model"] = {"name": model_name, "revision": revision, "feature": "normalized CLS token"}
        output.write_text(json.dumps(report, indent=2) + "\n")
    if args.judge:
        from worldsync.credentials import load_local_credentials
        from worldsync.judge import evidence_content, parse_json
        load_local_credentials()
        from anthropic import Anthropic
        sys.path.insert(0, str(ROOT / "external/PlayWorld"))
        from metrics.vqa.sampling import generate_dual_sampling
        prompt = (
            "Evaluate two anonymized world-model videos independently and then compare. Each was given the same "
            "image and fixed camera path: initial hold, four seconds forward, three seconds yaw right (180 degrees), "
            "four seconds forward back along the intended path, three seconds yaw right to restore the original "
            "heading, and final hold. Commands are NOT proof that motion happened. Inspect actual frames. "
            "Assess whether each video realizes a turnaround and return to the starting viewpoint; whether persistent "
            "building identity, sign text, geometry and spatial layout match on return; and whether there are sudden "
            "jumps, frozen motion or implausible transitions. Distinguish camera-control failures from forgetting. "
            "Do not reward a similar final building if reaching it required a reset. Do not invent unseen geometry. "
            "Return English JSON with keys A and B (each with realized_round_trip: yes|no|uncertain, "
            "starting_place_recovered: yes|no|uncertain, discontinuity: yes|no|uncertain, visual_evidence, "
            "failure_classification), preference: A|B|tie|neither, confidence: low|medium|high, and reasoning. "
            "This is a diagnostic comparison, not a benchmark score."
        )
        content = [{"type": "text", "text": prompt}]
        mapping = {"A": ("retrieval", args.candidate_video), "B": ("stock", args.stock_video)}
        with tempfile.TemporaryDirectory(prefix="roundtrip-judge-") as directory:
            for anonymous, (_, video) in mapping.items():
                sampled = generate_dual_sampling(video, Path(directory) / anonymous)
                content.extend(evidence_content(f"VIDEO {anonymous}: inspect this complete sequence.", sampled))
            client = Anthropic(timeout=240, max_retries=2, default_headers={"Accept-Encoding": "identity"})
            response = client.messages.create(model="claude-opus-5-5", max_tokens=6000,
                thinking={"type": "adaptive"}, output_config={"effort": "medium"},
                messages=[{"role": "user", "content": content}])
        raw = "\n".join(block.text for block in response.content if block.type == "text")
        report["visual_judge"] = {"model": response.model, "response_id": response.id,
            "usage": response.usage.model_dump(), "label_mapping": {k: v[0] for k, v in mapping.items()},
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "result": parse_json(raw)}
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"memory", "matched_settings"}}, indent=2))


if __name__ == "__main__":
    main()
