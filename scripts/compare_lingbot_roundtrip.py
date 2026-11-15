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
    parser.add_argument("--reference-label", default="stock")
    parser.add_argument("--candidate-label")
    parser.add_argument("--comparison", choices=("retrieval", "context", "sink-rephase", "retrieval-native"), default="retrieval")
    parser.add_argument("--dino", action="store_true", help="Add pinned DINOv2-small endpoint feature similarity")
    args = parser.parse_args()
    reference_label = args.reference_label
    candidate_label = args.candidate_label or ("context34" if args.comparison == "context" else args.comparison)
    if not reference_label.isidentifier() or not candidate_label.replace("-", "_").isidentifier() or reference_label == candidate_label:
        raise ValueError("Comparison labels must be distinct simple identifiers")
    stock = json.loads(args.stock_run.read_text())["run"]
    candidate = json.loads(args.candidate_run.read_text())["run"]
    fields = ("source_revision", "checkpoint_revision", "protocol_id", "task_id", "split", "controls_sha256", "seed", "fps", "max_chunks",
              "max_area", "chunk_size", "local_attn_size", "sink_size", "sampler_timesteps_index", "sample_shift", "fixed_actions")
    if args.comparison == "context":
        fields = tuple(field for field in fields if field != "local_attn_size")
    for field in fields:
        if stock[field] != candidate[field]:
            raise ValueError(f"Unmatched comparison setting: {field}")
    for run in (stock, candidate):
        if run["status"] != "completed" or len(run["chunks"]) != run["max_chunks"]:
            raise ValueError("Comparison requires two complete paths")
        if (not run["native_fixed_controls"] or len(run["camera_embedding_audit"]) != run["max_chunks"]
                or not all(a["within_tolerance"] for a in run["camera_embedding_audit"])):
            raise ValueError("Native camera conditioning audit failed")
    if stock.get("hardware") != candidate.get("hardware") or stock.get("text_encoder_offloaded_after_encoding", False) != candidate.get("text_encoder_offloaded_after_encoding", False):
        raise ValueError("Comparison requires matching hardware and text-encoder placement")
    if stock.get("turn_degrees_per_second", 60) != candidate.get("turn_degrees_per_second", 60):
        raise ValueError("Unmatched camera turn rate")
    memory = candidate.get("memory")
    if args.comparison != "context":
        modified = [item["chunk"] for item in memory["trace"] if item["attention_replaced"]]
        reference_modified = [item["chunk"] for item in (stock.get("memory") or {}).get("trace", []) if item["attention_replaced"]]
        if not modified and not reference_modified:
            raise ValueError("Neither policy retrieved memory; no intervention to compare")
        first = min(modified + reference_modified)
    else:
        if stock["local_attn_size"] != 18 or candidate["local_attn_size"] != 34:
            raise ValueError("Declared context pilot requires stock18 versus context34")
        if memory or stock.get("memory") or not candidate["controlled_ablation"]:
            raise ValueError("Context pilot must only change cache capacity")
        first = stock["local_attn_size"] // stock["chunk_size"]
        modified = list(range(first, stock["max_chunks"]))
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
    for label, video, run in ((reference_label, args.stock_video, stock), (candidate_label, args.candidate_video, candidate)):
        for part, frame in (("reference", 12), ("return", run["generated_frames"] - 1)):
            image = artifacts / f"{label}-{part}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{frame})",
                            "-frames:v", "1", str(image)], check=True)
            endpoints[label, part] = image
    report = {"comparison": args.comparison, "cache_frames": {"stock": stock["local_attn_size"], "candidate": candidate["local_attn_size"]}, "scope": "single controlled fixed path; no PlayWorld score or SOTA claim",
        "matched_settings": {key: stock[key] for key in fields}, "native_camera_inputs_verified": True,
        "pre_intervention_pixel_identical_chunks": first, "modified_chunks": modified,
        "stock_wall_s": stock["elapsed_s"], "candidate_wall_s": candidate["elapsed_s"],
        "stock_peak_allocated_bytes": stock["peak_allocated_bytes"],
        "candidate_peak_allocated_bytes": candidate["peak_allocated_bytes"], "memory": memory,
        "caveat": "Commanded camera closure does not prove realized closure. Endpoint image similarity cannot establish world memory if the realized route fails."}
    for label in (reference_label, candidate_label):
        reference = np.asarray(Image.open(endpoints[label, "reference"])).astype(np.float32)
        returned = np.asarray(Image.open(endpoints[label, "return"])).astype(np.float32)
        mse = float(np.mean((reference - returned) ** 2))
        report[label] = {"endpoint_psnr_db": float(10 * np.log10(255 ** 2 / max(mse, 1e-12))),
                         "video_sha256": hashlib.sha256((args.stock_video if label == reference_label else args.candidate_video).read_bytes()).hexdigest()}
    if reference_label != "stock":
        report.pop("stock_wall_s")
        report.pop("stock_peak_allocated_bytes")
    report["reference_label"] = reference_label
    report["reference_memory"] = stock.get("memory")
    report["reference_wall_s"] = stock["elapsed_s"]
    report["reference_peak_allocated_bytes"] = stock["peak_allocated_bytes"]
    report["candidate_label"] = candidate_label
    report["warmed_suffix_s"] = {label: run["chunks"][-1]["elapsed_s"] - run["chunks"][0]["elapsed_s"] for label, run in ((reference_label, stock), (candidate_label, candidate))}
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.dino:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        model_name = "facebook/dinov2-small"
        revision = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
        processor = AutoImageProcessor.from_pretrained(model_name, revision=revision, use_fast=False)
        model = AutoModel.from_pretrained(model_name, revision=revision).eval()
        torch.set_num_threads(8)
        order = [(label, part) for label in (reference_label, candidate_label) for part in ("reference", "return")]
        inputs = processor(images=[Image.open(endpoints[key]).convert("RGB") for key in order], return_tensors="pt")
        with torch.no_grad():
            features = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0], dim=-1)
        for i, label in enumerate((reference_label, candidate_label)):
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
            "image and fixed camera path. Commands by one-second chunk: "
            + str(stock["fixed_actions"]) + f". RIGHT and LEFT yaw {stock.get('turn_degrees_per_second', 60)} degrees per second; W moves forward. "
            "Commands are NOT proof that motion happened. Inspect actual frames. "
            "Assess whether each video realizes a turnaround and return to the starting viewpoint; whether persistent "
            "landmark identity, any sign text, geometry and spatial layout match on return; and whether there are sudden "
            "jumps, frozen motion or implausible transitions. Distinguish camera-control failures from forgetting. "
            "Do not reward a similar final building if reaching it required a reset. Do not invent unseen geometry. "
            "Return English JSON with keys A and B (each with realized_round_trip: yes|no|uncertain, "
            "starting_place_recovered: yes|no|uncertain, discontinuity: yes|no|uncertain, visual_evidence, "
            "failure_classification), preference: A|B|tie|neither, confidence: low|medium|high, and reasoning. "
            "This is a diagnostic comparison, not a benchmark score."
        )
        content = [{"type": "text", "text": prompt}]
        # Fixed assignment before judging, unrelated to image quality or model outputs.
        assignment = f"{stock['task_id']}/{stock['seed']}" + (f"/{candidate_label}" if reference_label != "stock" else "")
        reverse = hashlib.sha256(assignment.encode()).digest()[0] % 2
        pair = [(candidate_label, args.candidate_video), (reference_label, args.stock_video)]
        if reverse:
            pair.reverse()
        mapping = dict(zip(("A", "B"), pair))
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
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "result": parse_json(raw)}
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"memory", "matched_settings"}}, indent=2))


if __name__ == "__main__":
    main()
