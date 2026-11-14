"""Measure return-view appearance and build a contact sheet. NOT PlayWorld."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from PIL import Image, ImageDraw
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from huggingface_hub import try_to_load_from_cache

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    rows, results = [], []
    for result_path in sorted(args.directory.glob("seed-*/*/result.json")):
        record = json.loads(result_path.read_text())
        if record["status"] != "completed" or not record["complete_action_sequence"]:
            continue
        directory = result_path.parent
        # Compare the first generated block's endpoint to the last endpoint.
        # Both have been rendered by the model. Also score the original input
        # separately if needed; it is never substituted for a generated return.
        reference, returned = Image.open(directory / "block-000.png").convert("RGB"), Image.open(directory / "final.png").convert("RGB")
        inputs = processor(images=[reference, returned], return_tensors="pt").to(args.device)
        with torch.no_grad():
            features = model(**inputs).last_hidden_state[:, 0].float()
            similarity = torch.nn.functional.cosine_similarity(features[:1], features[1:]).item()
        times = [b["seconds"] for b in record["blocks"]]
        results.append({"mode": record["mode"], "seed": record["seed"],
                        "dino_cosine": similarity, "seconds_per_block": sum(times) / len(times),
                        "steady_seconds_per_block": sum(times[1:]) / len(times[1:]) if len(times) > 1 else None,
                        "peak_allocated_bytes": record["peak_allocated_bytes"],
                        "trajectory_validity": "unverified; inspect video", "run": str(directory)})
        indices = sorted(set([0, len(record["blocks"]) // 2, len(record["blocks"]) - 1]))
        row = Image.new("RGB", (3 * 320, 210), "white")
        draw = ImageDraw.Draw(row)
        draw.text((8, 5), f"{record['mode']} seed={record['seed']} DINO={similarity:.4f} (trajectory unverified)", fill="black")
        for column, index in enumerate(indices):
            frame = Image.open(directory / f"block-{index:03d}.png").convert("RGB")
            frame.thumbnail((320, 180))
            row.paste(frame, (column * 320, 28))
        rows.append(row)
    if not results:
        raise SystemExit("No completed full-length runs to score")
    sheet = Image.new("RGB", (960, len(rows) * 210), "white")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * 210))
    sheet.save(args.directory / "comparison.jpg")
    cached_config = try_to_load_from_cache(args.model, "config.json")
    revision = Path(cached_config).parent.name if isinstance(cached_config, str) else None
    report = {"metric": "DINOv2 CLS cosine, first generated block endpoint vs final endpoint",
              "model": args.model, "model_revision": revision,
              "warning": "Appearance proxy only; identical views require trajectory validation. Not a PlayWorld score.",
              "runs": results}
    (args.directory / "diagnostic_scores.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
