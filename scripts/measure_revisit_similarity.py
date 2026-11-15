"""Post-hoc return-segment similarity diagnostic; not a selection metric.

Report every sampled return frame, including the closest one, so heading
overshoot does not get mistaken for scene identity loss. Continuity, camera
motion and resets still require complete-video inspection.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-video", type=Path, required=True)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--return-start-chunk", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    from PIL import Image
    import torch
    from transformers import AutoImageProcessor, AutoModel
    torch.set_num_threads(8)
    name = "facebook/dinov2-small"
    revision = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    processor = AutoImageProcessor.from_pretrained(name, revision=revision, use_fast=False, local_files_only=True)
    model = AutoModel.from_pretrained(name, revision=revision, local_files_only=True).eval()
    report = {"scope": "Post-hoc supporting diagnostic; no benchmark score or replacement of frozen primary evaluation",
              "model": name, "revision": revision, "sample_fps": 4, "reference_frame": 12,
              "caveat": "High similarity can reward frozen frames or resets; full-video continuity and control evidence remain required."}
    for label, video in (("stock", args.stock_video), ("candidate", args.candidate_video)):
        capture = cv2.VideoCapture(str(video))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        start = args.return_start_chunk * 16 - 3
        indices = sorted(set(range(start, total, 4)) | {total - 1})
        images = []
        for index in [12] + indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Missing frame {index} in {video}")
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        capture.release()
        features = []
        with torch.no_grad():
            for offset in range(0, len(images), 8):
                inputs = processor(images=images[offset:offset + 8], return_tensors="pt")
                features.append(torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0], dim=-1))
        features = torch.cat(features)
        similarities = (features[1:] @ features[0]).tolist()
        best = max(range(len(indices)), key=lambda i: similarities[i])
        report[label] = {"peak_similarity": similarities[best], "peak_frame": indices[best],
                         "peak_time_s": indices[best] / fps, "endpoint_similarity": similarities[-1],
                         "samples": [{"frame": index, "time_s": index / fps, "cosine": score}
                                     for index, score in zip(indices, similarities)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "samples"} if isinstance(value, dict) else value
                      for key, value in report.items()}, indent=2))


if __name__ == "__main__":
    main()
