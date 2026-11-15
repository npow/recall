"""Judge-independent mechanism check on the full run: when the commanded camera
returns to the start pose after an excursion, how close is the returned view to
the start frame (DINOv2-small CLS cosine), stock vs budget memory? (read-only)

Usage: python scripts/audit_return_dino.py EPISODES_DIR OUT_JSON [--strips DIR]
"""
import collections
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldsync.lingbot_budgetmem import pose_cost  # noqa: E402
from worldsync.lingbot_controls import CameraPlan  # noqa: E402

RETURN_COST = 0.5      # within 2 units / 30 degrees of the start pose
EXCURSION_COST = 1.0   # must have been at least this far away
MIN_GAP_CHUNKS = 4     # return chunk is beyond the 12-frame stock recent window from the last near-start chunk


def return_chunk(actions):
    camera = CameraPlan()
    costs = []
    for action in actions:
        camera.advance(action)
        costs.append(pose_cost(np.eye(4), camera.poses[-1]))
    away = [i for i, c in enumerate(costs) if c >= EXCURSION_COST]
    if not away:
        return None, costs
    candidates = [i for i, c in enumerate(costs) if c <= RETURN_COST and i > away[0] and i - away[0] >= 1]
    # absence length: chunks spent away (cost > RETURN_COST) immediately before the return
    best = None
    for i in candidates:
        j = i - 1
        while j >= 0 and costs[j] > RETURN_COST:
            j -= 1
        absence = i - j - 1
        if absence >= MIN_GAP_CHUNKS - 1:
            best = (i, absence)
    return best, costs


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    strips = Path(sys.argv[sys.argv.index("--strips") + 1]) if "--strips" in sys.argv else None
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def embed(frame):
        with torch.no_grad():
            feature = model(**processor(images=Image.fromarray(frame), return_tensors="pt")).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(feature, dim=-1)[0]

    rows = []
    for directory in sorted(root.glob("*-seed42")):
        arm, split, task = directory.name.split("-")[:3]
        if split not in ("gc", "outsight"):
            continue
        run = json.loads((directory / "run.json").read_text())
        actions = [a["action"] for a in run["actions"] if a.get("operation") != "END" and "pose_after" in a]
        actions = actions[:len(run["chunks"])]
        found, costs = return_chunk(actions)
        if not found:
            continue
        chunk, absence = found
        video = directory / "rollout.mp4"
        if not video.exists():
            continue
        frames = np.stack(iio.mimread(video, memtest=False))
        index = min(13 + 16 * chunk - 1, len(frames) - 1)
        far = int(np.argmax(costs[:chunk]))
        far_index = min(13 + 16 * far - 1, len(frames) - 1)
        start = embed(frames[0])
        rows.append({"arm": arm, "split": split, "task": task, "return_chunk": chunk, "absence_chunks": absence,
                     "dino_return_vs_start": float(start @ embed(frames[index])),
                     "dino_far_vs_start": float(start @ embed(frames[far_index])),
                     "retrieval_calls": (run.get("memory") or {}).get("retrieval_attention_calls")})
        if strips is not None:
            strips.mkdir(parents=True, exist_ok=True)
            tiles = [Image.fromarray(frames[i]).resize((320, 184)) for i in (0, far_index, index)]
            sheet = Image.new("RGB", (960, 184))
            for position, tile in enumerate(tiles):
                sheet.paste(tile, (position * 320, 0))
            sheet.save(strips / f"{split}-{task}-{arm}.jpg", quality=85)
    by_case = collections.defaultdict(dict)
    for row in rows:
        by_case[(row["split"], row["task"])][row["arm"]] = row
    paired = [(k, v["stock"], v["budget"]) for k, v in sorted(by_case.items()) if "stock" in v and "budget" in v]
    summary = {}
    for name, select in (("all", lambda s, b: True), ("absence>=6_both", lambda s, b: min(s["absence_chunks"], b["absence_chunks"]) >= 6),
                         ("gc", lambda s, b: s["split"] == "gc"), ("outsight", lambda s, b: s["split"] == "outsight")):
        chosen = [(s, b) for _, s, b in paired if select(s, b)]
        if not chosen:
            continue
        stock = np.array([s["dino_return_vs_start"] for s, _ in chosen])
        budget = np.array([b["dino_return_vs_start"] for _, b in chosen])
        rng = np.random.default_rng(0)
        delta = budget - stock
        boots = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(5000)]
        summary[name] = {"pairs": len(chosen), "stock_mean": float(stock.mean()), "budget_mean": float(budget.mean()),
                         "delta": float(delta.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         "wins": int((delta > 0.05).sum()), "losses": int((delta < -0.05).sum())}
    unpaired = {arm: {"n": sum(r["arm"] == arm for r in rows),
                      "mean": float(np.mean([r["dino_return_vs_start"] for r in rows if r["arm"] == arm]))} for arm in ("stock", "budget")}
    out.write_text(json.dumps({"definition": {"return_cost": RETURN_COST, "excursion_cost": EXCURSION_COST,
        "min_absence_chunks": MIN_GAP_CHUNKS - 1, "note": "arms ran different adaptive player trajectories; pairs are by case, not by path"},
        "paired_summary": summary, "unpaired": unpaired, "rows": rows}, indent=1) + "\n")
    print(json.dumps({"paired": summary, "unpaired": unpaired}, indent=1))


if __name__ == "__main__":
    main()
