"""Stratified four-dimension (GC / IF / insight / out-of-sight) PlayWorld-style comparison helpers.

Cases are chosen by a salted hash of the case id, never by outcome. The overall figure follows the
paper: the equal mean of the four per-dimension means. Bootstrap intervals resample cases.
"""
import hashlib
import json
from pathlib import Path
import random

SPLITS = ("gc", "if", "insight", "outsight")
DIMENSION_NAMES = {"gc": "GC", "if": "IF", "insight": "insight", "outsight": "out-of-sight"}
SALT = "worldsync-fourdim-v1"
# Pre-declared exclusions (input defects found before any generation); the next case in hash order replaces them.
EXCLUDED = {("gc", "GC013"): "released image does not match caption/rubric (reproduction/input-audit-gc013.md)"}
PAPER = {  # LingBot-World2 hosted service, PlayWorld paper table (Genie 3 overall 2.12)
    "gc": 2.04, "if": 2.13, "insight": 1.95, "outsight": 1.16, "overall": 1.82, "genie3_overall": 2.12}


def hash_key(split, task_id, salt=SALT):
    return hashlib.sha256(f"{salt}:{split}:{task_id}".encode()).hexdigest()


def select_cases(data_root, per_dimension=12, salt=SALT):
    """First `per_dimension` task ids per split by salted hash order. Returns [(split, task_id)]."""
    chosen = []
    for split in SPLITS:
        records = json.loads((Path(data_root) / split / "data.json").read_text())
        ids = sorted((record["task_id"] for record in records if (split, record["task_id"]) not in EXCLUDED),
                     key=lambda task: hash_key(split, task, salt))
        chosen += [(split, task) for task in ids[:per_dimension]]
    return chosen


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def dimension_means(scores):
    """scores: {(split, task): value or None}. Returns {split: mean over scored cases} (None if empty)."""
    return {split: mean(v for (s, _), v in scores.items() if s == split and v is not None) for split in SPLITS}


def overall(dim_means):
    values = [dim_means[split] for split in SPLITS]
    return None if any(v is None for v in values) else sum(values) / len(values)


def bootstrap_overall(scores, n=5000, seed=0):
    """95% CI of the equal-mean-of-four overall, resampling cases within each dimension."""
    rng = random.Random(seed)
    by_split = {split: [v for (s, _), v in scores.items() if s == split and v is not None] for split in SPLITS}
    if any(not values for values in by_split.values()):
        return None
    stats = sorted(sum(mean(rng.choice(values) for _ in values) for values in by_split.values()) / 4 for _ in range(n))
    return [stats[int(0.025 * n)], stats[int(0.975 * n)]]


def bootstrap_paired_delta(scores_a, scores_b, n=5000, seed=0):
    """Paired overall delta (a - b): resample matched cases within each dimension."""
    rng = random.Random(seed)
    pairs = {split: [scores_a[k] - scores_b[k] for k in scores_a
                     if k[0] == split and k in scores_b and scores_a[k] is not None and scores_b[k] is not None]
             for split in SPLITS}
    if any(not values for values in pairs.values()):
        return None
    point = sum(mean(values) for values in pairs.values()) / 4
    stats = sorted(sum(mean(rng.choice(values) for _ in values) for values in pairs.values()) / 4 for _ in range(n))
    return {"delta": point, "ci95": [stats[int(0.025 * n)], stats[int(0.975 * n)]],
            "pairs": {split: len(values) for split, values in pairs.items()},
            "per_dimension": {split: mean(values) for split, values in pairs.items()}}
