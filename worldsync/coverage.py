"""Coverage-first summaries: missing evaluations cannot improve a benchmark mean."""
from collections import Counter

from worldsync.repro_assets import SPLITS


def diagnostic_summary(cases, runs, expected_counts=None):
    expected_counts = expected_counts or SPLITS
    selected = [(case["split"], case["task_id"]) for case in cases]
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate evaluation case")
    selected_set = set(selected)
    scored = {}
    for run in runs:
        key = (run["split"], run["task_id"])
        if key not in selected_set:
            raise ValueError("unexpected evaluation case")
        if run["status"] != "completed" or run.get("judge_status") != "completed":
            continue
        value = run.get("diagnostic_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 5:
            raise ValueError("invalid diagnostic score")
        if key in scored:
            raise ValueError("multiple scored attempts for one case")
        scored[key] = value
    counts = Counter(split for split, _ in selected)
    dimensions = {}
    for split, expected in expected_counts.items():
        values = [value for (group, _), value in scored.items() if group == split]
        dimensions[split] = {"expected": expected, "selected": counts[split], "scored": len(values),
            "diagnostic_mean_of_available_cases": sum(values) / len(values) if values else None}
    complete = dict(counts) == expected_counts and len(scored) == len(selected)
    overall = sum(item["diagnostic_mean_of_available_cases"] for item in dimensions.values()) / len(dimensions) if complete else None
    return {"protocol": "Claude diagnostics; not published Gemini scores", "coverage_complete": complete,
        "dimensions": dimensions, "diagnostic_overall": overall,
        "published_score_reproduced": False}
