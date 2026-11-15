"""Print a compact summary of one Gemini-scored episode JSON (no secrets involved)."""
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
print("keys:", list(result.keys()))
print("scores:", result.get("scores"), "calls:", result.get("calls"))
gate = result.get("gate") or {}
print("gate:", {k: gate.get(k) for k in ("verdict", "instruction_following_score", "meaningful_requested_trajectory_progress", "revisit_required", "revisit_observed")})
rubric = result.get("rubric") or {}
print("rubric keys:", list(rubric.keys())[:14])
categories = rubric.get("categories") or rubric.get("category_scores_1_to_5") or {}
for name, value in categories.items():
    print(" ", name, {k: value.get(k) for k in ("score", "weight")} if isinstance(value, dict) else value)
print("final:", rubric.get("final_score_1_to_5"), rubric.get("score"))
for key in rubric:
    if "summary" in key.lower() or "evidence" in key.lower() or "rationale" in key.lower():
        print(key, ":", str(rubric[key])[:300])
