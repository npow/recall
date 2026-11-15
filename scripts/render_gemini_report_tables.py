"""Render markdown tables (per-case pivot, per-dimension means) from gemini-results.json."""
import json
from pathlib import Path
import sys

data = json.loads(Path(sys.argv[1]).read_text())
episodes, report = data["episodes"], data["report"]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def cell(value):
    return "–" if value is None else f"{value:.2f}"


arms = ["stock", "cache46", "fb", "fbcomp"]
tasks = sorted({e["task"] for e in episodes if e["arm"] in arms})
print("| Case | " + " | ".join(f"{a} Gemini (Opus)" for a in arms) + " |")
print("| --- | " + " | ".join("---:" for _ in arms) + " |")
for task in tasks:
    cells = []
    for arm in arms:
        rows = [e for e in episodes if e["task"] == task and e["arm"] == arm]
        cells.append(f"{cell(mean([r['gemini'] for r in rows]))} ({cell(mean([r['opus'] for r in rows]))}) n={len(rows)}" if rows else "–")
    print(f"| {task} | " + " | ".join(cells) + " |")

print()
names = ["identity_id", "identity_texture", "color", "relative_position"]
print("GC dimension means (rubric called only when the gate did not fail)")
print("| Arm | n | " + " | ".join(names) + " |")
print("| --- | ---: | " + " | ".join("---:" for _ in names) + " |")
for arm in arms:
    table = report["category_means"].get(f"{arm}/gc", {})
    if table:
        n = next(iter(table.values()))["n"]
        print(f"| {arm} | {n} | " + " | ".join(cell(table.get(name, {}).get("mean")) for name in names) + " |")

print()
oe_names = sorted({n for key, t in report["category_means"].items() if key.endswith("/outsight") for n in t})
print("Out-of-sight OE dimension means")
print("| Arm | " + " | ".join(oe_names) + " |")
print("| --- | " + " | ".join("---:" for _ in oe_names) + " |")
for arm in arms:
    table = report["category_means"].get(f"{arm}/outsight", {})
    if table:
        print(f"| {arm} | " + " | ".join(f"{cell(table.get(n, {}).get('mean'))} (n={table.get(n, {}).get('n', 0)})" for n in oe_names) + " |")

print()
print("Gate verdict counts per arm (Gemini)")
for arm in arms + ["v3stock"]:
    rows = [e for e in episodes if e["arm"] == arm]
    counts = {}
    for r in rows:
        counts[r["gemini_gate"]] = counts.get(r["gemini_gate"], 0) + 1
    print(arm, dict(sorted(counts.items())))
