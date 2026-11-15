"""Compare two re-judged runs case by case: python scripts/compare_runs.py A B (labels under outputs/rejudge)."""
import json
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"gc": ["GC024", "GC017", "GC007", "GC027"], "if": ["IF025", "IF042", "IF001"], "outsight": ["OE026", "OE057", "OE069"],
          "insight": ["OE007", "OE038", "OE035", "OE025"]}


def means(label):
    data = json.loads((ROOT / f"outputs/rejudge/{label}-gemini.json").read_text())
    out = {}
    for task, items in data["cases"].items():
        scores = [i["score"] for i in items if i["score"] is not None]
        gates = [i["gate"] for i in items]
        out[task] = (round(st.mean(scores), 2) if scores else None, len(scores), gates)
    return out


def main(a, b):
    A, B = means(a), means(b)
    common = [t for t in A if t in B and A[t][0] is not None and B[t][0] is not None]
    print(f"{'case':7} {a:>14} {b:>14} delta")
    for t in common:
        print(f"{t:7} {A[t][0]:>14} {B[t][0]:>14} {B[t][0] - A[t][0]:+.2f}   gates {a}:{A[t][2]} {b}:{B[t][2]}")
    for split, ids in SPLITS.items():
        ids = [t for t in ids if t in common]
        if ids:
            x, y = st.mean(A[t][0] for t in ids), st.mean(B[t][0] for t in ids)
            print(f"{split:9} {x:.2f} -> {y:.2f} ({y - x:+.2f}, n={len(ids)})")
    x, y = st.mean(A[t][0] for t in common), st.mean(B[t][0] for t in common)
    print(f"ALL {len(common)} cases: {x:.3f} -> {y:.3f} ({y - x:+.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
