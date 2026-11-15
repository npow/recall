"""Priority follow-up jobs for the storage ablation: low-rank reruns and seed-43 repeats of the 8-bit result."""
import json
from pathlib import Path

from memstore_jobs import job

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"stock": (None, "all"), "exact": ("exact", "all"), "fp8": ("fp8", "all"), "int8": ("int8", "all"),
            "lr64": ("lr64", "all"), "lr256": ("lr256", "all")}

if __name__ == "__main__":
    jobs = [job(name, *VARIANTS[name], "GC007", "turn_hold_return", 42, 0) for name in ("lr64", "lr256")]
    jobs += [job(name, *VARIANTS[name], "GC007", "turn_hold_return", 43, 1) for name in ("stock", "exact", "fp8", "int8")]
    jobs += [job(name, *VARIANTS[name], "GC001", "turn_hold_return", 43, 2) for name in ("stock", "exact", "fp8", "int8")]
    (ROOT / "reproduction/memory-storage-jobs-priority.jsonl").write_text("".join(json.dumps(item) + "\n" for item in jobs))
    print(len(jobs), "jobs", sum(item["chunks"] for item in jobs), "chunks")
