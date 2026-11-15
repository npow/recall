"""Fetch rollout.mp4 and run.json of the memory-audit regression jobs from the Modal results volume.

Usage: MODAL_PROFILE=nissan-pow python scripts/audit_fetch_regression.py LOCAL_DIR
"""
import subprocess
import sys
from pathlib import Path

IDS = ["audit-budget-hold-a", "audit-stock-hold", "audit-budget-slow360", "audit-budget-hold-b-repeat",
       "audit-stock-slow360", "audit-budget-slow360-archive64"]


def main():
    local = Path(sys.argv[1])
    listing = subprocess.run(["modal", "volume", "ls", "worldsync-lingbot-results"], capture_output=True, text=True).stdout.split()
    for name in IDS:
        matches = sorted(entry for entry in listing if entry.startswith(name + "-") and entry[len(name) + 1:].isdigit())
        if not matches:
            print("missing", name)
            continue
        target = local / name
        target.mkdir(parents=True, exist_ok=True)
        for file in ("rollout.mp4", "run.json"):
            if not (target / file).exists():
                subprocess.run(["modal", "volume", "get", "worldsync-lingbot-results", f"{matches[-1]}/{file}", str(target / file)], check=True,
                               capture_output=True)
        print("ok", name, matches[-1])


if __name__ == "__main__":
    main()
