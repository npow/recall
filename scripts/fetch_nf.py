"""Download finished nf-* result directories from the results volume (skips ones already local).

    MODAL_PROFILE=nissan-pow python3 scripts/fetch_nf.py outputs/nf
"""
import json
from pathlib import Path
import subprocess
import sys

dest = Path(sys.argv[1])
dest.mkdir(parents=True, exist_ok=True)
out = subprocess.run(["modal", "volume", "ls", "worldsync-lingbot-results", "/", "--json"],
                     capture_output=True, text=True).stdout
names = sorted({(e.get("Filename") or e.get("filename") or "").strip("/") for e in json.loads(out)})
fetched = 0
for name in names:
    if not name.startswith("nf-") or (dest / name / "run.json").exists():
        continue
    result = subprocess.run(["modal", "volume", "get", "worldsync-lingbot-results", name, str(dest)],
                            capture_output=True, text=True)
    fetched += result.returncode == 0
    if result.returncode:
        print("failed", name, result.stderr[-200:])
print(f"{fetched} new directories; {len([d for d in dest.iterdir() if (d / 'run.json').exists()])} total")
