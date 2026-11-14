"""Run PlayWorld's unmodified judge, reading a local credential without shell sourcing.

Pass the official metrics.vqa.score arguments after this script's filename.
Requires a local PlayWorld checkout and the project's benchmark dependencies.
"""
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external/PlayWorld"
REVISION = "eeaea12e050a90b815d11d030e943e7b3181da21"
if not REPO.is_dir():
    raise SystemExit("Clone https://github.com/hku-sail/PlayWorld into external/PlayWorld first")
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
if revision != REVISION:
    raise SystemExit(f"Expected PlayWorld revision {REVISION}; found {revision}")
local_env = ROOT / ".env.local"
if local_env.is_file():
    for line in local_env.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            os.environ.setdefault("GEMINI_API_KEY", line.split("=", 1)[1].strip())
sys.path.insert(0, str(REPO))
runpy.run_module("metrics.vqa.score", run_name="__main__")
