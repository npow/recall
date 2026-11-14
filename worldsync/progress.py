"""Publish a bounded experiment progress file without staging unrelated changes."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def publish(progress, filename):
    if filename not in {"lingbot-progress.json", "matrix-progress.json"}:
        raise ValueError("unsupported progress destination")
    path = ROOT / "reproduction" / filename
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    temporary.replace(path)
    relative = str(path.relative_to(ROOT))
    subprocess.run(["git", "add", "--", relative], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "--only", "-m", "Update world model reproduction progress", "--", relative], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT, check=True)
