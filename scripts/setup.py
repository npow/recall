"""Fetch the pinned upstream and apply the minimal opt-in memory patch."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external/ABot-World"
REVISION = "5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5"
PATCH = ROOT / "patches/abot-world-memory.patch"


def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)


if not REPO.exists():
    run("git", "clone", "--filter=blob:none", "https://github.com/amap-cvlab/ABot-World.git", str(REPO))
head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
if head != REVISION:
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Upstream checkout has local changes; refusing to change revision.")
    run("git", "-C", str(REPO), "fetch", "origin", REVISION)
    run("git", "-C", str(REPO), "checkout", "--detach", REVISION)
already_applied = subprocess.run(["git", "-C", str(REPO), "apply", "--reverse", "--check", str(PATCH)], capture_output=True).returncode == 0
if not already_applied:
    run("git", "-C", str(REPO), "apply", "--check", str(PATCH))
    run("git", "-C", str(REPO), "apply", str(PATCH))
print(f"ABot ready: {REVISION}; memory patch applied")
