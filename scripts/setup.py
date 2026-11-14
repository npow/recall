"""Fetch the pinned upstream and apply the minimal opt-in memory patch."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external/ABot-World"
REVISION = "5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5"
PATCH = ROOT / "patches/abot-world-memory.patch"


def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)


def prepare(repo, url, revision, patch=None):
    if not repo.exists():
        run("git", "clone", "--depth", "1", "--filter=blob:none", url, str(repo))
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
            raise SystemExit(f"{repo.name} has local changes; refusing to change revision.")
        run("git", "-C", str(repo), "fetch", "origin", revision)
        run("git", "-C", str(repo), "checkout", "--detach", revision)
    if patch:
        applied = subprocess.run(["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch)], capture_output=True).returncode == 0
        if not applied:
            run("git", "-C", str(repo), "apply", "--check", str(patch))
            run("git", "-C", str(repo), "apply", str(patch))
    print(f"{repo.name} ready: {revision}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--with-playworld", action="store_true")
args = parser.parse_args()
prepare(REPO, "https://github.com/amap-cvlab/ABot-World.git", REVISION, PATCH)
if args.with_playworld:
    prepare(ROOT / "external/PlayWorld", "https://github.com/hku-sail/PlayWorld.git", "eeaea12e050a90b815d11d030e943e7b3181da21")
