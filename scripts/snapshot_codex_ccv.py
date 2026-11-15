"""Publish collected CCV evidence, including failures, without selecting favorable repeats."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from run_codex_ccv import CASES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/codex-ccv/artifacts/codex-ccv'
DEST = ROOT/'reproduction/baseline-gap/ccv-snapshot'


def copy_immutable(source, destination):
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError('Previously published evidence changed: '+str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    DEST.mkdir(exist_ok=True)
    shutil.copy2(SOURCE/'selection.json', DEST/'selection.json')
    for split, tasks in CASES.items():
        for task in tasks:
            for arm in ['C', 'L']:
                identifier = f'{arm}-{split}-{task}'
                status = SOURCE/(identifier+'-status.json')
                if not status.exists():
                    continue
                state = json.loads(status.read_text())
                if type(state.get('returncode')) is not int:
                    continue  # Ownership reservations are not completed attempts.
                relative = Path(arm)/f'{split}-{task}'
                if arm == 'C':
                    relative /= 'episode'
                episode = SOURCE/relative
                video = episode/'rollout.mp4'
                digest = hashlib.sha256(video.read_bytes()).hexdigest() if video.exists() else None
                copy_immutable(status, DEST/status.relative_to(SOURCE))
                run = episode/'run.json'
                if run.exists():
                    copy_immutable(run, DEST/run.relative_to(SOURCE))
                for rep in ['rep1', 'rep2', 'rep3']:
                    report = SOURCE/'judge'/rep/'episodes'/(identifier+'.json')
                    if not report.exists():
                        continue
                    result = json.loads(report.read_text())
                    if result.get('video_sha256') and result['video_sha256'] != digest:
                        raise ValueError('Judge video does not match collected evidence: '+identifier)
                    if result.get('episode_id') != identifier:
                        raise ValueError('Judge episode identity mismatch: '+identifier)
                    copy_immutable(report, DEST/report.relative_to(SOURCE))
    subprocess.run([sys.executable, str(ROOT/'scripts/ccv_report.py'), '--run', str(DEST),
                    '--selection', str(DEST/'selection.json'), '--output', str(DEST/'report.md')], check=True)


if __name__ == '__main__':
    main()
