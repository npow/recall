"""Aggregate LoRA-eval runs: per (task, path, seed) DINO table plus side-by-side sheets.

  python scripts/lora_eval_table.py --runs outputs/lora-eval --output reproduction/lora

Directory names: lora-eval-<TASK>-<arm>-<path>-seed<N>-<ns>. Diagnostic only; not a PlayWorld score.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import subprocess
import sys

PATTERN = re.compile(r"lora-eval-(?P<task>GC\d+)-(?P<arm>[a-z0-9_]+)-(?P<path>[a-z_0-9]+)-seed(?P<seed>\d+)-\d+")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = collections.defaultdict(dict)
    for directory in sorted(args.runs.iterdir()):
        match = PATTERN.fullmatch(directory.name)
        if match and (directory / "run.json").exists():
            groups[(match["task"], match["path"], int(match["seed"]))][match["arm"]] = directory
    table = {}
    for (task, path, seed), arms in sorted(groups.items()):
        out = args.output / f"{task}-{path}-seed{seed}"
        runs = [f"{arm}={directory}" for arm, directory in sorted(arms.items())]
        subprocess.run([sys.executable, str(Path(__file__).parent / "revisit_report.py"), "--runs", *runs, "--output", str(out)],
                       check=True, stdout=subprocess.DEVNULL)
        report = json.loads((out / "report.json").read_text())
        for arm, values in report.items():
            table.setdefault(f"{task}/{path}", {}).setdefault(arm, {})[seed] = {
                "end_vs_f12": values["dino_end_vs_frame12"], "end_vs_f0": values["dino_end_vs_frame0"],
                "peak_gb": values["peak_allocated_gb"], "elapsed_s": values["elapsed_s"]}
    (args.output / "table.json").write_text(json.dumps(table, indent=2) + "\n")
    lines = ["| scene/path | arm | DINO end-vs-f12 (per seed) | mean | peak GB | s |", "| --- | --- | --- | --- | --- | --- |"]
    for key, arms in table.items():
        for arm, seeds in sorted(arms.items()):
            values = [seeds[s]["end_vs_f12"] for s in sorted(seeds)]
            peak = max(seeds[s]["peak_gb"] for s in seeds)
            sec = sum(seeds[s]["elapsed_s"] for s in seeds) / len(seeds)
            lines.append(f"| {key} | {arm} | {', '.join(f'{v:.3f}' for v in values)} | {sum(values) / len(values):.3f} | {peak:.0f} | {sec:.0f} |")
    (args.output / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
