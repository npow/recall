"""Print per-episode player decisions, END reasons and commanded pose for run.json dirs."""
import glob
import json
import sys

pattern = sys.argv[1]
for d in sorted(glob.glob(pattern)):
    try:
        r = json.load(open(d + "/run.json"))
    except FileNotFoundError:
        continue
    print("==", d.rsplit("/", 1)[-1].split("-seed")[0], "term", r.get("termination"), "interactions", r.get("agent_interactions"))
    for a in r["actions"]:
        print("  ", a["operation"], a.get("action"), "|", (a.get("reason") or "")[:140].replace("\n", " "))
    last = r["actions"][-1]
    state = last.get("pose_after") or last.get("pose_before") or {}
    print("   final", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in state.items() if k != "scope"})
