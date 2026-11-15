"""Compose the three-agent demo: duel-a | viewer | duel-b, with each agent's shared-memory facts per chunk."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worldsync import shared_world as sw
from worldsync.duel_demo import compose

# Run dirs (rollout.mp4 + run.json per agent, from scripts/modal_lingbot_trio.py) and the SQLite DB built by
# `python -m worldsync.shared_world --db outputs/shared-world/trio.db ingest-run outputs/shared-world/trio/<agent> ...`.
T = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/shared-world/trio").resolve()
db = sw.FrameDB(T.parent / "trio.db")
WID = "w-ea4cc9b26e3af54b"
FPS = 16


def captions(agent, others):
    run, _ = sw.load_run(T / agent)
    sightings = sw.run_sightings(run)
    trace = {row["chunk"]: row for row in run["memory"]["trace"]}
    windows, start = [], 0
    for chunk, s in zip(run["chunks"], sightings):
        end = start + chunk["frames"]
        row = trace.get(s["chunk"], {})
        ids = row.get("retrieved_frame_ids", [])
        shared = [i for i in ids if isinstance(i, str)]
        own = [i for i in ids if not isinstance(i, str)]
        if row.get("attention_replaced"):
            src = (f"viewer's views {min(int(i.split(':')[1]) for i in shared)}-{max(int(i.split(':')[1]) for i in shared)}"
                   if shared else f"own views {min(own)}-{max(own)}")
            mem = f"RECALL: attending to {src}"
        elif shared:
            mem = f"RECALL: {len(shared)} viewer view(s) match by pose, not applied (needs 4)"
        else:
            mem = "RECALL: no stored view within range"
        where = f"{s['action']}  heading {s['heading']:.0f}  z {s['position'][2]:+.0f}  t {s['world_time']:.1f}s"
        facts = []
        for other in others:
            try:
                cue = sw.opponent_cue(db, WID, agent, other, at=s["world_time"], forward_axis=2)
            except LookupError:
                continue
            rel = "ahead" if abs(cue["relative"]) <= 30 else "behind" if abs(cue["relative"]) >= 150 else "to the side"
            facts.append(f"{other} {cue['distance']:.0f} u {rel}")
        text = f"{where}\n{mem}\npast.dev: " + (", ".join(facts) if facts else "no other agent yet")
        windows.append((start / FPS, end / FPS, text))
        start = end
    return windows


panels = [(T / "duel-a/rollout.mp4", "duel-a  (faces 0, backs away)", captions("duel-a", ["viewer", "duel-b"])),
          (T / "viewer/rollout.mp4", "viewer  (pans 0 -> 180 -> 0)", captions("viewer", ["duel-a", "duel-b"])),
          (T / "duel-b/rollout.mp4", "duel-b  (turns to 180, backs away)", captions("duel-b", ["viewer", "duel-a"]))]
out = compose(panels, T / "trio-shared-world-demo.mp4", strip=96,
              title=f"Three agents, one LingBot world {WID}: seed 42, GC010 - shared RECALL archive + past.dev positions")
print(out)
