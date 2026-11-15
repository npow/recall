"""Compose a shared-world demo: the agents' rollouts side by side, with each agent's memory facts per chunk.

usage: make_trio_demo.py RUN_ROOT [AGENT ...]   (RUN_ROOT/<agent>/{rollout.mp4,run.json}; DB at RUN_ROOT/../<root>.db)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worldsync import shared_world as sw
from worldsync.duel_demo import compose

T = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/shared-world/trio").resolve()
AGENTS = sys.argv[2:] or ["duel-a", "viewer", "duel-b"]
db = sw.FrameDB(T.parent / f"{T.name}.db")
FPS = 16
WID = sw.world_id(sw.world_manifest(sw.load_run(T / AGENTS[0])[0]))


def source_label(ids):
    """'agent-a's views 19-22' or 'own views 8-11' for a retrieved id list (own ints, foreign 'name:idx')."""
    shared = [i for i in ids if isinstance(i, str)]
    own = [i for i in ids if not isinstance(i, str)]
    if shared:
        frames = [int(i.split(":")[1]) for i in shared]
        return f"{shared[0].split(':')[0]}'s views {min(frames)}-{max(frames)}"
    return f"own views {min(own)}-{max(own)}"


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
            mem = f"RECALL: attending to {source_label(ids)}"
        elif shared:
            mem = f"RECALL: {len(shared)} other-agent view(s) match by pose, not applied"
        elif own:
            mem = "RECALL: own views match by pose, not applied"
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


LABELS = {"viewer": "viewer  (pans 0 -> 180 -> 0)", "duel-a": "duel-a  (faces 0, backs away)",
          "duel-b": "duel-b  (turns to 180, backs away)", "agent-a": "agent-a  (orbits the centre, full circle)",
          "agent-b": "agent-b  (walks to the far side, then orbits)"}
panels = [(T / agent / "rollout.mp4", LABELS.get(agent, agent), captions(agent, [o for o in AGENTS if o != agent]))
          for agent in AGENTS]
out = compose(panels, T / f"{T.name}-shared-world-demo.mp4", strip=96,
              title=f"{len(AGENTS)} agents, one LingBot world {WID}: seed 42, GC010 - shared RECALL archive + past.dev positions")
print(out)
