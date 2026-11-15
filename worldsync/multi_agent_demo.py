"""Two scripted agents in separate processes, sharing a transactional world.

Run: python -m worldsync.multi_agent_demo --output outputs/multi-agent
Pillow is needed only for the visual artifact. past.dev is optional and advisory.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import uuid

from worldsync.shared_world import INITIAL, World, render_context


def worker(path, actor, inbox, outbox):
    world = World(path)
    observed = world.read()
    outbox.put({"state": observed, "pid": os.getpid()})
    while True:
        request = inbox.get()
        if request is None:
            return
        if request["action"] == "observe":
            observed = world.read()
            outbox.put({"state": observed})
        else:
            reply = world.submit(actor, request["action"], observed["revision"],
                                 request["token"], **request.get("arguments", {}))
            if reply["accepted"]:
                observed = reply["state"]
            outbox.put(reply)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    database = output / "world.sqlite"
    if database.exists():
        raise ValueError("world.sqlite already exists; use a new output directory")
    world = World(database)
    world.initialize()
    ctx = mp.get_context("spawn")
    workers, beliefs, pids, frames = {}, {}, {}, []

    def record(caption, reply=None):
        state = world.read()
        frames.append({"caption": caption, "state": state, "beliefs": copy.deepcopy(beliefs),
                       "reply": reply,
                       "contexts": {a: render_context(state, a) for a in INITIAL["agents"]}})

    def step(actor, action, caption, **arguments):
        inbox, outbox, _ = workers[actor]
        inbox.put({"action": action, "arguments": arguments, "token": f"step-{len(frames)}"})
        reply = outbox.get(timeout=20)
        if reply.get("accepted") is not False:
            beliefs[actor] = reply["state"]
        record(caption, reply)

    try:
        for actor in INITIAL["agents"]:
            inbox, outbox = ctx.Queue(), ctx.Queue()
            process = ctx.Process(target=worker, args=(str(database), actor, inbox, outbox))
            process.start()
            workers[actor] = inbox, outbox, process
            hello = outbox.get(timeout=20)
            beliefs[actor], pids[actor] = hello["state"], hello["pid"]
        record("Both agents see the red parcel at (2, 2).")
        step("alice", "move", "Alice approaches the parcel. Bob keeps his old observation.", target=[2, 2])
        step("alice", "pickup", "Alice picks up the only red parcel.")
        step("alice", "move", "Alice carries it toward the meeting point.", target=[3, 2])
        step("alice", "move", "Alice arrives at the meeting point.", target=[4, 2])
        step("alice", "drop", "Alice leaves the parcel at (4, 2), out of Bob's local memory.")
        step("bob", "pickup", "Bob acts on revision 0. REJECTED: the world is now revision 5.")
        step("bob", "observe", "Bob refreshes: the parcel moved to (4, 2). No duplicate is created.")
        for x in [6, 5, 4]:
            step("bob", "move", "Bob walks to the parcel's committed location.", target=[x, 2])
        step("bob", "pickup", "Bob picks up the same parcel Alice left.")
        step("bob", "handoff", "Bob hands the parcel back to Alice. Ownership changes atomically.", recipient="alice")
        step("alice", "observe", "Alice refreshes and confirms she holds the parcel.")
        step("bob", "observe", "Both agents agree on revision 10: Alice holds the only parcel.")
    finally:
        for inbox, _, _ in workers.values():
            inbox.put(None)
        for _, _, process in workers.values():
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
    final = world.read()
    assert all(b == final for b in beliefs.values())
    events = world.history()
    report = {"kind": "scripted coordination prototype; not LingBot video or LLM agents",
              "agent_pids": pids, "final_state": final,
              "accepted": sum(e["accepted"] for e in events),
              "rejected": sum(not e["accepted"] for e in events),
              "agents_converged": all(b == final for b in beliefs.values()), "frames": frames}
    (output / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    write_past_outbox(events, output)
    render_gif(frames, output / "demo.gif")
    return report


def write_past_outbox(events, output):
    """Immutable committed events only; retries share the same ingestion key."""
    from worldsync.research_memory import build_finding
    run_id = uuid.uuid4().hex
    findings = []
    for event in events:
        if not event["accepted"]:
            continue
        state = event["state"]
        text = (f"Courtyard run {run_id}, committed world revision {state['revision']}. "
                f"{event['actor']} performed {event['action']}. "
                + render_context(state, event["actor"])["scene_text"])
        finding = build_finding(text, event["occurred_at"], f"multi-agent:{run_id}",
                                tags=("shared-world", run_id), section=f"revision-{state['revision']}")
        finding["identity"] = "worldsync-shared-world"
        finding["metadata"].update(run_id=run_id, revision=state["revision"])
        findings.append(finding)
    (output / "past-outbox.jsonl").write_text("".join(json.dumps(f) + "\n" for f in findings))


def publish_past(output, client=None):
    """Explicit opt-in: publish synthetic committed events using the existing client.

    Receipts may be queued. Recall is deliberately separate from the control loop.
    """
    from worldsync.research_memory import PastClient
    client = client or PastClient()
    output = Path(output)
    with (output / "past-receipts.jsonl").open("a") as receipts:
        for line in (output / "past-outbox.jsonl").read_text().splitlines():
            finding = json.loads(line)
            receipt = client.ingest(finding)
            receipts.write(json.dumps({"id": finding["id"], "receipt": receipt}) + "\n")
            receipts.flush()


def render_gif(frames, path):
    import textwrap
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    images = []
    colors = {"alice": "#64b5ff", "bob": "#ffca65"}
    for index, frame in enumerate(frames):
        canvas = Image.new("RGB", (1080, 660), "#101724")
        draw = ImageDraw.Draw(canvas)
        state = frame["state"]
        draw.text((30, 20), "RECALL / TWO AGENTS, ONE WORLD", font=font(27), fill="white")
        draw.text((30, 61), "Scripted IPC prototype | separate processes | authoritative SQLite state", font=font(17), fill="#a9b8ca")
        draw.text((30, 104), f"COMMITTED WORLD   revision {state['revision']}", font=font(20), fill="white")
        cell, left, top = 64, 30, 153
        for x in range(9):
            for y in range(5):
                draw.rectangle((left+x*cell, top+y*cell, left+(x+1)*cell-3, top+(y+1)*cell-3), fill="#223146")
        for actor, position in state["agents"].items():
            x, y = position
            offset = -13 if actor == "alice" else 13
            cx, cy = left+x*cell+31+offset, top+y*cell+31
            draw.ellipse((cx-13, cy-13, cx+13, cy+13), fill=colors[actor])
            draw.text((cx-6, cy-10), actor[0].upper(), font=font(15), fill="#101724")
        parcel = state["parcel"]
        pos = state["agents"][parcel["holder"]] if parcel["holder"] else parcel["position"]
        px, py = left+pos[0]*cell+25, top+pos[1]*cell+5
        draw.rectangle((px, py, px+13, py+13), fill="#ff596c")
        draw.text((30, 490), "A = Alice    B = Bob    red square = the one parcel", font=font(17), fill="#a9b8ca")
        for i, actor in enumerate(state["agents"]):
            belief = frame["beliefs"][actor]
            y = 145+i*165
            draw.text((645, y), actor.upper()+" / LOCAL OBSERVATION", font=font(18), fill=colors[actor])
            fresh = belief["revision"] == state["revision"]
            draw.text((645, y+35), f"revision {belief['revision']}  /  "+("current" if fresh else "STALE"), font=font(20), fill="white" if fresh else "#ff8794")
            remembered = belief["parcel"]
            where = "held by "+remembered["holder"] if remembered["holder"] else str(remembered["position"])
            draw.text((645, y+70), "Parcel: "+where, font=font(19), fill="#c8d5e5")
        reply = frame["reply"] or {}
        if reply.get("accepted") is False:
            draw.text((645, 475), "STALE WRITE REJECTED", font=font(21), fill="#ff8794")
        else:
            draw.text((645, 475), "ONE PARCEL / ONE OWNER", font=font(20), fill="#81ddb5")
        for i, line in enumerate(textwrap.wrap(frame["caption"], width=88)):
            draw.text((30, 555+i*29), line, font=font(21), fill="white")
        draw.text((30, 631), f"Step {index+1}/{len(frames)}   |   past.dev outbox: committed events only; optional durable recall", font=font(15), fill="#a9b8ca")
        images.append(canvas)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=[1600]*len(images[:-1])+[3500], loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/multi-agent"))
    parser.add_argument("--publish-past", action="store_true", help="send synthetic committed events to past.dev (PAST_API_KEY)")
    parser.add_argument("--publish-only", action="store_true", help="retry an existing outbox; requires --publish-past")
    args = parser.parse_args()
    if args.publish_only and not args.publish_past:
        parser.error("--publish-only requires --publish-past")
    if not args.publish_only:
        report = run(args.output)
        print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
        print(f"Visual: {args.output / 'demo.gif'}")
    if args.publish_past:
        publish_past(args.output)


if __name__ == "__main__":
    main()
