"""Shared world memory for several agents, backed by a local frame database and past.dev.

Two agents inhabit the same generated world when they start from the same initial image, caption, seed, model
revision, frame rate and chunk size (the world id hashes these). Each agent writes what it saw as one dated
observation per chunk:

* the generated frame itself goes into a local SQLite database (JPEG blob, addressed by its sha256); no frame is
  written to the filesystem as a file;
* past.dev receives text only: the agent, world time, commanded heading/position, an optional caption and a
  ``frame:sha256:<hex>`` reference (plus an optional small base64 thumbnail in metadata).

Another agent recalls what is known about a heading as of its own world time and resolves the frame references
against the local database. Runs with identical pixel hashes are recorded as sharing the world up to the first
differing chunk.

CLI::

    python -m worldsync.shared_world ingest-run reproduction/full-history-GC010 --agent agent-a --dry-run
    python -m worldsync.shared_world ingest-run <run-dir> --agent agent-a --confirm-send
    python -m worldsync.shared_world compare <run-dir-a> <run-dir-b> --agents agent-a agent-b [--confirm-send]
    python -m worldsync.shared_world recall <world-id> --heading 90 --identity agent-b [--at 12.75]
    python -m worldsync.shared_world nearby <world-id> --heading 90 [--exclude agent-b]
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import io
import json
import math
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

from worldsync.research_memory import PastClient, PastError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "outputs" / "shared-world" / "world.db"
WORLD_EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
FRAME_REF = re.compile(r"frame:sha256:([0-9a-f]{64})")

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    sha256 TEXT PRIMARY KEY, jpeg BLOB NOT NULL, width INTEGER, height INTEGER);
CREATE TABLE IF NOT EXISTS worlds (
    world_id TEXT PRIMARY KEY, manifest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sightings (
    world_id TEXT NOT NULL, agent TEXT NOT NULL, chunk INTEGER NOT NULL, frame_index INTEGER NOT NULL,
    world_time REAL NOT NULL, heading REAL NOT NULL, position TEXT NOT NULL, frame_sha256 TEXT NOT NULL,
    chunk_pixel_sha256 TEXT, caption TEXT, PRIMARY KEY (world_id, agent, chunk));
CREATE TABLE IF NOT EXISTS outbox (
    id TEXT PRIMARY KEY, body TEXT NOT NULL, sha256 TEXT NOT NULL, ingestion_id TEXT, status TEXT);
"""


def world_manifest(run):
    """The fields that make two rollouts the same world, taken from a rollout's run.json."""
    keys = ("model", "source_revision", "checkpoint_revision", "task_id", "initial_image_sha256",
            "image_caption_sha256", "seed", "fps", "chunk_size")
    manifest = {key: run.get(key) for key in keys}
    # A run whose conditioning text was changed (world_text_sha256) is a different world from the plain caption.
    manifest["world_text_sha256"] = run.get("world_text_sha256") or run.get("image_caption_sha256")
    return manifest


def world_id(manifest):
    identity = {key: manifest.get(key) for key in ("source_revision", "checkpoint_revision", "initial_image_sha256",
                                                   "image_caption_sha256", "seed", "fps", "chunk_size")}
    identity["world_text_sha256"] = manifest.get("world_text_sha256") or manifest.get("image_caption_sha256")
    missing = sorted(key for key, value in identity.items() if value in (None, ""))
    if missing:
        raise ValueError(f"run is missing world identity fields: {missing}")
    return "w-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def heading_gap(a, b):
    return abs((a - b + 180) % 360 - 180)


def world_timestamp(seconds):
    return (WORLD_EPOCH + dt.timedelta(seconds=float(seconds))).isoformat().replace("+00:00", "Z")


class FrameDB:
    """Local SQLite store: frame blobs by sha256, per-agent sightings, and the past.dev outbox."""

    def __init__(self, path=DEFAULT_DB):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def put_frame(self, jpeg, width=None, height=None):
        digest = hashlib.sha256(jpeg).hexdigest()
        self.db.execute("INSERT OR IGNORE INTO frames VALUES (?, ?, ?, ?)", (digest, jpeg, width, height))
        self.db.commit()
        return digest

    def get_frame(self, digest):
        row = self.db.execute("SELECT jpeg FROM frames WHERE sha256 = ?", (digest,)).fetchone()
        return row[0] if row else None

    def put_world(self, wid, manifest):
        self.db.execute("INSERT OR REPLACE INTO worlds VALUES (?, ?)", (wid, json.dumps(manifest, sort_keys=True)))
        self.db.commit()

    def add_sighting(self, wid, agent, sighting):
        self.db.execute("INSERT OR REPLACE INTO sightings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            wid, agent, sighting["chunk"], sighting["frame_index"], sighting["world_time"], sighting["heading"],
            json.dumps(sighting["position"]), sighting["frame_sha256"], sighting.get("chunk_pixel_sha256"),
            sighting.get("caption")))
        self.db.commit()

    def sightings(self, wid, agent=None):
        query = "SELECT agent, chunk, frame_index, world_time, heading, position, frame_sha256, chunk_pixel_sha256, caption " \
                "FROM sightings WHERE world_id = ?" + (" AND agent = ?" if agent else "") + " ORDER BY agent, chunk"
        rows = self.db.execute(query, (wid, agent) if agent else (wid,)).fetchall()
        names = ("agent", "chunk", "frame_index", "world_time", "heading", "position", "frame_sha256",
                 "chunk_pixel_sha256", "caption")
        return [{**dict(zip(names, row)), "position": json.loads(row[5])} for row in rows]

    def nearby(self, wid, heading, tolerance=30, exclude_agent=None, before=None):
        """Sightings within ``tolerance`` degrees of ``heading``, closest first (exact local lookup)."""
        found = [s for s in self.sightings(wid) if heading_gap(s["heading"], heading) <= tolerance
                 and s["agent"] != exclude_agent and (before is None or s["world_time"] <= before)]
        return sorted(found, key=lambda s: (heading_gap(s["heading"], heading), -s["world_time"]))

    def queue(self, point):
        digest = hashlib.sha256(json.dumps(point, sort_keys=True).encode()).hexdigest()
        known = self.db.execute("SELECT sha256 FROM outbox WHERE id = ?", (point["id"],)).fetchone()
        if known and known[0] == digest:
            return False
        self.db.execute("INSERT OR REPLACE INTO outbox (id, body, sha256) VALUES (?, ?, ?)",
                        (point["id"], json.dumps(point, sort_keys=True), digest))
        self.db.commit()
        return True

    def pending(self):
        rows = self.db.execute("SELECT body FROM outbox WHERE ingestion_id IS NULL ORDER BY id").fetchall()
        return [json.loads(row[0]) for row in rows]

    def mark_sent(self, point_id, ingestion_id, status):
        self.db.execute("UPDATE outbox SET ingestion_id = ?, status = ? WHERE id = ?", (ingestion_id, status, point_id))
        self.db.commit()


def frames_from_video(path, indices, quality=90):
    """Decode the requested frame indices of a video to JPEG bytes in memory (no files are written)."""
    from PIL import Image
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    width, height = (int(v) for v in probe.strip().split(",")[:2])
    wanted = sorted(set(indices))
    expr = "+".join(f"eq(n\\,{i})" for i in wanted)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select={expr}", "-vsync", "0",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    size = width * height * 3
    if len(raw) != size * len(wanted):
        raise ValueError(f"decoded {len(raw) // size} of {len(wanted)} requested frames from {path}")
    frames = {}
    for slot, index in enumerate(wanted):
        buffer = io.BytesIO()
        Image.frombytes("RGB", (width, height), raw[slot * size:(slot + 1) * size]).save(buffer, "JPEG", quality=quality)
        frames[index] = (buffer.getvalue(), width, height)
    return frames


def thumbnail_b64(jpeg, width=160, quality=70):
    from PIL import Image
    image = Image.open(io.BytesIO(jpeg))
    image.thumbnail((width, width * image.height // image.width))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def run_sightings(run):
    """One sighting per chunk: its last frame, with the commanded pose after that chunk's action."""
    chunks, actions = run["chunks"], run["actions"]
    if len(actions) < len(chunks):
        raise ValueError("run has fewer recorded actions than chunks")
    sightings, end = [], -1
    for chunk, action in zip(chunks, actions):
        end += chunk["frames"]
        pose = action["pose_after"]
        sightings.append({"chunk": chunk["index"], "frame_index": end, "world_time": pose["world_time_seconds"],
                          "heading": round(pose["heading_degrees"] % 360, 3) % 360,
                          "position": pose["position_normalized_units"], "action": action.get("action"),
                          "chunk_pixel_sha256": chunk.get("pixel_sha256")})
    return sightings


def opponent_cue(db, wid, me, other, at=None, right_axis=0, forward_axis=1):
    """Where ``other`` stands relative to ``me`` from their latest sightings at or before ``at`` (exact, local).

    Heading 0 faces +forward_axis and grows clockwise (towards +right_axis), matching RIGHT turns increasing it.
    """
    def latest(agent):
        seen = [s for s in db.sightings(wid, agent) if at is None or s["world_time"] <= at]
        if not seen:
            raise LookupError(f"no sightings of {agent} in {wid}" + (f" by world time {at}" if at is not None else ""))
        return max(seen, key=lambda s: s["world_time"])
    mine, theirs = latest(me), latest(other)
    dx = theirs["position"][right_axis] - mine["position"][right_axis]
    dy = theirs["position"][forward_axis] - mine["position"][forward_axis]
    distance = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dx, dy)) % 360
    relative = (bearing - mine["heading"] + 180) % 360 - 180
    return {"distance": distance, "bearing": bearing, "relative": relative, "my_heading": mine["heading"],
            "my_time": mine["world_time"], "their_time": theirs["world_time"]}


def cowboy_prompt(cue, unit="units"):
    """Text-only description of the other agent as a duelling cowboy, placed from ``opponent_cue``."""
    side = abs(cue["relative"])
    where = ("straight ahead" if side <= 15 else
             f"ahead to your {'right' if cue['relative'] > 0 else 'left'}" if side <= 60 else
             f"off to your {'right' if cue['relative'] > 0 else 'left'}" if side <= 120 else "behind you")
    return (f"a cowboy gunslinger standing in the dusty street {where}, about {cue['distance']:.1f} {unit} away, "
            "facing you with his hand by his holster, a western duel standoff")


def observation_point(wid, agent, sighting, audience=None, thumbnail=None):
    """The exact past.dev ingest body for one observation. Text only; the frame is a reference."""
    position = ", ".join(f"{value:g}" for value in sighting["position"])
    seen = f" and saw {sighting['caption']}" if sighting.get("caption") else ""
    content = (f"In world {wid}, agent {agent} at world time {sighting['world_time']:.2f} s, after action "
               f"{sighting.get('action') or 'unknown'}, faced heading {sighting['heading']:.0f} degrees at position "
               f"({position}){seen}. Frame reference frame:sha256:{sighting['frame_sha256']} "
               f"(chunk {sighting['chunk']}, frame {sighting['frame_index']}).")
    metadata = {"kind": "observation", "world_id": wid, "agent": agent, "chunk": sighting["chunk"],
                "heading_degrees": sighting["heading"], "position": sighting["position"],
                "world_time_seconds": sighting["world_time"], "frame_ref": f"frame:sha256:{sighting['frame_sha256']}",
                "chunk_pixel_sha256": sighting.get("chunk_pixel_sha256")}
    if thumbnail:
        metadata["thumbnail_jpeg_b64"] = thumbnail
    point = {"id": f"world:{wid}:{agent}:c{sighting['chunk']:03d}", "content": content,
             "label": f"{agent} heading {sighting['heading']:.0f} deg, t={sighting['world_time']:.2f}s",
             "timestamp": world_timestamp(sighting["world_time"]), "metadata": metadata, "identity": agent}
    if audience:
        point["audience"] = audience
    return point


def manifest_point(wid, manifest):
    fields = "; ".join(f"{key} {value}" for key, value in sorted(manifest.items()) if value is not None)
    return {"id": f"world:{wid}:manifest", "content": f"World {wid} is defined by: {fields}. Agents that start "
            "from this initial image, caption, seed, model revision, frame rate and chunk size start in the same "
            "world.",
            "label": f"World {wid} definition", "timestamp": world_timestamp(0),
            "metadata": {"kind": "world", "world_id": wid, **manifest}}


def divergence(run_a, run_b):
    """Position of the first chunk whose raw pixels differ or are not recorded (None if all compared chunks match)."""
    for position, (a, b) in enumerate(zip(run_a["chunks"], run_b["chunks"])):
        if not a.get("pixel_sha256") or a.get("pixel_sha256") != b.get("pixel_sha256"):
            return position
    return None


def divergence_point(wid, agents, run_a, run_b):
    first = divergence(run_a, run_b)
    compared = min(len(run_a["chunks"]), len(run_b["chunks"]))
    sightings = run_sightings(run_a)
    if first is None:
        shared = sightings[compared - 1]["world_time"]
        text = f"identical pixels in all {compared} compared chunks (through world time {shared:.2f} s)"
    elif first == 0:
        shared = 0
        text = ("their views differ from the first chunk, so observations by one agent are not guaranteed to hold "
                "for the other")
    else:
        shared = sightings[first - 1]["world_time"]
        text = (f"identical pixels through chunk {first - 1} (world time {shared:.2f} s); their views first differ at "
                f"chunk {first}, so observations by one agent after that are not guaranteed to hold for the other")
    return {"id": f"world:{wid}:sync:{agents[0]}:{agents[1]}",
            "content": f"In world {wid}, agents {agents[0]} and {agents[1]} share the world: {text}.",
            "label": f"Shared-world sync {agents[0]} / {agents[1]}", "timestamp": world_timestamp(shared),
            "metadata": {"kind": "sync", "world_id": wid, "agents": list(agents), "first_divergent_chunk": first}}


def from_hynomas(raw, walk_speed=2.8):
    """Convert a Hy-Nomas duel run JSON to this module's run format.

    Heading is the measured yaw. Position is dead-reckoned from it: each W chunk moves ``walk_speed`` units/s along the
    mean of the measured headings before and after that chunk (the commanded plan's position ignores how far the model
    actually turned). The commanded pose is kept alongside.
    """
    revisions = raw["revisions"]
    run = {"model": raw["model"], "source_revision": revisions["hy_nomas"], "checkpoint_revision": revisions["hy_weights"],
           "task_id": raw["task_id"], "initial_image_sha256": raw["initial_image_sha256"],
           "image_caption_sha256": raw["caption_sha256"], "seed": raw["seed"], "fps": raw["fps"],
           "chunk_size": raw["frames_per_chunk_requested"], "chunks": [], "actions": []}
    x = z = previous = 0.0
    for expected, chunk in enumerate(raw["chunks"]):
        if chunk["index"] != expected or chunk["first_frame_index"] + chunk["frame_count"] != chunk["cumulative_frames"]:
            raise ValueError(f"chunk {expected} has inconsistent indices: {chunk}")
        after = chunk["after"]
        heading = after["measured_heading_deg"]
        if chunk["action"] == "W":
            mean = math.radians(previous + ((heading - previous + 180) % 360 - 180) / 2)
            step = walk_speed * chunk["frame_count"] / raw["fps"]
            x, z = x + step * math.sin(mean), z + step * math.cos(mean)
        elif chunk["action"] in ("S", "A", "D"):
            raise ValueError(f"dead reckoning for {chunk['action']} is not implemented")
        previous = heading
        run["chunks"].append({"index": chunk["index"], "frames": chunk["frame_count"],
                              "pixel_sha256": chunk["pixel_sha256"]})
        run["actions"].append({"action": chunk["action"], "prompt": chunk.get("prompt"),
                               "pose_after": {"heading_degrees": heading, "world_time_seconds": after["world_time_s"],
                                              "position_normalized_units": [round(x, 3), 0.0, round(z, 3)],
                                              "commanded_heading_degrees": after["commanded_heading_deg"],
                                              "commanded_position": after["commanded_position"]}})
    return run


def load_run(run_dir):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    if "revisions" in run:
        run = from_hynomas(run)
    video = run_dir / "rollout.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"{video} not found")
    return run, video


def ingest_run(run_dir, agent, db, audience=None, thumbnails=True, captions=None):
    """Store every chunk's last frame in the database and queue one observation per chunk for past.dev."""
    run, video = load_run(run_dir)
    manifest = world_manifest(run)
    wid = world_id(manifest)
    db.put_world(wid, manifest)
    db.queue(manifest_point(wid, manifest))
    sightings = run_sightings(run)
    frames = frames_from_video(video, [s["frame_index"] for s in sightings])
    queued = 0
    for sighting in sightings:
        jpeg, width, height = frames[sighting["frame_index"]]
        sighting["frame_sha256"] = db.put_frame(jpeg, width, height)
        sighting["caption"] = (captions or {}).get(sighting["chunk"])
        db.add_sighting(wid, agent, sighting)
        point = observation_point(wid, agent, sighting, audience, thumbnail_b64(jpeg) if thumbnails else None)
        queued += db.queue(point)
    return {"world_id": wid, "agent": agent, "sightings": len(sightings), "queued": queued}


def ingest_credits(point):
    """1 credit per 350 bytes of content, at least 1; metadata (thumbnails) was not billed in the live run."""
    return max(1, -(-len(point["content"].encode()) // 350))


def send_pending(db, client=None, dry_run=True, log=print):
    points = db.pending()
    size = sum(len(point["content"].encode()) for point in points)
    credits = sum(ingest_credits(point) for point in points)
    if dry_run:
        log(f"dry run: {len(points)} data points, {size} content bytes, about {credits} ingest credits; nothing sent")
        return {"pending": len(points), "bytes": size, "credits": credits, "sent": 0}
    client = client or PastClient()
    for point in points:
        receipt = client._call("POST", "/ingest", point)
        db.mark_sent(point["id"], receipt.get("ingestionId"), receipt.get("status"))
    log(f"sent {len(points)} data points (about {credits} credits)")
    return {"pending": 0, "bytes": size, "credits": credits, "sent": len(points)}


def recall_view(wid, heading, identity, db=None, client=None, at=None, limit=8):
    """Ask past.dev what is known about a heading as of world time ``at``; resolve frame refs locally."""
    client = client or PastClient()
    body = {"query": f"What has been seen in world {wid} around heading {heading:.0f} degrees?",
            "identity": identity, "limit": int(limit)}
    if at is not None:
        body["queryTimestamp"] = world_timestamp(at)
    page = client._call("POST", "/recall", body)
    results = []
    for item in page.get("results", []):
        worlds = {(source.get("metadata") or {}).get("world_id") for source in item.get("sources", [])} - {None}
        if wid not in item.get("content", "") and wid not in worlds:
            continue
        refs = set(FRAME_REF.findall(item.get("content", "")))
        for source in item.get("sources", []):
            meta = source.get("metadata") or {}
            refs.update(FRAME_REF.findall(str(meta.get("frame_ref", ""))))
            for excerpt in source.get("excerpts", []):
                refs.update(FRAME_REF.findall(excerpt))
        results.append({"rank": item.get("rank"), "date": item.get("occurredAt"), "text": item.get("content", ""),
                        "frames": sorted(refs),
                        "frames_available": sorted(ref for ref in refs if db is not None and db.get_frame(ref))})
    return {"results": results, "pending": page.get("pending"), "as_of": page.get("asOf")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=str(DEFAULT_DB))
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest-run", help="store a rollout's frames and queue its observations")
    ingest.add_argument("run_dir")
    ingest.add_argument("--agent", required=True)
    ingest.add_argument("--audience", help="past.dev audience slug for private observations")
    ingest.add_argument("--no-thumbnails", action="store_true")
    ingest.add_argument("--captions", help="JSON file mapping chunk index to a caption")
    mode = ingest.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="show what would be sent; send nothing")
    mode.add_argument("--confirm-send", action="store_true", help="send the queued data points to past.dev")
    compare = commands.add_parser("compare", help="record where two agents' worlds stop matching")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.add_argument("--agents", nargs=2, required=True)
    mode = compare.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="show what would be sent; send nothing")
    mode.add_argument("--confirm-send", action="store_true", help="send the queued data points to past.dev")
    ask = commands.add_parser("recall", help="recall a heading from past.dev as one agent")
    ask.add_argument("world_id")
    ask.add_argument("--heading", type=float, required=True)
    ask.add_argument("--identity", required=True)
    ask.add_argument("--at", type=float, help="world time in seconds (point-in-time read)")
    near = commands.add_parser("nearby", help="exact local lookup of stored sightings by heading")
    near.add_argument("world_id")
    near.add_argument("--heading", type=float, required=True)
    near.add_argument("--tolerance", type=float, default=30)
    near.add_argument("--exclude")
    args = parser.parse_args(argv)
    db = FrameDB(args.db)
    try:
        if args.command in ("ingest-run", "compare"):
            if args.command == "ingest-run":
                captions = ({int(k): v for k, v in json.loads(Path(args.captions).read_text()).items()}
                            if args.captions else None)
                print(json.dumps(ingest_run(args.run_dir, args.agent, db, args.audience, not args.no_thumbnails,
                                            captions)))
            else:
                (run_a, _), (run_b, _) = load_run(args.run_a), load_run(args.run_b)
                wid_a, wid_b = world_id(world_manifest(run_a)), world_id(world_manifest(run_b))
                if wid_a != wid_b:
                    print(f"different worlds: {wid_a} vs {wid_b}", file=sys.stderr)
                    return 1
                point = divergence_point(wid_a, args.agents, run_a, run_b)
                db.queue(point)
                print(point["content"])
            send_pending(db, dry_run=args.dry_run)
        elif args.command == "recall":
            print(json.dumps(recall_view(args.world_id, args.heading, args.identity, db, at=args.at), indent=2))
        else:
            print(json.dumps(db.nearby(args.world_id, args.heading, args.tolerance, args.exclude), indent=2))
    except PastError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
