"""Small authoritative world for the multi-agent demo; no model/GPU imports.

SQLite serializes writers across processes. Every command names the revision it
observed and a retry token. Rejected commands never advance the world revision.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from pathlib import Path


INITIAL = {
    "revision": 0,
    "agents": {"alice": [1, 2], "bob": [7, 2]},
    "parcel": {"id": "red-parcel", "position": [2, 2], "holder": None},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class World:
    def __init__(self, path):
        self.path = str(path)

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS world (id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS commands (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT UNIQUE NOT NULL, fingerprint TEXT NOT NULL, reply TEXT NOT NULL);
            """)
            db.execute("INSERT OR IGNORE INTO world VALUES (1, ?)", (canonical(INITIAL),))

    def read(self):
        with self.connect() as db:
            return json.loads(db.execute("SELECT state FROM world WHERE id=1").fetchone()[0])

    def history(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT reply FROM commands ORDER BY seq")]

    def submit(self, actor, action, expected_revision, token, **arguments):
        command = {"actor": actor, "action": action, "expected_revision": expected_revision,
                   "arguments": arguments}
        fingerprint = hashlib.sha256(canonical(command).encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM commands WHERE token=?", (token,)).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise ValueError("retry token reused for a different command")
                return json.loads(old["reply"])
            before = json.loads(db.execute("SELECT state FROM world WHERE id=1").fetchone()[0])
            after = copy.deepcopy(before)
            if actor not in before["agents"]:
                reason = "unknown actor"
            elif expected_revision != before["revision"]:
                reason = "stale revision: refresh before acting"
            else:
                reason = self._apply(after, actor, action, arguments)
            accepted = reason is None
            if accepted:
                after["revision"] += 1
                db.execute("UPDATE world SET state=? WHERE id=1", (canonical(after),))
            else:
                after = before
            reply = {"token": token, **command, "accepted": accepted,
                     "reason": reason, "state": after,
                     "occurred_at": datetime.now(timezone.utc).isoformat()}
            db.execute("INSERT INTO commands(token, fingerprint, reply) VALUES (?, ?, ?)",
                       (token, fingerprint, canonical(reply)))
            db.commit()
            return reply

    @staticmethod
    def _apply(state, actor, action, args):
        position = state["agents"][actor]
        parcel = state["parcel"]
        if action == "move":
            target = args.get("target")
            if (not isinstance(target, list) or len(target) != 2
                    or any(type(v) is not int for v in target)
                    or not (0 <= target[0] < 9 and 0 <= target[1] < 5)
                    or sum(abs(a - b) for a, b in zip(target, position)) != 1):
                return "move must be one adjacent cell inside the world"
            state["agents"][actor] = target
        elif action == "pickup":
            if parcel["holder"] is not None or parcel["position"] != position:
                return "parcel is not available at this position"
            parcel.update(holder=actor, position=None)
        elif action == "drop":
            if parcel["holder"] != actor:
                return "actor does not hold parcel"
            parcel.update(holder=None, position=position[:])
        elif action == "handoff":
            recipient = args.get("recipient")
            if (recipient not in state["agents"] or recipient == actor
                    or state["agents"][recipient] != position or parcel["holder"] != actor):
                return "handoff requires the holder and a different agent at the same position"
            parcel["holder"] = recipient
        else:
            return "unknown action"
        return None


def render_context(state, actor):
    """Revision-tagged input contract for a future world-model renderer.

    A renderer consuming this contract still needs visual consistency evaluation.
    It must not promote its generated pixels back into authoritative state.
    """
    parcel = state["parcel"]
    location = (f"held by {parcel['holder']}" if parcel["holder"]
                else f"on the ground at {parcel['position']}")
    return {"world_revision": state["revision"], "observer": actor,
            "position": state["agents"][actor],
            "scene_text": f"Two agents, Alice in blue and Bob in amber, share one courtyard. "
                          f"There is exactly one red parcel, {location}. "
                          f"Alice is at {state['agents']['alice']}; Bob is at {state['agents']['bob']}.",
            "visual_memory_namespace": f"courtyard:revision:{state['revision']}"}
