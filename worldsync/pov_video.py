"""Shared handoff intent projected into two first-person LingBot render plans.

Symbolic state and time are shared. Visual histories remain independent and
must be evaluated for disagreement; no joint multiview latent is claimed.
"""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess

ACTORS = ("orange", "purple")
CHUNKS = 12
FPS = 16
EXPECTED_FRAMES = CHUNKS * 16 - 3
STATES = (
    {"revision": 0, "chunk": 0, "holder": "orange", "phase": "hold"},
    {"revision": 1, "chunk": 2, "holder": "orange", "phase": "offer"},
    {"revision": 2, "chunk": 4, "holder": "orange", "phase": "transfer"},
    {"revision": 3, "chunk": 7, "holder": "purple", "phase": "received"},
)


def validate_states(states):
    previous = -1
    for index, state in enumerate(states):
        if state["revision"] != index or state["chunk"] <= previous:
            raise ValueError("world revisions and chunk times must increase")
        if state["holder"] not in ACTORS:
            raise ValueError("exactly one actor must hold the basket")
        previous = state["chunk"]
    if states[0]["chunk"] != 0 or states[-1]["chunk"] >= CHUNKS:
        raise ValueError("world timeline must start at zero and fit inside the video")


def viewpoint_text(actor, state):
    if actor not in ACTORS:
        raise ValueError("unknown viewpoint")
    common = ("A continuous first-person view at seated eye level beside a sunny Japanese river. "
              "There is one empty round brown woven basket with a tall arched handle. "
              "The same companion and same riverbank remain visible throughout. ")
    if actor == "orange":
        identities = ("Only the viewer's own orange floral sleeves and hands are visible in the foreground. "
                      "The companion sitting opposite wears a purple floral kimono with a pale green bow "
                      "and a pink flower in her pinned up dark hair. The river is on the right. ")
        action = {
            "hold": "The viewer is holding the basket low in their lap with both hands. The purple-clothed companion is looking at the basket, her empty hands resting on her knees.",
            "offer": "The viewer's orange-sleeved hands are slowly extending the single basket toward the purple-clothed companion. She is reaching her two hands forward toward the sides of the offered basket.",
            "transfer": "The purple-clothed companion is gripping the sides of the basket with both hands and drawing it toward her lap. The viewer's orange-sleeved hands are opening and releasing the basket, then moving back empty.",
            "received": "The purple-clothed companion is holding the single basket still in her lap with both hands, smiling. The viewer's empty orange-sleeved hands are resting on their own knees. The basket stays with the companion.",
        }
    else:
        identities = ("Only the viewer's own purple floral sleeves, knees and hands are visible in the foreground. "
                      "The companion sitting opposite wears an orange floral kimono with a silver bow "
                      "and a pale green flower in her pinned up dark hair. The river is on the left. ")
        action = {
            "hold": "The orange-clothed companion is holding the single basket in her lap with both hands. The viewer's own purple-sleeved hands are resting empty on their knees.",
            "offer": "The orange-clothed companion is slowly extending the single basket toward the viewer. The viewer's purple-sleeved hands are lifting from their knees and reaching forward toward its sides.",
            "transfer": "The viewer's purple-sleeved hands are gripping the sides of the offered basket and drawing it into their own lap. The orange-clothed companion is releasing her grip and withdrawing her empty hands.",
            "received": "The viewer's purple-sleeved hands are holding the single basket still in their own lap in the foreground. The orange-clothed companion is smiling, her two empty hands resting on her own knees. The basket stays in the viewer's lap.",
        }
    return common + identities + action[state["phase"]]


def prepare(inputs):
    """Build inputs without touching the original PlayWorld tasks or images."""
    inputs = Path(inputs)
    validate_states(STATES)
    for actor in ACTORS:
        if not (inputs / f"{actor}.png").is_file():
            raise ValueError(f"missing first-person reference: {actor}.png")
        dataset = inputs / actor
        (dataset / "insight").mkdir(parents=True, exist_ok=True)
        (dataset / "images").mkdir(exist_ok=True)
        (dataset / "images/start.png").write_bytes((inputs / f"{actor}.png").read_bytes())
        initial = viewpoint_text(actor, STATES[0])
        task = {"task_id": "POV001", "image_path": "images/start.png", "image_caption": initial,
                "prompt": "Receive or offer the single basket according to the shared handoff timeline.",
                "action_sequence_steps": ["wait(12000ms)"]}
        (dataset / "insight/data.json").write_text(json.dumps([task], indent=2)+"\n")
        scheduled = [[state["chunk"], viewpoint_text(actor, state)] for state in STATES]
        (inputs / f"{actor}-director.json").write_text(json.dumps({"cases": {
            "insight:POV001": {"static": initial, "scheduled": scheduled}}}, indent=2)+"\n")
    manifest = {"actors": list(ACTORS), "chunks": CHUNKS, "fps": FPS,
                "expected_frames_per_view": EXPECTED_FRAMES, "states": STATES,
                "controller": "scripted", "ipc": "one shared versioned intent schedule",
                "reference_source": "image-generation tool, reverse viewpoints using the same scene reference",
                "visual_memory": "independent per-agent histories, shared weights only",
                "cross_view_consistency": "unverified; symbolic intent does not guarantee consistent pixels"}
    (inputs / "shared-plan.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=width,height,nb_frames,avg_frame_rate:format=duration", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return data["streams"][0], float(data["format"]["duration"])


def tile(orange, purple, destination, labels=None):
    """Tile synchronized actual video streams; reject length or cadence mismatch.

    Each view is letterboxed to fit its panel. No spatial crop, frame freezing,
    retiming, or separate time origin is used to hide cross-view disagreements.
    """
    a, da = probe(orange)
    b, db = probe(purple)
    if (int(a["nb_frames"]) != int(b["nb_frames"])
            or abs(da-db) > 1/FPS
            or abs(float(Fraction(a["avg_frame_rate"]))-FPS) > .01
            or abs(float(Fraction(b["avg_frame_rate"]))-FPS) > .01):
        raise ValueError("view frame counts, durations and cadence must match")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    labels = labels or ("AGENT A - ORANGE POV", "AGENT B - PURPLE POV")
    if len(labels) != 2 or any(not label or any(c in label for c in "':;[]\\") for label in labels):
        raise ValueError("two plain-text agent labels are required")
    def panel(index, label, color):
        return (f"[{index}:v]setpts=PTS-STARTPTS,scale=832:464:force_original_aspect_ratio=decrease,"
                f"pad=832:464:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
                f"pad=832:522:0:58:color=0x101724,drawtext=fontfile={font}:"
                f"text='{label}':x=22:y=17:fontsize=23:fontcolor={color}[v{index}]")
    filters = (panel(0, labels[0], "0xffca65")+";"+
               panel(1, labels[1], "0xc5a3ff")+";"+
               "[v0][v1]hstack=inputs=2,format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(orange), "-i", str(purple),
                    "-filter_complex", filters, "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "21",
                    "-preset", "fast", "-movflags", "+faststart", str(destination)], check=True)
    result = {"left": {"path": str(orange), "label": labels[0], **a},
              "right": {"path": str(purple), "label": labels[1], **b},
              "duration": da, "output": str(destination), "synchronization": "same source frame index"}
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    return result
