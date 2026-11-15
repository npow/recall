"""Opt-in LingBot inference knobs. Defaults are the stock native settings.

Adaptive runs stay gated. Explicit fixed-path native-camera ablations are
allowed separately and must not be mixed into a stock benchmark baseline.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"local_attn_size": 18, "sink_size": 6, "decision_interval": 1}


def add_arguments(parser):
    parser.add_argument("--controlled-ablation", action="store_true",
        help="Allow declared non-stock knobs only on fixed native-camera diagnostic paths")
    parser.add_argument("--local-attn-size", type=int, default=DEFAULTS["local_attn_size"],
        help="KV cache length in latent frames, including sink (stock 18)")
    parser.add_argument("--sink-size", type=int, default=DEFAULTS["sink_size"],
        help="Retained initial-frame anchor latents (stock 6)")
    parser.add_argument("--decision-interval", type=int, default=DEFAULTS["decision_interval"],
        help="Chunks per player decision; the action is held between decisions (stock 1)")


def resolve(args, parser, state_path=None):
    knobs = {name: getattr(args, name) for name in DEFAULTS}
    if knobs["sink_size"] < 0 or knobs["decision_interval"] < 1 or knobs["local_attn_size"] <= knobs["sink_size"]:
        parser.error("require 0 <= sink < local attention and decision interval >= 1")
    if (knobs["local_attn_size"] - knobs["sink_size"]) % 4:
        parser.error("rolling window (local attention minus sink) must be a multiple of the 4-latent chunk size")
    if knobs != DEFAULTS:
        if getattr(args, "controlled_ablation", False):
            if not getattr(args, "native_fixed_controls", False) or not getattr(args, "fixed_actions", None):
                parser.error("controlled ablations require fixed actions and native camera controls")
            return knobs
        state = json.loads((state_path or ROOT / "reproduction/state.json").read_text())
        if state.get("phase") != "hillclimbing" or not state.get("baseline_reproduced"):
            parser.error("non-stock inference knobs are locked until a baseline report opens hillclimbing")
    return knobs


def is_stock(knobs):
    return knobs == DEFAULTS


def held_decision(history, index, interval):
    """Return the held previous action between decisions, else None."""
    if interval > 1 and index % interval and history and history[-1]["operation"] != "END":
        return {"operation": "KEEP", "action": history[-1]["action"], "reason": "held between player decisions"}
    return None
