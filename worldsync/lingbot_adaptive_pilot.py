"""Explicit local adaptive-memory protocol; separate from the stock benchmark."""
from worldsync.lingbot_tuning import is_stock

PROTOCOL = "lingbot-world2-adaptive-four-view-pilot-v1"


def validate(args, knobs, parser):
    if not args.adaptive_retrieval_pilot:
        return
    other_memory = any(getattr(args, name) for name in
                       ("rephase_sink", "kv_retention", "decoprune", "reanchor", "budget_memory"))
    if (args.split != "gc" or args.task_id != "GC027" or args.fixed_actions
            or args.native_fixed_controls or args.turn_degrees_per_second != 30
            or not is_stock(knobs) or other_memory or args.denoise_schedule != "stock"
            or args.retrieved_frames != 4 or args.retrieval_selection != "nearest"
            or args.retrieval_positions != "virtual" or args.history_output_weight is not None
            or not args.offload_text_encoder):
        parser.error("Adaptive pilot requires GC027, online player controls,30deg/s, stock18/6 sampler, "
                     "text encoder offload and optional four-nearest virtual retrieval only")


def player_state(camera):
    return {**camera.commanded_state(), "turn_degrees_per_second": 30,
            "pilot_note": "The initial WAIT bootstraps the image and is not a reference-plan step. "
            "Camera turns30degrees per simulated second, so a360degree goal needs about12seconds "
            "of turning even if the web reference plan has different timing. Verify completion visually; "
            "commanded yaw alone does not establish a revisit."}
