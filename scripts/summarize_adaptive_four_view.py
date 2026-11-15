"""Compare complete pilot arms without treating commanded poses as visual success."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATCHED_FIELDS = (
    "protocol_id", "seed", "task_id", "initial_image_sha256", "image_caption_sha256",
    "adapter_sha256", "controls_sha256", "player_sha256", "hardware", "attention_backend",
    "attention_implementation_sha256", "checkpoint_revision", "source_revision", "quantization",
    "local_attn_size", "sink_size", "sampler_timesteps_index", "sample_shift",
    "turn_degrees_per_second", "max_chunks", "initial_neutral_bootstrap", "player_temperature",
    "decision_interval", "chunk_size", "max_area", "fps", "denoise_schedule",
)


def paired_evidence(stock, memory):
    baseline, candidate = stock["run"], memory["run"]
    for key in MATCHED_FIELDS:
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
            raise ValueError(f"Paired generation differs in {key}")
    if stock["frozen_plan_sha256"] != memory["frozen_plan_sha256"]:
        raise ValueError("Paired attempts used different plans")
    interventions = [row["chunk"] for row in candidate.get("memory", {}).get("trace", [])
                     if row["attention_replaced"]]
    first = min(interventions) if interventions else None
    common = min(len(baseline["chunks"]), len(candidate["chunks"]))
    prefix_end = min(first if first is not None else common, common)
    action_divergence, pixel_divergence, unexpected = None, None, []
    for index in range(common):
        if (action_divergence is None
                and baseline["actions"][index]["action"] != candidate["actions"][index]["action"]):
            action_divergence = index
        if baseline["chunks"][index]["pixel_sha256"] != candidate["chunks"][index]["pixel_sha256"]:
            if pixel_divergence is None:
                pixel_divergence = index
            if index < prefix_end and action_divergence is None:
                unexpected.append(index)
    return {
        "seed": baseline["seed"], "retrieval_chunks": interventions,
        "first_retrieval_chunk": first, "first_action_divergence_chunk": action_divergence,
        "first_raw_pixel_divergence_chunk": pixel_divergence,
        "pre_retrieval_common_chunks": prefix_end,
        "unexpected_pixel_differences_before_retrieval_with_identical_actions": unexpected,
        "causal_interpretation": (
            "No retrieval occurred; score differences cannot demonstrate a retrieval effect" if first is None else
            "Unexpected pixel differences precede retrieval; investigate integration before attribution" if unexpected else
            "Player actions diverged before retrieval; action replay needed to isolate generator effect"
            if action_divergence is not None and action_divergence < first else
            "Identical action/pixel prefix until retrieval; inspect subsequent video and any policy divergence"),
    }


def main():
    directory = ROOT / "reproduction/adaptive-four-view"
    summaries = {mode: json.loads((directory / f"{mode}-summary.json").read_text())
                 for mode in ("stock", "memory")}
    if not all(value["complete"] for value in summaries.values()):
        raise ValueError("Score both complete arms before comparison")
    results = []
    for seed in (42, 43):
        runs = {mode: json.loads((directory / f"{mode}-seed{seed}/run.json").read_text())
                for mode in ("stock", "memory")}
        row = paired_evidence(runs["stock"], runs["memory"])
        for mode in ("stock", "memory"):
            score = next(value for value in summaries[mode]["results"] if value["seed"] == seed)
            row[mode] = {key: score[key] for key in ("effective_score", "trajectory_gate", "generated_seconds")}
        row["score_change"] = row["memory"]["effective_score"] - row["stock"]["effective_score"]
        results.append(row)
    report = {
        "scope": summaries["stock"]["scope"], "results": results,
        "stock_mean": summaries["stock"]["mean_effective_score"],
        "memory_mean": summaries["memory"]["mean_effective_score"],
        "limitations": "Two seeds, one scene, local Claude rubric; stochastic judge. A score increase without a visual revisit is not evidence of solved long-horizon geometry. Not a published-score reproduction.",
    }
    (directory / "paired-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
