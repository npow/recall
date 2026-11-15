"""Regenerate the repository leaderboard from complete, separate cohorts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    path = ROOT / name
    return json.loads(path.read_text()) if path.exists() else None


def generate():
    cohorts = []
    definitions = (
        ("Scripted task pilot", "task-pilot-", ("stock", "four-view"), 3),
        ("Adaptive GC027", "adaptive-", ("stock", "memory"), 2),
        ("Start-anchor replay", "start-anchor-", ("stock", "four-view", "start-anchor"), 2),
    )
    for title, prefix, arms, expected in definitions:
        for arm in arms:
            results = [read(str(path.relative_to(ROOT))) for path in
                       (ROOT / "reproduction/gemini-current/episodes").glob(f"{prefix}*-{arm}.json")]
            scored = [row for row in results if not row.get("error")
                      and (row.get("scores") or {}).get("official") is not None]
            complete = len(scored) == expected
            cohorts.append({"cohort": title, "method": arm, "expected": expected, "scored": len(scored),
                "errors": sum(bool(row.get("error")) for row in results), "complete": complete,
                "score": sum(row["scores"]["official"] for row in scored) / expected if complete else None,
                "visual_revisits": sum((row.get("gate") or {}).get("revisit_observed") == "yes" for row in scored),
                "source": "reproduction/gemini-current/episodes/"})
    historical = []
    pilot = read("reproduction/task-memory-benchmark/summary.json")
    if pilot and pilot["complete"]:
        historical.extend({"cohort": "Scripted tasks: 3 cases, seed42", "method": method, "score": value,
                           "source": "reproduction/task-memory-benchmark/summary.json"}
                          for method, value in pilot["local_cohort_means"].items())
    adaptive = read("reproduction/adaptive-four-view/paired-summary.json")
    if adaptive:
        historical.extend({"cohort": "Adaptive GC027: seeds42/43", "method": method,
                           "score": adaptive[f"{method}_mean"], "source": "reproduction/adaptive-four-view/paired-summary.json"}
                          for method in ("stock", "memory"))
    replay = read("reproduction/start-anchor-replay/scores-summary.json")
    if replay and replay["complete"]:
        historical.extend({"cohort": "Fixed GC027 replay: seeds42/43", "method": method, "score": value,
                           "source": "reproduction/start-anchor-replay/scores-summary.json"}
                          for method, value in replay["means"].items())
    snapshot = {"target": 2.12, "full_benchmark_score": None, "published_score_beaten": False,
                "gemini_model": "gemini-3.1-pro-preview", "endpoint": "Vertex AI global",
                "gemini_local_cohorts": cohorts, "historical_claude_cohorts": historical,
                "comparability": "Subset scores are not comparable to published four-dimension overall. Methods only compared within the same cohort/judge. Gemini model alias, player and adapter differences remain documented."}
    (ROOT / "leaderboard.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    lines = ["# WorldSync leaderboard", "",
        "**Goal: beat Genie 3’s published overall 2.12. No comparable full-benchmark score yet.**", "",
        "Compare methods only within the same table/cohort and judge. A higher subset score does not beat the published record.", "",
        "## Published references — not reproduced here", "",
        "| Model | Overall | Scope |", "|---|---:|---|",
        "| Genie 3 |2.12|Published closed-model result|",
        "| LingBot-World2 |1.82|Published hosted-service result; not our local checkpoint measurement|", "",
        "Source: [pinned PlayWorld leaderboard](external/PlayWorld/README.md#leaderboard). Overall averages GC, IF, in-sight evolution and out-of-sight evolution; the full dataset has 171 tasks (48/50/30/43).", "",
        "## Gemini — local cohorts", "",
        "Released prompts, sampling, gate and aggregation; Vertex `gemini-3.1-pro-preview`. The paper's `gemini-3.1-pro` API name is unavailable to our configured API key. Preview-model equivalence is not established. These are GC subsets, not overall scores.", "",
        "| Cohort | Method | Scored | Score | Judge reports revisit |", "|---|---|---:|---:|---:|"]
    for row in cohorts:
        score = f"{row['score']:.2f}" if row["complete"] else "pending"
        lines.append(f"| {row['cohort']} | {row['method']} |{row['scored']}/{row['expected']}|{score}|{row['visual_revisits']}/{row['scored']}|")
    lines.extend(["", "Evidence: [individual Gemini judgments](reproduction/gemini-current/episodes/). Pending/error rows never receive a partial-cohort mean.", "",
        "The earlier [Gemini cohort](reproduction/gemini-rescore.md) found a paired 46/6-cache change of +0.69 over 24 pairs, but only GC/OE coverage, one judge draw per video, generation failures and divergent player trajectories. Its unmatched arm means are not an overall leaderboard.", "",
        "## Claude — historical experiments", "",
        "Retained for audit; do not compare these numbers with Gemini or across different cohorts.", "",
        "| Cohort | Method | Score | Evidence |", "|---|---|---:|---|"])
    for row in historical:
        lines.append(f"| {row['cohort']} | {row['method']} |{row['score']:.2f}|[summary]({row['source']})|")
    lines.extend(["", "The adaptive four-view pilot changed 3.60→3.90, but neither seed visibly returned; the gain is inconclusive. The fixed start-anchor follow-up also failed all visual revisits. [Identical-video judge variability](reproduction/judge-repeatability.json) reached 2.70→3.75. GC013 is quarantined for an upstream image/caption mismatch.", "",
        "## Other measured results", "",
        "- [Long-return memory](reproduction/memory-validation/README.md): a 31.8-second episode with 24 seconds away retained the landmark; residual framing/text drift remains. Not a benchmark score.",
        "- [Local submaps](reproduction/spatial-local-submaps/summary.json): better mall localization, bridge and drifted-stock failures; not generator conditioning.",
        "- [Persistent runner](reproduction/modal-runner.md): 3.6x steady short-job throughput in the reported seven-job trial, exact outputs on the same GPU model. [Mixed-scene batch](reproduction/lingbot-batch-validation/) preserves all 42 chunk hashes.",
        "", "Regenerate: `.venv-lingbot/bin/python scripts/update_leaderboard.py`. Machine-readable snapshot: [leaderboard.json](leaderboard.json).", ""])
    (ROOT / "LEADERBOARD.md").write_text("\n".join(lines))


if __name__ == "__main__":
    generate()
