import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_fourdim  # noqa: E402
import sync_fourdim  # noqa: E402
from worldsync import fourdim  # noqa: E402


def make_seed_dir(tmp_path, scores):
    """scores: {(arm, split, task): (gemini, opus)}"""
    (tmp_path / "gemini/episodes").mkdir(parents=True)
    meta = {}
    for (arm, split, task), (gemini, opus) in scores.items():
        episode = f"fourdim-{arm}-{split}-{task}-seed42-1"
        meta[episode] = {"arm": arm, "seed": 42, "task": task, "split": split, "opus": {"verdict": "pass", "score": opus}}
        (tmp_path / "gemini/episodes" / f"{episode}.json").write_text(json.dumps(
            {"scores": {"official": gemini, "final_score": gemini}, "gate": {"verdict": "pass"}, "rubric": {"categories": {}}}))
    (tmp_path / "gemini/meta.json").write_text(json.dumps(meta))
    (tmp_path / "generated.json").write_text(json.dumps([{"arm": "stock", "returncode": 1, "split": "gc", "task": "GC999", "seed": 42}]))
    return tmp_path


def test_analysis_overall_and_pairing(tmp_path):
    scores = {}
    for split in fourdim.SPLITS:
        for index in range(3):
            scores[("stock", split, f"T{index}")] = (2.0, 2.0)
            scores[("budget", split, f"T{index}")] = (3.0, 2.0)
    out = analyze_fourdim.analyse([make_seed_dir(tmp_path, scores)])
    assert out["arms"]["stock"]["gemini"]["overall"] == 2.0
    assert out["arms"]["budget"]["gemini"]["overall"] == 3.0
    assert out["paired"]["gemini"]["delta"] == 1.0 and out["paired"]["opus"]["delta"] == 0.0
    assert len(out["generation_failures"]) == 1
    assert "paper" in analyze_fourdim.render(out).lower()


def test_partial_coverage_gives_no_overall(tmp_path):
    out = analyze_fourdim.analyse([make_seed_dir(tmp_path, {("stock", "gc", "T0"): (2.0, 2.0), ("budget", "gc", "T0"): (3.0, 2.0)})])
    assert out["arms"]["stock"]["gemini"]["overall"] is None and out["paired"]["gemini"] is None


def test_episode_name_pattern():
    match = sync_fourdim.PATTERN.match("fourdim-budget-insight-OE001-seed42-1794700000000000000")
    assert match and match["arm"] == "budget" and match["split"] == "insight" and match["task"] == "OE001"
