"""Recall what the project already knows before proposing the next experiment.

Prints dated findings with their sources as the context block an agent (or a person) reads before planning:

    python scripts/propose_next_experiment.py "what do we know about memory on out-of-sight tasks?"

With PAST_API_KEY set, findings come from past.dev (worldsync.research_memory.recall). Without a key, or with
--offline, a keyword search over the local outbox is used instead and the output says so: it ranks by word
overlap only and has none of past.dev's dating, supersession or semantic ranking.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worldsync import research_memory  # noqa: E402

STOP = set("a an and are about as at be by do does for from how in is it know of on or our that the to we what "
           "when which with".split())


def offline_recall(question, k, outbox=None):
    outbox = outbox or research_memory.OUTBOX
    if not outbox.exists():
        return {"results": [], "pending": None}
    words = {word for word in re.findall(r"[a-z0-9-]+", question.lower()) if word not in STOP}
    scored = []
    for line in outbox.read_text().splitlines():
        finding = json.loads(line)
        text = finding["content"].lower()
        score = sum(text.count(word) for word in words) + 3 * len(words & set(finding["metadata"]["tags"]))
        if score:
            scored.append((score, finding))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    results = [{"rank": rank, "date": finding["timestamp"], "text": finding["content"], "kind": "source",
                "superseded": False,
                "sources": [{"source": finding["metadata"]["source"], "section": finding["metadata"]["section"]}]}
               for rank, (_, finding) in enumerate(scored[:k], 1)]
    return {"results": results, "pending": None}


def context(question, k=6, offline=False):
    if offline or not os.environ.get("PAST_API_KEY"):
        origin = "local outbox keyword search (past.dev not queried)"
        page = offline_recall(question, k)
    else:
        origin = "past.dev recall"
        page = research_memory.recall(question, k=k)
    return "\n".join([f"Question: {question}", f"Prior findings ({origin}):",
                      research_memory.format_recall(page), "",
                      "Before proposing an experiment: state which finding it builds on, what it changes, the "
                      "measurement that would confirm or refute it, and whether an earlier finding already "
                      "answers it."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--offline", action="store_true", help="search the local outbox instead of past.dev")
    args = parser.parse_args(argv)
    try:
        print(context(args.question, args.k, args.offline))
    except research_memory.PastError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
