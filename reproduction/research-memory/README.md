# Research memory on past.dev

What it does: every experiment finding in this repository's write-ups is stored as dated text in
[past.dev](https://past.dev) long-term memory, and a planner can ask a question ("which memory result is unconfirmed on a
second scene?", "where does the gain shrink?") and get back ranked, dated, sourced findings before choosing the next experiment.
Code: `worldsync/research_memory.py` (client + CLI), `scripts/propose_next_experiment.py` (recall as planning context).

```
python -m worldsync.research_memory ingest-repo --dry-run        # write outbox.jsonl, send nothing
python -m worldsync.research_memory ingest-repo --confirm-send   # send (needs PAST_API_KEY)
python -m worldsync.research_memory recall "<question>"
python -m worldsync.research_memory status
```

## What was sent (exactly)

- 100 findings, 86,431 bytes of text, about 295 ingest credits. Each is one section of a tracked write-up
  (`LEADERBOARD.md`, `reproduction/*.md`, ...), with its original commit time as the timestamp and an id of the form
  `worldsync:<file>#<section>`. The exact request bodies are in [`outbox.jsonl`](outbox.jsonl); ids, content hashes and
  ingestion ids are in [`manifest.jsonl`](manifest.jsonl). Re-running sends only new or changed findings (idempotent).
- Nothing else: no videos, no keys, no local paths, no email addresses, no IP addresses (redacted before building the outbox and
  re-checked by grep). The key is read from `PAST_API_KEY` and is never logged or stored.

## What was verified

- End to end: one test point was ingested, its ingestion reached `completed`, and a recall returned it with its source excerpt.
  It was then deleted.
- Bulk ingest of the 100 findings was accepted. When [recall-demo.md](recall-demo.md) was generated, 66 of 100 ingestions had
  `completed` and 34 were still `processing`; the recall output there is therefore from a partial index.
  Five planning questions return on-topic findings with dates and source sections.
- Mocked-HTTP tests (`tests/test_research_memory.py`, 6 passing): request shape and auth header, no key in `repr`, retry rules,
  redaction, idempotent manifest, refusal without a key.
- One real bug found and fixed against the live API: past.dev refuses `/` in a source id (`400 id-invalid`).

## What was not done

- The two planned recall-driven GPU experiments were not run (scope cut for time). No experiment result in this repository was
  produced with, or influenced by, this memory: the integration was added on 2026-11-15, after the experiments.
- No claim is made that recall improves research outcomes; this shows the integration works, not that it helped.
