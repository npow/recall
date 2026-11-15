"""Research memory for the autoresearch loop, backed by the past.dev Memory API.

Each experiment finding (what was tried, the result with its numbers, the source file and when it was
written) is sent to past.dev as one timestamped data point. Before proposing the next experiment an agent
recalls the dated findings relevant to its question.

Only text built from tracked write-ups in this repository is ever sent, after redaction. The API key is read
from ``PAST_API_KEY`` and is never logged or written anywhere.

CLI::

    python -m worldsync.research_memory ingest-repo --dry-run      # write the outbox, send nothing
    python -m worldsync.research_memory ingest-repo --confirm-send # send (needs PAST_API_KEY)
    python -m worldsync.research_memory recall "what do we know about memory on out-of-sight tasks?"
    python -m worldsync.research_memory status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://api.past.dev/api/v1"
MEMORY_DIR = ROOT / "reproduction" / "research-memory"
MANIFEST = MEMORY_DIR / "manifest.jsonl"
OUTBOX = MEMORY_DIR / "outbox.jsonl"
AUTHOR_IDENTITY = "worldsync-autoresearch"
READER_IDENTITY = "worldsync-agent"
MAX_CHARS = 2400
MAX_SENTENCES = 8

# Tracked write-ups that hold experiment findings. Missing files are skipped.
SOURCES = (
    "reproduction/INTEGRATION.md",
    "LEADERBOARD.md",
    "reproduction/full-run/table.md",
    "reproduction/final-run/table.md",
    "reproduction/revisit-horizon.md",
    "reproduction/memory-v2.md",
    "reproduction/memory-storage-ablation.md",
    "reproduction/full-run-memory-audit.md",
    "reproduction/full-run-memory-fired.md",
    "reproduction/baseline-fidelity-audit.md",
    "reproduction/director-dev.md",
    "reproduction/score-drivers.md",
    "reproduction/oe-insight-caps.md",
    "reproduction/near-field-motion.md",
    "reproduction/judge-repeat/README.md",
    "reproduction/corrected-controller-validation.md",
    "reproduction/text-prompt-finding.md",
)

_REDACTIONS = (
    (re.compile(r"/home/[A-Za-z0-9_.-]+/(?:[^\s`'\")\]]*/)?(?=(?:reproduction|worldsync|scripts|docs|outputs|tests|external)/)"), ""),
    (re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+(?:/[^\s`'\")\]]*)?"), "<local-path>"),
    (re.compile(r"/tmp/[^\s`'\")\]]+"), "<local-path>"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<email>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b(?:sk|ak|as|pk|rk)-[A-Za-z0-9_-]{16,}\b"), "<secret>"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "<secret>"),
    (re.compile(r"\bya29\.[0-9A-Za-z_.-]+"), "<secret>"),
    (re.compile(r"\b(?:ghp|gho|github_pat|hf)_[A-Za-z0-9_]{16,}\b"), "<secret>"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"), "Bearer <secret>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "<secret>"),
)


def redact(text: str) -> str:
    """Remove local paths, email addresses, IP addresses and anything shaped like a credential."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class PastError(RuntimeError):
    def __init__(self, status, code, message):
        super().__init__(f"past.dev {status} {code}: {message}")
        self.status, self.code = status, code


class PastClient:
    """Minimal past.dev Memory API client (stdlib only). Never logs the key."""

    RETRY_STATUS = {429, 500, 502, 503, 504}
    NO_RETRY_CODES = {"out-of-credits", "project-cap-reached", "spend-cap-reached"}

    def __init__(self, api_key=None, base=None, timeout=15, retries=3, opener=None, sleep=time.sleep):
        self._key = api_key if api_key is not None else os.environ.get("PAST_API_KEY", "")
        if not self._key:
            raise PastError(0, "no-key", "PAST_API_KEY is not set")
        self.base = (base or os.environ.get("PAST_API_BASE") or API_BASE).rstrip("/")
        self.timeout, self.retries = timeout, retries
        self._open = opener or urllib.request.urlopen
        self._sleep = sleep

    def __repr__(self):
        return f"PastClient(base={self.base!r}, key=<hidden>)"

    def _call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        last = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.base + path, data=data, method=method, headers={
                "Authorization": "Bearer " + self._key, "Content-Type": "application/json",
                "Accept": "application/json", "User-Agent": "research-memory-client"})
            try:
                with self._open(request, timeout=self.timeout) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as error:
                raw = error.read() or b""
                try:
                    payload = json.loads(raw) if raw else {}
                except ValueError:
                    payload = {}
                code = str(payload.get("code") or payload.get("error") or "")
                message = str(payload.get("message") or payload.get("detail") or "")[:300]
                last = PastError(error.code, code or "http-error", message)
                if error.code not in self.RETRY_STATUS or code in self.NO_RETRY_CODES:
                    raise last from None
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last = PastError(0, "network", type(error).__name__)
            if attempt < self.retries:
                self._sleep(min(2 ** attempt, 8))
        raise last

    def ingest(self, finding):
        """Send one finding (see ``build_finding``). Returns the API receipt."""
        return self._call("POST", "/ingest", {key: finding[key] for key in
                          ("id", "content", "label", "timestamp", "metadata", "identity", "idempotencyKey")})

    def ingestion(self, ingestion_id):
        return self._call("GET", "/ingest/" + str(ingestion_id))

    def recall(self, query, k=8, level="low", identity=READER_IDENTITY):
        return self._call("POST", "/recall", {"query": query, "identity": identity, "limit": int(k), "level": level})


def build_finding(text, timestamp, source, tags=(), section="", commit=""):
    """Turn redacted text into the exact JSON body sent to POST /ingest."""
    content = redact(text).strip()
    if not content:
        raise ValueError("empty finding")
    slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")[:60] or "summary"
    # past.dev refuses "/" in a source id (400 id-invalid); use ":" as its docs suggest.
    finding_id = f"worldsync:{source.replace('/', ':')}#{slug}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "id": finding_id,
        "content": content,
        "label": f"Experiment finding: {section or source}"[:180],
        "timestamp": timestamp,
        "metadata": {"project": "worldsync", "source": source, "section": section, "commit": commit,
                     "tags": sorted(set(tags)), "sha256": digest},
        "identity": AUTHOR_IDENTITY,
        "idempotencyKey": hashlib.sha256((finding_id + digest + timestamp).encode()).hexdigest(),
    }


def ingest_finding(text, timestamp, source, tags=(), client=None, section="", commit=""):
    """Ingest one finding and record it in the manifest. Returns the manifest row."""
    finding = build_finding(text, timestamp, source, tags, section, commit)
    client = client or PastClient()
    receipt = client.ingest(finding)
    row = _manifest_row(finding, receipt.get("ingestionId"), receipt.get("status"))
    _write_manifest({**_read_manifest(), row["id"]: row})
    return row


def recall(query, k=8, client=None, level="low"):
    """Recall dated findings for a question. Returns a list of {rank, date, text, kind, sources}."""
    client = client or PastClient()
    page = client.recall(query, k=k, level=level)
    results = []
    for item in page.get("results", []):
        sources = [{"source": (group.get("metadata") or {}).get("source") or group.get("sourceId"),
                    "section": (group.get("metadata") or {}).get("section"),
                    "date": group.get("occurredAt"), "label": group.get("label")}
                   for group in item.get("sources", [])]
        results.append({"rank": item.get("rank"), "date": item.get("occurredAt"), "text": item.get("content", ""),
                        "kind": (item.get("artifact") or {}).get("kind"),
                        "superseded": bool((item.get("artifact") or {}).get("supersededAt")), "sources": sources})
    return {"results": results, "pending": page.get("pending"), "as_of": page.get("asOf")}


# ---- building findings from the repository's own write-ups -------------------------------------------------

def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False).stdout.strip()


def _file_times(path):
    """(ISO time of the last commit that changed the file, time of the commit that added it, short sha)."""
    last = _git("log", "-1", "--format=%aI %h", "--", path).split()
    added = _git("log", "--diff-filter=A", "--follow", "--format=%aI", "--", path).splitlines()
    if not last:
        return None, None, ""
    return last[0], (added[-1] if added else last[0]), last[1]


def _plain(markdown):
    """Markdown section -> plain sentences. Tables become 'a; b; c' rows; code blocks and images are dropped."""
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("![") or re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            line = "; ".join(cell for cell in cells if cell) + "."
        line = re.sub(r"^(?:[#>]+\s*|[-*+]\s+|\d+[.)]\s+)+", "", line)
        line = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"[*_`]+", "", line)
        if line:
            lines.append(line if line[-1] in ".:;!?" else line + ".")
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _clip(text):
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[+\-])", text)
    kept, size = [], 0
    for sentence in sentences:
        if len(kept) >= MAX_SENTENCES or size + len(sentence) > MAX_CHARS:
            break
        kept.append(sentence)
        size += len(sentence) + 1
    return " ".join(kept) if kept else text[:MAX_CHARS]


def _tags(source, section, text):
    words = (source + " " + section + " " + text).lower()
    table = {"memory": "memory", "director": "director", "out-of-sight": "out-of-sight", "outsight": "out-of-sight",
             "insight": "insight", "baseline": "baseline", "judge": "judge", "revisit": "revisit",
             "storage": "storage", "recompute": "recompute", "interaction": "interaction", "audit": "audit",
             "gemini": "judge", "lora": "fine-tuning", "controller": "player", "player": "player"}
    return sorted({tag for key, tag in table.items() if key in words})


def repo_findings(sources=SOURCES):
    """One finding per H2 section of each tracked write-up (the whole file when it has no H2)."""
    findings = []
    for source in sources:
        path = ROOT / source
        if not path.is_file() or not _git("ls-files", "--", source):
            continue
        modified, added, commit = _file_times(source)
        if not modified:
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        title = next((line[2:].strip() for line in body.splitlines() if line.startswith("# ")), source)
        parts = re.split(r"(?m)^## +", body)
        sections = [(title, parts[0])] + [(part.split("\n", 1)[0].strip(), part.split("\n", 1)[1] if "\n" in part else "")
                                          for part in parts[1:]]
        for heading, section_text in sections:
            plain = _clip(_plain(section_text))
            if len(plain) < 120:
                continue
            text = f"WorldSync experiment finding from {source}, section '{heading}'. {plain}"
            finding = build_finding(text, modified, source, _tags(source, heading, plain), heading, commit)
            finding["metadata"]["first_added"] = added
            findings.append(finding)
    unique = {}
    for finding in findings:
        if finding["id"] in unique:
            finding["id"] += "-" + finding["metadata"]["sha256"][:6]
        unique[finding["id"]] = finding
    return list(unique.values())


# ---- manifest and outbox -----------------------------------------------------------------------------------

def _manifest_row(finding, ingestion_id, status):
    return {"id": finding["id"], "source": finding["metadata"]["source"], "section": finding["metadata"]["section"],
            "timestamp": finding["timestamp"], "sha256": finding["metadata"]["sha256"],
            "ingestion_id": ingestion_id, "status": status}


def _read_manifest(path=None):
    path = path or MANIFEST
    if not path.exists():
        return {}
    return {row["id"]: row for row in map(json.loads, filter(None, path.read_text().splitlines()))}


def _write_manifest(rows, path=None):
    path = path or MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(rows[key], sort_keys=True) + "\n" for key in sorted(rows)))


def write_outbox(findings, path=None):
    path = path or OUTBOX
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(finding, sort_keys=True, ensure_ascii=False) + "\n" for finding in findings))
    return path


def ingest_repo(client=None, dry_run=True, limit=None, manifest_path=None, findings=None, log=print):
    """Send every finding that is new or changed. With ``dry_run`` nothing leaves the machine."""
    findings = repo_findings() if findings is None else findings
    if limit:
        findings = findings[:limit]
    if dry_run:
        path = write_outbox(findings)
        size = sum(len(finding["content"].encode("utf-8")) for finding in findings)
        credits = sum(max(1, -(-len(finding["content"].encode("utf-8")) // 350)) for finding in findings)
        log(f"dry run: {len(findings)} findings, {size} bytes of content, about {credits} ingest credits; "
            f"written to {path.relative_to(ROOT)}; nothing sent")
        return {"findings": len(findings), "bytes": size, "credits": credits, "sent": 0}
    client = client or PastClient()
    manifest = _read_manifest(manifest_path)
    sent = skipped = 0
    for finding in findings:
        known = manifest.get(finding["id"])
        if known and known["sha256"] == finding["metadata"]["sha256"] and known.get("ingestion_id"):
            skipped += 1
            continue
        receipt = client.ingest(finding)
        manifest[finding["id"]] = _manifest_row(finding, receipt.get("ingestionId"), receipt.get("status"))
        _write_manifest(manifest, manifest_path)
        sent += 1
    log(f"sent {sent}, unchanged {skipped}, total {len(findings)}")
    return {"findings": len(findings), "sent": sent, "skipped": skipped}


def status(client=None, log=print):
    manifest = _read_manifest()
    key = bool(os.environ.get("PAST_API_KEY"))
    outbox = len(OUTBOX.read_text().splitlines()) if OUTBOX.exists() else 0
    report = {"api_key_set": key, "outbox_findings": outbox, "manifest_rows": len(manifest),
              "ingested": sum(1 for row in manifest.values() if row.get("ingestion_id"))}
    if key and manifest and client is not False:
        client = client or PastClient()
        last = next((row for row in reversed(list(manifest.values())) if row.get("ingestion_id")), None)
        if last:
            state = client.ingestion(last["ingestion_id"])
            report["last_ingestion_status"] = state.get("status")
            report["project_settled"] = state.get("settled")
    log(json.dumps(report, indent=2))
    return report


def format_recall(page):
    lines = []
    if page.get("pending"):
        lines.append("(some findings are still being processed by past.dev and are not in this recall yet)")
    for item in page["results"]:
        origin = "; ".join(f"{source['source']}" + (f" ({source['section']})" if source.get("section") else "")
                           for source in item["sources"]) or "unknown source"
        flag = " [superseded]" if item["superseded"] else ""
        lines.append(f"{item['rank']}. [{(item['date'] or '')[:16]}]{flag} {item['text']}\n   source: {origin}")
    return "\n".join(lines) if lines else "No findings recalled."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest-repo", help="build findings from tracked write-ups and send them")
    ingest.add_argument("--dry-run", action="store_true", help="write the outbox only; send nothing")
    ingest.add_argument("--confirm-send", action="store_true", help="required to send anything to past.dev")
    ingest.add_argument("--limit", type=int)
    ask = commands.add_parser("recall", help="recall findings for a question")
    ask.add_argument("query")
    ask.add_argument("-k", type=int, default=8)
    ask.add_argument("--level", default="low", choices=["low", "medium", "high", "extra-high"])
    commands.add_parser("status", help="show the manifest and the state of the last send")
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest-repo":
            if not args.dry_run and not args.confirm_send:
                parser.error("pass --dry-run to inspect what would be sent, or --confirm-send to send it")
            ingest_repo(dry_run=args.dry_run, limit=args.limit)
        elif args.command == "recall":
            print(format_recall(recall(args.query, k=args.k, level=args.level)))
        else:
            status()
    except PastError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
