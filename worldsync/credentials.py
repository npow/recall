"""Load only expected API variables from an ignored local file; never execute it."""
import os
from pathlib import Path


def load_local_credentials(path: Path | None = None):
    path = path or Path(__file__).resolve().parents[1] / ".env.local"
    if path.is_file():
        for line in path.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name in {"OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"}:
                os.environ.setdefault(name, value.strip())
