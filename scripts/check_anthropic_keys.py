"""Check which Anthropic credentials can still make a tiny Haiku call. Never prints key material."""
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def keys():
    found = {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        found["environment"] = os.environ["ANTHROPIC_API_KEY"]
    path = ROOT / ".env.local"
    if path.is_file():
        for line in path.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name == "ANTHROPIC_API_KEY":
                found["env.local"] = value.strip()
    return found


def main():
    from anthropic import Anthropic
    for source, key in keys().items():
        tag = hashlib.sha256(key.encode()).hexdigest()[:8]
        try:
            Anthropic(api_key=key, timeout=30, max_retries=0).messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=5, messages=[{"role": "user", "content": "Reply: ok"}])
            print(source, f"sha256[:8]={tag}", "OK")
        except Exception as error:
            print(source, f"sha256[:8]={tag}", "FAIL", f"{type(error).__name__}: {str(error)[:160]}")


if __name__ == "__main__":
    main()
