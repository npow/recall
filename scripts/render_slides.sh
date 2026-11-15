#!/usr/bin/env bash
# Render every slide of docs/slides/index.html to PNG with headless Firefox (for visual checks).
# Usage: scripts/render_slides.sh OUT_DIR [slide numbers...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$1"; shift || true
mkdir -p "$OUT"
N=$(grep -c '<section class="slide"' "$ROOT/docs/slides/index.html")
if [ "$#" -gt 0 ]; then LIST="$*"; else LIST=$(seq 1 "$N"); fi
PROFILE=$(mktemp -d)
for i in $LIST; do
  timeout 60 firefox --headless --no-remote --profile "$PROFILE" --window-size=1920,1080 \
    --screenshot "$OUT/slide-$(printf %02d "$i").png" "file://$ROOT/docs/slides/index.html?allfrags=1#$i" >/dev/null 2>&1 || echo "slide $i: render failed"
done
rm -rf "$PROFILE"
ls "$OUT"
