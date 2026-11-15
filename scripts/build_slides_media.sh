#!/usr/bin/env bash
# Rebuild docs/slides/media/ from the demo clips and the frozen final-run stock videos.
# Usage: scripts/build_slides_media.sh [FINAL1_OUTPUTS_DIR] [REVISIT_HORIZON_OUTPUTS_DIR] [MEMV2_STEP2_OUTPUTS_DIR]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$ROOT/docs/slides/media"
O="${1:-/home/npow/code/iteratehack/.claude/worktrees/agent-ad03b54239f02c9ae/outputs/final1}"
mkdir -p "$M"
enc() { ffmpeg -loglevel error -y -i "$1" -vf "$2" -an -c:v libx264 -pix_fmt yuv420p -crf 24 -preset medium -movflags +faststart "$3"; }
enc "$O/stock-gc-GC038-seed42/rollout.mp4" "scale=640:-2" "$M/fail-GC038.mp4"
enc "$O/stock-if-IF013-seed42/rollout.mp4" "scale=640:-2" "$M/fail-IF013.mp4"
enc "$O/stock-insight-OE007-seed42/rollout.mp4" "setpts=PTS/4,scale=640:-2" "$M/fail-OE007.mp4"   # 60 s shown at 4x
enc "$O/stock-outsight-OE036-seed42/rollout.mp4" "scale=640:-2" "$M/fail-OE036.mp4"
H="${2:-/home/npow/code/iteratehack/.claude/worktrees/agent-afce29f0192e55298/outputs/revisit-horizon}"
V2="${3:-/home/npow/code/iteratehack/.claude/worktrees/agent-a9e1baed8edee1ee7/outputs/memv2-step2}"
# Side-by-side pairs: as released (left) | with our memory (right); no baked-in labels (the page labels them).
pair() { # stock.mp4 ours.mp4 out.mp4 speed
  ffmpeg -loglevel error -y -i "$1" -i "$2" -filter_complex \
    "[0:v]setpts=PTS/$4,scale=640:-2,tpad=stop_mode=clone:stop_duration=2.5[a];[1:v]setpts=PTS/$4,scale=640:-2,tpad=stop_mode=clone:stop_duration=2.5[b];[a][b]hstack=inputs=2[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 24 -preset medium -movflags +faststart "$3"
}
for s in GC011 GC034 GC026; do
  pair "$H/rh-stock-$s-H12-seed42/rollout.mp4" "$H/rh-memory-$s-H12-seed42/rollout.mp4" "$M/pair-$s.mp4" 1.5
done
pair "$O/stock-gc-GC045-seed42/rollout.mp4" "$V2/memv2-gc-GC045-seed42/rollout.mp4" "$M/real-GC045.mp4" 1.0
# Stills for the hero scene: start view, and the view after turning back in each arm.
still() { ffmpeg -loglevel error -y -i "$1" -vf "select='eq(n\,$2)',scale=640:-2" -frames:v 1 -q:v 3 "$3"; }
still "$H/rh-stock-GC011-H12-seed42/rollout.mp4" 10 "$M/GC011-start.jpg"
still "$H/rh-stock-GC011-H12-seed42/rollout.mp4" 316 "$M/GC011-return-stock.jpg"
still "$H/rh-memory-GC011-H12-seed42/rollout.mp4" 316 "$M/GC011-return-ours.jpg"
rm -f "$M/hero-GC011.mp4" "$M/hero-GC011.jpg"
for f in "$M"/*.mp4; do
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  t=$(python3 -c "print(max(0.2, $d * 0.3))")
  ffmpeg -loglevel error -y -ss "$t" -i "$f" -frames:v 1 -q:v 4 "${f%.mp4}.jpg"
done
ls -la "$M" | awk '{print $5, $9}'
du -sh "$M"
