"""Copy the existing fixed-path memory comparison sheets into docs/gallery/ as compact JPEGs (no new generation)."""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "memory-budget-GC007-hold": "reproduction/budgetmem/turn_hold_return/comparison.png",
    "memory-revisit-GC007-hold": "reproduction/revisit/turn_hold_return/comparison.png",
    "memory-keyframe-GC007-hold": "reproduction/keyframe-restart/strip-B-GC007-hold.png",
    "memory-judged-returns": "reproduction/full-run-memory-audit/return-examples.jpg",
}

if __name__ == "__main__":
    for name, source in SOURCES.items():
        image = Image.open(ROOT / source).convert("RGB")
        if image.width > 1600:
            image = image.resize((1600, round(image.height * 1600 / image.width)), Image.LANCZOS)
        target = ROOT / "docs/gallery" / f"{name}.jpg"
        image.save(target, quality=80)
        print(name, image.size, target.stat().st_size)
