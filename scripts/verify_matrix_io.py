"""Compare all headless control tensors with the actual upstream input function."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/Matrix-Game/Matrix-Game-3"))

import torch
from pipeline.inference_interactive_pipeline import get_current_action
from worldsync.matrix_baseline import native_action, FPS, SOURCE_REVISION

controls = {"W": ("u", "w"), "S": ("u", "s"), "A": ("u", "a"), "D": ("u", "d"),
            "UP": ("i", "q"), "DOWN": ("k", "q"), "LEFT": ("j", "q"), "RIGHT": ("l", "q"), "WAIT": ("u", "q")}
for name, inputs in controls.items():
    with patch("builtins.input", side_effect=inputs), redirect_stdout(io.StringIO()):
        expected = get_current_action()
    actual = native_action(name)
    for channel in ("keyboard", "mouse"):
        assert torch.equal(expected[channel], torch.tensor(actual[channel])), (name, channel)

source = ROOT / "external/Matrix-Game"
inference = "Matrix-Game-3/pipeline/inference_interactive_pipeline.py"
baseline_blob = subprocess.check_output(["git", "show", f"6b213ca:{inference}"], cwd=source)
current = (source / inference).read_bytes()
assert baseline_blob == current
report = {"status": "passed", "verified_controls": list(controls), "source_revision": SOURCE_REVISION,
          "pre_paper_commit": "6b213ca004dce79eecc6938d3213ae9ffc5a5ef8",
          "inference_unchanged_since_pre_paper_commit": True,
          "inference_sha256": hashlib.sha256(current).hexdigest(), "native_export_fps": FPS,
          "scope": "Control tensor parity and source identity only; no score or rollout reproduction"}
(ROOT / "reproduction/matrix-io-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
