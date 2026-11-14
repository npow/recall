"""Download pinned ABot checkpoints (about 25GB); resumes interrupted downloads."""
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    "acvlab/ABot-World-0-5B-LF",
    revision="d26a3003701e70d38cd17e2af2faf63bd4ff7632",
    local_dir=Path(__file__).resolve().parents[1] / "external/ABot-World/checkpoints/ABot-World-0-5B-LF",
)
