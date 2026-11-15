# Caption-ablation runtime smoke

GPU-validated on local NVIDIA Thor at source `ad352d1`: two chunks,29frames,1.8125seconds, stock18/6cache, caption-plus-task text and canonical WAIT controller. Zero player calls. Source hashes, chunk hashes, actual video frame count/duration, and saved identity commanded poses pass verification.

[Video](rollout.mp4) · [Receipt](verification.json) · [Run](run.json) · [Protocol](protocol.json)

Model loading plus generation took approximately421seconds. This validates execution only, not full-length quality, score improvement, or equivalence to the H100 baseline. No judge was called. A matched full-length comparison remains outstanding.
