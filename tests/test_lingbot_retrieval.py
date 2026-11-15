import numpy as np
import pytest

from worldsync.lingbot_paths import PATHS
from worldsync.lingbot_controls import CameraPlan
from worldsync.lingbot_retrieval import select_frames


def test_retrieval_uses_only_old_non_sink_views_and_requires_camera_overlap():
    camera = CameraPlan()
    for action in PATHS["round_trip"]:
        camera.advance(action)
    poses = np.stack(camera.poses)
    selected = select_frames(poses, range(64), 60)
    assert len(selected) == 4
    assert all(6 <= index < 56 for index in selected)
    assert select_frames(poses[:32], range(6, 20), 28) == []  # Facing away.
    assert select_frames(poses[:8], range(8), 4) == []  # No future or sink leakage.


@pytest.mark.parametrize("method", ["retrieval", "sink_rephase"])
def test_actual_native_attention_retains_stock_output_until_retrieval_and_bounds_memory(monkeypatch, method):
    import copy
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    source = Path(__file__).resolve().parents[1] / "external/LingBot-World2"
    if not source.exists():
        pytest.skip("pinned upstream checkout not installed")
    existing = sys.modules.get("wan")
    if existing and not Path(existing.__file__).resolve().is_relative_to(source.resolve()):
        # ABot and LingBot both vendor a package named wan. Exercise this native
        # integration in a fresh interpreter rather than replacing live modules.
        import subprocess
        subprocess.run([sys.executable, "-m", "pytest", "-q",
            f"{__file__}::test_actual_native_attention_retains_stock_output_until_retrieval_and_bounds_memory[{method}]"], check=True)
        return
    monkeypatch.syspath_prepend(str(source))
    from wan.modules import model_fast
    from worldsync.matrix_attention import sdpa_attention
    from worldsync.lingbot_retrieval import PoseKVMemory
    monkeypatch.setattr(model_fast, "attention", sdpa_attention)
    torch.manual_seed(7)
    original = model_fast.CausalWanSelfAttention(64, 4, local_attn_size=18, sink_size=6).eval()
    candidate = copy.deepcopy(original)
    pipeline = SimpleNamespace(model=SimpleNamespace(blocks=[SimpleNamespace(self_attn=candidate)]))
    from worldsync.lingbot_sink_rephase import SinkRephaseMemory
    memory = (PoseKVMemory if method == "retrieval" else SinkRephaseMemory)(pipeline)
    def cache():
        return {"k": torch.zeros(1, 72, 4, 16), "v": torch.zeros(1, 72, 4, 16),
                "global_end_index": torch.tensor([0]), "local_end_index": torch.tensor([0])}
    stock_cache, modified_cache = cache(), cache()
    frequency = torch.arange(1024)[:, None] * torch.linspace(.001, .1, 8)[None]
    freqs = torch.polar(torch.ones_like(frequency), frequency).to(torch.complex128)
    camera = CameraPlan()
    try:
        with torch.no_grad():
            for chunk, action in enumerate(PATHS["round_trip" if method == "retrieval" else "turn_return"]):
                camera.advance(action)
                memory.begin(chunk, camera.poses)
                memory.clean = True
                x = torch.randn(1, 16, 64)
                common = (x, torch.tensor([16]), torch.tensor([[4, 2, 2]]), freqs)
                expected = original(*common, stock_cache, chunk * 16, 72, frame_seqlen=4)
                actual = candidate(*common, modified_cache, chunk * 16, 72, frame_seqlen=4)
                if not memory.trace[-1]["attention_replaced"]:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                assert modified_cache["k"].shape == stock_cache["k"].shape
                assert len(memory.banks[0]) <= (16 if method == "retrieval" else 6)
                if method == "sink_rephase":
                    torch.testing.assert_close(modified_cache["k"], stock_cache["k"], atol=0, rtol=0)
                    torch.testing.assert_close(modified_cache["v"], stock_cache["v"], atol=0, rtol=0)
        assert memory.retrieval_calls > 0
        assert memory.trace[-1]["attention_replaced"]
    finally:
        memory.close()
