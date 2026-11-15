from pathlib import Path

import pytest
import torch

from worldsync.lingbot_file_archive import FileArchiveBank


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_file_bank_preserves_every_bit_after_flush_eviction_and_overwrite(tmp_path, dtype):
    bank = FileArchiveBank(tmp_path / "archive.bin")
    torch.manual_seed(71)
    expected = {}
    try:
        for index in [6, 9, 7, 21, 6]:
            pair = (torch.randn(1, 4, 2, 16).to(dtype), torch.randn(1, 4, 2, 16).to(dtype))
            expected[index] = pair
            bank[index] = pair
            bank.flush_and_evict()
        for index in [21, 6, 7, 9]:
            for actual, original in zip(bank[index], expected[index]):
                assert torch.equal(actual.view(torch.uint8), original.view(torch.uint8))
            bank.flush_and_evict()
        assert bank.metadata()["page_evictions"] == 9
        assert bank.metadata()["logical_bytes_written"] > 0
        assert bank.path.stat().st_size == 16 * 2 * expected[6][0].numel() * expected[6][0].element_size()
        with pytest.raises(ValueError):
            bank[22] = expected[6]
    finally:
        bank.close()


def test_native_attention_file_bank_matches_original_retrieval_exactly(monkeypatch, tmp_path):
    import copy
    import sys
    from types import SimpleNamespace
    from worldsync.lingbot_controls import CameraPlan
    from worldsync.lingbot_paths import PATHS
    from worldsync.lingbot_retrieval import PoseKVMemory
    from worldsync.lingbot_file_archive import FileArchiveMemory
    from worldsync.matrix_attention import sdpa_attention
    source = Path(__file__).resolve().parents[1] / "external/LingBot-World2"
    if not source.exists():
        pytest.skip("pinned native checkout not installed")
    existing = sys.modules.get("wan")
    if existing and not Path(existing.__file__).resolve().is_relative_to(source.resolve()):
        import subprocess
        subprocess.run([sys.executable, "-m", "pytest", "-q", f"{__file__}::{sys._getframe().f_code.co_name}"], check=True)
        return
    monkeypatch.syspath_prepend(str(source))
    from wan.modules import model_fast
    monkeypatch.setattr(model_fast, "attention", sdpa_attention)
    torch.manual_seed(17)
    original = model_fast.CausalWanSelfAttention(64, 4, local_attn_size=18, sink_size=6).eval()
    inputs = [torch.randn(1, 16, 64) for _ in PATHS["turn_wait_return"]]
    frequency = torch.arange(1024)[:, None] * torch.linspace(.001, .1, 8)[None]
    freqs = torch.polar(torch.ones_like(frequency), frequency).to(torch.complex128)
    observations = []
    for policy in ("original", "file"):
        candidate = copy.deepcopy(original)
        pipeline = SimpleNamespace(model=SimpleNamespace(blocks=[SimpleNamespace(self_attn=candidate)]))
        memory = (PoseKVMemory(pipeline) if policy == "original"
                  else FileArchiveMemory(pipeline, archive_root=tmp_path))
        camera = CameraPlan()
        cache = {"k": torch.zeros(1, 72, 4, 16), "v": torch.zeros(1, 72, 4, 16),
                 "global_end_index": torch.tensor([0]), "local_end_index": torch.tensor([0])}
        values = []
        try:
            with torch.no_grad():
                for chunk, action in enumerate(PATHS["turn_wait_return"]):
                    camera.advance(action)
                    memory.begin(chunk, camera.poses)
                    memory.clean = True
                    output = candidate(inputs[chunk], torch.tensor([16]), torch.tensor([[4, 2, 2]]),
                                       freqs, cache, chunk * 16, 72, frame_seqlen=4)
                    values.append((output.clone(), cache["k"].clone(), cache["v"].clone()))
                    memory.release_read_cache()
            assert memory.retrieval_calls > 0
            observations.append(values)
        finally:
            memory.close()
    for a, b in zip(*observations):
        for actual, expected in zip(a, b):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
