import pytest
import torch

from worldsync.lingbot_decoprune import (DeCoPruneMemory, keep_mask, rope_tokens, token_mse_scores,
                                         virtual_positions)

HEAD_DIM = 128


def rope_table(max_len=64, dim=HEAD_DIM):
    """Same construction as wan.modules.model.rope_params, concatenated like WanModel.freqs."""
    def params(length, d, theta=10000):
        angles = torch.outer(torch.arange(length), 1.0 / torch.pow(theta, torch.arange(0, d, 2).to(torch.float64).div(d)))
        return torch.polar(torch.ones_like(angles), angles)
    return torch.cat([params(max_len, dim - 4 * (dim // 6)), params(max_len, 2 * (dim // 6)),
                      params(max_len, 2 * (dim // 6))], dim=1)


def reference_rope(x, grid, freqs, start_frame):
    """Verbatim logic of wan.modules.model_fast.causal_rope_apply for one sample."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid
    seq_len = f * h * w
    x_i = torch.view_as_complex(x[0, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
    freqs_i = torch.cat([freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                         freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                         freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(seq_len, 1, -1)
    return torch.view_as_real(x_i * freqs_i).flatten(2)[None].type_as(x)


def test_token_scores_follow_frame_h_w_order_and_average_2x2_patches_and_channels():
    target = torch.zeros(16, 2, 4, 6)
    predicted = torch.zeros(16, 2, 4, 6)
    predicted[:, 1, 2:4, 4:6] = 2.0      # frame 1, token (h=1, w=2)
    scores = token_mse_scores(predicted, target)
    assert scores.shape == (2 * 2 * 3,)
    assert scores.nonzero().flatten().tolist() == [1 * 6 + 1 * 3 + 2]
    assert scores[1 * 6 + 1 * 3 + 2].item() == pytest.approx(4.0)
    predicted = torch.zeros(16, 1, 2, 2)
    predicted[0, 0, 0, 0] = 4.0          # one pixel, one channel of a 2x2x16 patch
    assert token_mse_scores(predicted, torch.zeros_like(predicted)).item() == pytest.approx(16 / (16 * 4))


def test_keep_rule_is_strictly_above_threshold_and_keeps_nonfinite():
    scores = torch.tensor([0.05, 0.10, 0.11, float("nan"), float("inf")])
    assert keep_mask(scores, 0.10).tolist() == [False, False, True, True, True]
    with pytest.raises(ValueError):
        keep_mask(scores, float("inf"))


def test_virtual_positions_keep_recent_exact_and_compress_old_into_span():
    q0 = 40
    positions = virtual_positions(q0)
    real = torch.arange(q0, dtype=torch.float64)
    assert torch.equal(positions[-8:], real[-8:])               # offsets 1..8 untouched
    assert positions[0].item() == pytest.approx(q0 - 17)        # oldest lands one span back
    assert (positions[1:] > positions[:-1]).all()               # monotone
    assert positions[:-8].max() < positions[-8:].min() + 1e-9   # compressed block precedes the recent frames


def test_virtual_positions_guard_and_off_mode_stay_real():
    for q0 in (4, 8, 12, 16, 17):
        assert torch.equal(virtual_positions(q0), torch.arange(q0, dtype=torch.float64))
    assert torch.equal(virtual_positions(40, "off"), torch.arange(40, dtype=torch.float64))
    assert virtual_positions(20).min() >= 0


def test_rope_tokens_matches_upstream_at_integer_positions():
    torch.manual_seed(0)
    freqs = rope_table()
    f, h, w, heads = 2, 3, 4, 2
    x = torch.randn(1, f * h * w, heads, HEAD_DIM)
    start = 5
    frame = torch.arange(f).repeat_interleave(h * w).to(torch.float64) + start
    hh = torch.arange(h).repeat_interleave(w).repeat(f)
    ww = torch.arange(w).repeat(f * h)
    expected = reference_rope(x, (f, h, w), freqs, start)
    actual = rope_tokens(x, frame, hh, ww, freqs)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_rope_tokens_fractional_position_rotates_only_temporal_bands():
    freqs = rope_table()
    x = torch.randn(1, 1, 1, HEAD_DIM)
    zero = torch.zeros(1, dtype=torch.long)
    at3 = rope_tokens(x, torch.tensor([3.0], dtype=torch.float64), zero, zero, freqs)
    at35 = rope_tokens(x, torch.tensor([3.5], dtype=torch.float64), zero, zero, freqs)
    half = HEAD_DIM // 2
    temporal_pairs = half - 2 * (half // 3)
    omega = torch.angle(freqs.split([temporal_pairs, half // 3, half // 3], dim=1)[0][1])
    pairs = torch.view_as_complex(at3.to(torch.float64).reshape(1, 1, half, 2)).clone()
    pairs[..., :temporal_pairs] *= torch.polar(torch.ones_like(omega), 0.5 * omega)
    expected = torch.view_as_real(pairs).flatten(2)[None]
    assert torch.allclose(at35.to(torch.float64), expected, atol=1e-5)


def make_memory(tokens=48, grid=(6, 8), chunks=6, mode="all_bands"):
    memory = object.__new__(DeCoPruneMemory)
    memory.rope = mode
    memory.positions_cache = {}
    memory.token_grid = grid
    memory.archive = {0: {}}
    memory.masks = {}
    torch.manual_seed(1)
    for chunk in range(chunks):
        memory.archive[0][chunk] = {"k": torch.randn(1, 4 * tokens, 2, HEAD_DIM), "v": torch.randn(1, 4 * tokens, 2, HEAD_DIM),
                                    "idx": None}
        if chunk >= 1:
            memory.masks[chunk] = torch.rand(4 * tokens) > 0.7
    return memory


def test_history_prunes_only_chunks_at_or_below_cutoff_and_never_chunk_zero():
    tokens = 48
    memory = make_memory(tokens)
    freqs = rope_table()
    n = 5
    key, value, count = memory._history(0, n, n - 3, freqs, tokens, torch.float32, torch.device("cpu"))
    kept = {c: int(memory.masks[c].sum()) for c in (1, 2)}
    assert count == 4 * tokens + kept[1] + kept[2] + 2 * 4 * tokens  # chunk0, pruned 1-2, whole 3-4
    assert key.shape[1] == value.shape[1] == count
    assert memory.archive[0][0]["idx"] is None
    assert memory.archive[0][3]["idx"] is None and memory.archive[0][4]["idx"] is None
    assert memory.archive[0][1]["idx"].numel() == kept[1]


def test_history_compaction_is_permanent_and_matches_a_later_larger_cutoff():
    tokens = 48
    memory = make_memory(tokens)
    freqs = rope_table()
    memory._history(0, 5, 2, freqs, tokens, torch.float32, torch.device("cpu"))
    _, _, count_clean = memory._history(0, 5, 3, freqs, tokens, torch.float32, torch.device("cpu"))
    kept = sum(int(memory.masks[c].sum()) for c in (1, 2, 3))
    assert count_clean == 4 * tokens + kept + 4 * tokens
    _, _, again = memory._history(0, 5, 2, freqs, tokens, torch.float32, torch.device("cpu"))
    assert again == count_clean  # chunk 3 stays pruned once compacted


def test_history_with_everything_kept_and_real_rope_equals_stock_roped_cache():
    """With no pruning and q0 <= 17, re-roping canonical keys reproduces what stock would have cached."""
    tokens, grid = 48, (6, 8)
    memory = make_memory(tokens, grid, chunks=4, mode="off")
    for chunk in range(1, 4):
        memory.masks[chunk] = torch.ones(4 * tokens, dtype=torch.bool)
    freqs = rope_table()
    key, _, count = memory._history(0, 3, 0, freqs, tokens, torch.float32, torch.device("cpu"))
    assert count == 3 * 4 * tokens
    for chunk in range(3):
        raw = memory.archive[0][chunk]["k"]
        expected = reference_rope(raw, (4, grid[0], grid[1]), freqs, 4 * chunk)
        got = key[:, chunk * 4 * tokens:(chunk + 1) * 4 * tokens]
        assert torch.allclose(got, expected, atol=1e-6)


def test_history_reindexing_changes_only_old_keys_when_history_exceeds_span():
    tokens, grid = 48, (6, 8)
    freqs = rope_table(128)
    shifts = {}
    for mode in ("off", "all_bands"):
        memory = make_memory(tokens, grid, chunks=8, mode=mode)
        for chunk in range(1, 8):
            memory.masks[chunk] = torch.ones(4 * tokens, dtype=torch.bool)
        shifts[mode] = memory._history(0, 7, 0, freqs, tokens, torch.float32, torch.device("cpu"))[0]
    span = 4 * tokens
    assert torch.allclose(shifts["off"][:, -2 * span:], shifts["all_bands"][:, -2 * span:], atol=1e-6)  # last 8 frames
    assert not torch.allclose(shifts["off"][:, :span], shifts["all_bands"][:, :span])
