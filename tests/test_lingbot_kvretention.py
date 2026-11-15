import torch
from worldsync.lingbot_kvretention import pool_tokens, rank_frames


def test_relevance_ranks_matching_frame_first():
    torch.manual_seed(0)
    heads, dim = 4, 8
    frames = [pool_tokens(torch.randn(1, 130, heads * dim), heads) for _ in range(6)]
    query = frames[3] + 0.01 * torch.randn_like(frames[3])
    score = rank_frames(query, torch.stack(frames))
    assert int(score.argmax()) == 3
    assert frames[0].shape == (64, heads, dim)
