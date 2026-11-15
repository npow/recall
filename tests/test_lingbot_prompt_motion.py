import torch

from worldsync.lingbot_prompt_motion import PromptSwitcher, gain_compute_relative_poses, prompt_for


def test_off_is_stock_caption():
    assert prompt_for("A mall.", "off", "W") == "A mall."


def test_perchunk_flags_only_changes():
    s = PromptSwitcher("A mall.", "perchunk", ["WAIT", "W", "W", "RIGHT"])
    s.contexts = {t: [torch.zeros(1)] for t in s.texts.values()}
    flags = [s.for_chunk(i, a)[1] for i, a in enumerate(["WAIT", "W", "W", "RIGHT"])]
    assert flags == [False, True, False, True]
    assert [x["chunk"] for x in s.switches] == [1, 3]


def test_static_never_switches():
    s = PromptSwitcher("A mall.", "static", ["WAIT", "W", "RIGHT"])
    s.contexts = {t: [torch.zeros(1)] for t in s.texts.values()}
    assert not any(s.for_chunk(i, a)[1] for i, a in enumerate(["WAIT", "W", "RIGHT"]))


def test_gain_scales_translation_only():
    def original(poses, **kw):
        out = poses.clone()
        return out
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, :3, 3] = 1.0
    out = gain_compute_relative_poses(original, 2.0)(poses)
    assert torch.allclose(out[:, :3, 3], torch.full((3, 3), 2.0))
    assert torch.equal(out[:, :3, :3], poses[:, :3, :3])
    assert gain_compute_relative_poses(original, 1.0) is original
