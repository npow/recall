import torch

from worldsync.lingbot_nag import nag_combine


def test_nag_identity_when_negative_equals_positive():
    z = torch.randn(2, 5, 8)
    assert torch.allclose(nag_combine(z, z.clone(), scale=8.0, tau=2.5, alpha=0.25), z)


def test_nag_norm_is_clipped_and_blended():
    torch.manual_seed(0)
    z_pos, z_neg = torch.randn(1, 4, 16), torch.randn(1, 4, 16) * 5
    out = nag_combine(z_pos, z_neg, scale=8.0, tau=2.5, alpha=1.0)
    ratio = out.abs().sum(-1) / z_pos.abs().sum(-1)
    assert (ratio <= 2.5 + 1e-4).all()
    blended = nag_combine(z_pos, z_neg, scale=8.0, tau=2.5, alpha=0.25)
    assert torch.allclose(blended, 0.25 * out + 0.75 * z_pos, atol=1e-5)
