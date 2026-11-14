"""Expose LingBot's native sequential VAE decoder at latent chunk boundaries.

The native decoder already processes one latent frame at a time with a feature
cache. Keep that same cache across observations instead of decoding each chunk
as a new video or repeatedly decoding the entire history.
"""
import torch


class NativeVaeStream:
    def __init__(self, vae):
        self.vae = vae
        self.model = vae.model
        self.model.clear_cache()

    @torch.no_grad()
    def decode(self, latent):
        z = latent.unsqueeze(0)
        scale = self.vae.scale
        with torch.amp.autocast("cuda", dtype=self.vae.dtype):
            z = z / scale[1].view(1, self.model.z_dim, 1, 1, 1) + scale[0].view(1, self.model.z_dim, 1, 1, 1)
            x = self.model.conv2(z)
            frames = []
            for index in range(z.shape[2]):
                self.model._conv_idx = [0]
                frames.append(self.model.decoder(x[:, :, index:index + 1],
                    feat_cache=self.model._feat_map, feat_idx=self.model._conv_idx))
            return torch.cat(frames, dim=2).float().clamp_(-1, 1).squeeze(0)
