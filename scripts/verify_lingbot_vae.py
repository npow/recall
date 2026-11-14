"""Check streaming observation decoding against the actual stock VAE on CUDA."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/LingBot-World2"))

import torch
from wan.modules.vae2_1 import Wan2_1_VAE
from worldsync.lingbot_vae import NativeVaeStream

checkpoint = ROOT / "external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast/Wan2.1_VAE.pth"
vae = Wan2_1_VAE(vae_pth=str(checkpoint), device="cuda")
torch.manual_seed(42)
z = torch.randn(16, 8, 8, 8, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    reference = vae.decode([z])[0]
    stream = NativeVaeStream(vae)
    first = stream.decode(z[:, :4])
    second = stream.decode(z[:, 4:])
    actual = torch.cat([first, second], dim=1)
    maximum = (actual - reference).abs().max().item()
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
result = {"status": "passed", "scope": "native VAE observation stream parity; not world or score reproduction",
    "latent_shape": list(z.shape), "native_decoded_frames": reference.shape[1],
    "stream_chunk_frames": [first.shape[1], second.shape[1]], "max_abs_error": maximum,
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
(ROOT / "reproduction/lingbot-vae-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
