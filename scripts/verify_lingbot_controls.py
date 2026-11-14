"""Compare neutral hook controls with native interpolation and ray embeddings."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/LingBot-World2"))
import numpy as np
import torch
from einops import rearrange
from wan.utils.cam_utils import interpolate_camera_poses, compute_relative_poses, get_Ks_transformed, get_plucker_embeddings
from worldsync.lingbot_controls import CameraPlan

height, width = 464, 848
raw = np.repeat(np.eye(4)[None], 13, axis=0)
native = interpolate_camera_poses(np.arange(13), raw[:, :3, :3], raw[:, :3, 3], np.linspace(0, 12, 4))
hook = torch.from_numpy(CameraPlan().advance("WAIT")).float()
intrinsics = np.load(ROOT / "external/LingBot-World2/examples/03/intrinsics.npy")[0]
values = []
for pose, count in [(native, 13), (hook, 1)]:
    ks = torch.from_numpy(np.repeat(intrinsics[None], count, axis=0)).float()
    ks = get_Ks_transformed(ks, 480, 832, height, width, height, width)[0].repeat(4, 1)
    relative = compute_relative_poses(pose, framewise=True).cuda()
    embedding = get_plucker_embeddings(relative, ks.cuda(), height, width)
    embedding = rearrange(embedding, "f (h a) (w b) c -> (f h w) (c a b)", a=8, b=8)[None]
    embedding = rearrange(embedding, "b (f h w) c -> b c f h w", f=4, h=height // 8, w=width // 8).to(torch.bfloat16)
    values.append(embedding)
assert torch.equal(*values)
assert values[0].stride() == values[1].stride()
result = {"status": "passed", "scope": "neutral camera input tensor parity only; no generated-world score",
    "shape": list(values[0].shape), "strides": list(values[0].stride()),
    "dtype": str(values[0].dtype), "pixel_resolution": [height, width], "tensor_identical": True}
(ROOT / "reproduction/lingbot-control-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
