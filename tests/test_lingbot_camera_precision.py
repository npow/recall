import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from worldsync.lingbot_controls import CameraPlan, fp32_camera_rays


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_turning_camera_rays_match_native_fp32_inside_bf16_denoising():
    source = Path(__file__).resolve().parents[1] / "external/LingBot-World2/wan/utils/cam_utils.py"
    if not source.exists():
        pytest.skip("pinned upstream checkout not installed")
    spec = importlib.util.spec_from_file_location("native_cam_utils", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    camera = CameraPlan()
    camera.advance("WAIT")
    poses = torch.from_numpy(camera.advance("RIGHT")).float()
    relative = module.compute_relative_poses(poses, framewise=True)[-4:].cuda()
    intrinsics = torch.tensor([[50., 50., 32., 32.]], device="cuda").repeat(4, 1)
    expected = module.get_plucker_embeddings(relative, intrinsics, 64, 64).bfloat16()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        old = module.get_plucker_embeddings(relative, intrinsics, 64, 64).bfloat16()
        actual = fp32_camera_rays(module.get_plucker_embeddings, relative, intrinsics, 64, 64).bfloat16()
    assert not torch.equal(old, expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
