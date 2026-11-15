import importlib.util
from pathlib import Path

import numpy as np
import torch

from worldsync.camera_conditioning import plan_translation_scale, fixed_scale_relative_poses
from worldsync.lingbot_controls import CameraPlan
from worldsync.native_reference_timeline import NativeReferenceTimeline, apply_camera_steps

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_cam_utils', ROOT / 'external/LingBot-World2/wan/utils/cam_utils.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_streamed_conditioning_matches_full_native_trajectory_and_closes_rectangle():
    actions = [{'type': 'hold', 'keys': [key], 'duration_ms': 5400}
               for key in ['KeyW', 'KeyD', 'KeyS', 'KeyA']]
    timeline = NativeReferenceTimeline(actions, '')
    scale = plan_translation_scale(timeline)
    camera = CameraPlan()
    fixed, legacy = [], []
    for chunk in range(23):
        steps = ([{'action': 'WAIT', 'turn_degrees': 0}] * 3 if chunk == 0 else timeline.take())
        poses = torch.from_numpy(apply_camera_steps(camera, steps)).float()
        fixed.append(fixed_scale_relative_poses(native.compute_relative_poses, poses, scale)[-4:])
        legacy.append(native.compute_relative_poses(poses, framewise=True)[-4:])
    streamed = torch.cat(fixed)
    full = native.compute_relative_poses(poses, framewise=True)
    torch.testing.assert_close(streamed, full, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(torch.cat(legacy), full)
    assert np.allclose(camera.poses[-1], np.eye(4))
    # Verify the actual six-channel camera embeddings, not only bookkeeping.
    intrinsics = torch.tensor([[4., 4., 2., 2.]]).repeat(len(full), 1)
    torch.testing.assert_close(native.get_plucker_embeddings(streamed, intrinsics, 4, 4),
                               native.get_plucker_embeddings(full, intrinsics, 4, 4))
    composed = torch.eye(4)
    for step in streamed:
        composed = composed @ step
    torch.testing.assert_close(composed, torch.eye(4), atol=3e-6, rtol=0)


def test_early_release_and_extension_do_not_change_committed_conditioning():
    actions = [{'type': 'hold', 'keys': [key], 'duration_ms': duration}
               for key, duration in [('KeyW', 5400), ('KeyS', 1800)]]
    timeline = NativeReferenceTimeline(actions, '')
    scale = plan_translation_scale(timeline)
    camera = CameraPlan()
    apply_camera_steps(camera, [{'action': 'WAIT', 'turn_degrees': 0}]*3)
    first = torch.from_numpy(apply_camera_steps(camera, timeline.take())).float()
    committed = fixed_scale_relative_poses(native.compute_relative_poses, first, scale)
    timeline.apply_decision({'release_now': True, 'adaptive_direction': None, 'extend_ms': 0})
    timeline.apply_decision({'release_now': False, 'adaptive_direction': None, 'extend_ms': 1000})
    for _ in range(4):
        expanded = torch.from_numpy(apply_camera_steps(camera, timeline.take())).float()
    recomputed = fixed_scale_relative_poses(native.compute_relative_poses, expanded, scale)
    torch.testing.assert_close(committed, recomputed[:len(committed)], atol=0, rtol=0)
