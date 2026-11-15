import math
import sys
import types

import numpy as np
import pytest
import torch

# The memory classes import the upstream attention module lazily; selection and storage do not need it.
from worldsync.lingbot_controls import CameraPlan
from worldsync.lingbot_memv2 import (DiskKVStore, latent_pixel_indices, select_content, select_verified,
                                     thumbnail_descriptor, translation_between)


def plan(actions):
    camera = CameraPlan()
    for action in actions:
        camera.advance(action)
    return np.stack(camera.poses)


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_latent_pixel_indices():
    assert latent_pixel_indices(0, 13) == [0, 4, 8, 12]
    assert latent_pixel_indices(3, 16) == [3, 7, 11, 15]
    with pytest.raises(ValueError):
        latent_pixel_indices(0, 16)


def test_rotation_only_return_is_trusted_without_descriptors():
    actions = ["WAIT", "WAIT"] + ["RIGHT"] * 3 + ["WAIT"] * 3 + ["LEFT"] * 3
    poses = plan(actions)
    first_frame = (len(actions) - 1) * 4
    selected, info = select_verified(poses, {}, set(range(6, first_frame)), first_frame, 6, 10)
    assert selected and min(selected) == 6 and info["trusted_rotation_only"] == len(selected) or len(selected) == 10
    # Earliest copies of the start view are preferred, and nothing in the live tail is used.
    assert max(selected) < first_frame - 8
    heading = [math.degrees(math.atan2(poses[i][0, 2], poses[i][2, 2])) for i in selected]
    assert all(abs(h) <= 60 + 1e-6 for h in heading)


def test_translation_return_needs_content_verification():
    actions = ["WAIT", "WAIT"] + ["W"] * 4 + ["S"] * 4
    poses = plan(actions)
    first_frame = (len(actions) - 1) * 4
    available = set(range(6, first_frame))
    start_view, far_view = unit(1, 0, 0), unit(0, 1, 0)
    # Translation was not visually undone: the current view still looks like the far end.
    # Outbound frames near the start look like the start; everything from mid-way on looks like the far end.
    descriptors = {i: (start_view if i < 16 else far_view) for i in range(first_frame)}
    selected, info = select_verified(poses, descriptors, available, first_frame, 6, 10)
    assert selected == [] and info["rejected_untrusted"] > 0 and info["place_verified"] is False
    # If the view really came back, the same commanded poses are trusted.
    descriptors[first_frame - 1] = start_view
    selected, info = select_verified(poses, descriptors, available, first_frame, 6, 10)
    assert selected and info["trusted_by_content"] == len(selected)


def test_no_pose_match_returns_nothing():
    actions = ["WAIT"] + ["W"] * 9
    poses = plan(actions)
    first_frame = (len(actions) - 1) * 4
    selected, info = select_verified(poses, {}, set(range(6, first_frame)), first_frame, 6, 10)
    assert selected == [] and info["pose_matches"] == 0


def test_content_selection_threshold_and_earliest_tie():
    descriptors = {i: unit(0, 1, 0) for i in range(40)}
    for i in (6, 7, 20):
        descriptors[i] = unit(1, 0, 0)
    descriptors[39] = unit(1, 0.01, 0)
    selected, info = select_content(descriptors, set(range(6, 40)), 40, 6, 2)
    assert selected == [6, 7] and info["above_threshold"] == 3
    descriptors[39] = unit(0, 0, 1)
    assert select_content(descriptors, set(range(6, 40)), 40, 6, 2)[0] == []


def test_translation_between():
    poses = plan(["WAIT", "RIGHT", "W", "LEFT"])
    assert translation_between(poses, 0, 7) == 0
    assert translation_between(poses, 4, 11) == pytest.approx(4)


def test_disk_store_roundtrip(tmp_path):
    store = DiskKVStore(layers=3, root=tmp_path)
    try:
        frames = {}
        for index in (6, 7):
            for layer in range(3):
                key = torch.randn(1, 5, 2, 4).to(torch.bfloat16)
                value = torch.randn(1, 5, 2, 4).to(torch.bfloat16)
                frames[index, layer] = (key, value)
                store.put(index, layer, key, value)
        loaded = store.read(7)
        assert loaded.shape == (3, 2, 5, 2, 4) and loaded.dtype == torch.bfloat16
        for layer in range(3):
            assert torch.equal(loaded[layer, 0], frames[7, layer][0][0])
            assert torch.equal(loaded[layer, 1], frames[7, layer][1][0])
        assert store.bytes_written == 2 * 3 * 2 * 5 * 2 * 4 * 2
        root = store.root
    finally:
        store.close()
    assert not root.exists()


def test_thumbnail_descriptor_is_unit_and_view_sensitive():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (48, 80, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (48, 80, 3), dtype=np.uint8)
    da, db = thumbnail_descriptor(a), thumbnail_descriptor(b)
    assert abs(np.linalg.norm(da) - 1) < 1e-4 and float(da @ da) > float(da @ db) + 0.5
