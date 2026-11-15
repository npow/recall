import numpy as np
import pytest

from worldsync.lingbot_keyframe_restart import (KeyframeBank, NATIVE_FRAMES_PER_CHUNK, crossfade, ncc,
                                                pose_distance, segment_poses)
from worldsync.lingbot_paths import PATHS, native_frame_poses


def frame(value, shape=(8, 14, 3)):
    return np.full(shape, value, dtype=np.uint8)


def pose_at(native_poses, native_index):
    return native_poses[native_index]


def test_pose_distance_zero_for_identical_and_90_degrees_for_turn():
    a = np.eye(4)
    b = np.eye(4)
    b[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    assert pose_distance(a, a) == (0.0, 0.0)
    translation, angle = pose_distance(a, b)
    assert translation == 0 and angle == pytest.approx(90.0)


def test_native_frame_count_matches_chunk_accounting():
    for name in ("turn_return", "turn_hold_return", "translate_return"):
        poses, _ = native_frame_poses(PATHS[name], turn_degrees_per_second=68)
        assert len(poses) == 16 * len(PATHS[name]) - 3


def test_turn_return_triggers_only_at_final_wait_chunk():
    actions = PATHS["turn_return"]
    poses, _ = native_frame_poses(actions, turn_degrees_per_second=60)
    bank = KeyframeBank()
    bank.add(0, poses[0], frame(10), "start")
    for chunk in range(1, len(actions)):
        native_now = 16 * chunk - 4              # last emitted native frame before chunk `chunk`
        bank.add(native_now, poses[native_now], frame(chunk * 20), f"chunk{chunk - 1}-end")
    fired = [chunk for chunk in range(1, len(actions))
             if bank.find(16 * chunk - 4, poses[16 * chunk - 4]) is not None]
    # RIGHT x3 then LEFT x3: pose is back at the start only before the final WAIT chunk (index 7)
    assert fired == [7]


def test_gap_requirement_excludes_recent_keyframes():
    poses, _ = native_frame_poses(["WAIT"] * 6, turn_degrees_per_second=60)
    bank = KeyframeBank()
    bank.add(0, poses[0], frame(1), "start")
    assert bank.find(16 * 3 - 4, poses[16 * 3 - 4]) is None     # 44 native frames: still inside the window
    assert bank.find(16 * 6 - 4, poses[16 * 6 - 4]) is not None  # 92 native frames: outside


def test_cooldown_blocks_immediate_retrigger():
    poses, _ = native_frame_poses(["WAIT"] * 14, turn_degrees_per_second=60)
    bank = KeyframeBank()
    bank.add(0, poses[0], frame(1), "start")
    native_now = 16 * 8 - 4
    keyframe = bank.find(native_now, poses[native_now])
    bank.record_restart(8, native_now, keyframe, frame(2))
    assert bank.find(native_now + 1, poses[native_now + 1]) is None
    assert bank.find(native_now + 16 * 5, poses[native_now + 16 * 5]) is not None


def test_heading_threshold_rejects_offset_view():
    actions = ["WAIT", "RIGHT", "RIGHT", "WAIT"]
    poses, _ = native_frame_poses(actions, turn_degrees_per_second=60)
    bank = KeyframeBank(min_gap_native=16)
    bank.add(0, poses[0], frame(1), "start")
    assert bank.find(16 * 3 - 4, poses[16 * 3 - 4]) is None      # 120 degrees away


def test_drift_threshold_skips_restart_when_view_still_matches():
    poses, _ = native_frame_poses(["WAIT"] * 8, turn_degrees_per_second=60)
    keyframe_pixels = np.random.default_rng(0).integers(0, 255, (64, 112, 3), dtype=np.uint8)
    bank = KeyframeBank(drift_threshold=0.9)
    bank.add(0, poses[0], keyframe_pixels, "start")
    native_now = 16 * 7 - 4
    assert bank.find(native_now, poses[native_now], keyframe_pixels) is None             # identical view: no restart
    other = np.random.default_rng(1).integers(0, 255, (64, 112, 3), dtype=np.uint8)
    assert bank.find(native_now, poses[native_now], other) is not None                   # drifted view: restart


def test_ncc_identity_and_inverse():
    image = np.random.default_rng(2).integers(0, 255, (64, 112, 3), dtype=np.uint8)
    assert ncc(image, image) == pytest.approx(1.0, abs=1e-5)
    assert ncc(image, 255 - image) == pytest.approx(-1.0, abs=1e-3)


def test_segment_pose_track_gives_exactly_remaining_chunks():
    for name, restart_chunk in (("turn_return", 7), ("turn_hold_return", 13), ("translate_return", 9)):
        actions = PATHS[name]
        poses, _ = native_frame_poses(actions, turn_degrees_per_second=60)
        native_now = 16 * restart_chunk - 4
        for aligned, offset in ((True, 4), (False, 0)):
            track = segment_poses(poses, native_now, aligned)
            latent = (len(track) - 1) // 4 + 1
            assert latent - latent % 4 == 4 * (len(actions) - restart_chunk)   # pipeline rounds to whole chunks
            assert np.allclose(track[0], poses[native_now + offset])


def test_crossfade_ramps_towards_new_frames():
    previous = frame(0)
    new = np.stack([frame(200)] * 6)
    blended = crossfade(previous, new, 3)
    assert [int(f.mean()) for f in blended[:4]] == [50, 100, 150, 200]
    assert crossfade(previous, new, 0) is new


def test_excursion_rule_skips_hold_and_fires_on_real_revisits():
    from worldsync.lingbot_keyframe_restart import excursion
    actions = PATHS["turn_hold_return"]
    poses, _ = native_frame_poses(actions, turn_degrees_per_second=68)
    bank = KeyframeBank(min_excursion_degrees=30.0, min_excursion_position=2.0)
    bank.add(0, poses[0], frame(1), "start")
    for chunk in range(1, len(actions)):
        native_now = 16 * chunk - 4
        bank.add(native_now, poses[native_now], frame(chunk), f"chunk{chunk - 1}-end")
    fired = [chunk for chunk in range(1, len(actions))
             if bank.find(16 * chunk - 4, poses[16 * chunk - 4], native_poses=poses) is not None]
    # with the excursion rule nothing fires during the far-heading hold; the first legitimate revisit is the
    # outbound view at the same heading while sweeping back (chunk 11), then the true return (chunk 13)
    assert fired == [11, 12, 13]
    # without the rule the first trigger is the far-heading hold (the v1 behaviour that blocked the real return)
    plain = KeyframeBank()
    for keyframe in bank.frames:
        plain.add(keyframe.native_index, keyframe.pose, keyframe.pixels, keyframe.label)
    first = next(chunk for chunk in range(1, len(actions)) if plain.find(16 * chunk - 4, poses[16 * chunk - 4]) is not None)
    assert first == 9
    moved, turned = excursion(poses, 0, 16 * 13 - 4, poses[0])
    assert turned > 150 and moved == 0.0


def test_native_index_bookkeeping_both_schemes():
    from worldsync.lingbot_keyframe_restart import native_last_index
    # no restart: chunk 0 emits 13 frames, later chunks 16 -> last native index = frames - 1
    assert native_last_index(13, 0) == 12 and native_last_index(13 + 16 * 3, 0) == 12 + 16 * 3
    # restart at chunk g=7 (16g-3 = 109 frames emitted), scheme B: chunk 0' adds 13 frames and ends on native 16g+12
    assert native_last_index(109 + 13, 1, True) == 16 * 7 + 12
    assert native_last_index(109 + 13 + 16, 1, True) == 16 * 8 + 12
    # scheme A lags one latent frame (4 native frames: the conditioning frame at 16g-4, 12 new frames to 16g+8)
    assert native_last_index(109 + 13, 1, False) == 16 * 7 + 8
