import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from worldsync.motion_feedback import MotionLog, chunk_motion, feedback_note, image_motion, pair_motion

INTRINSICS = np.array([[420.0, 0, 208.0], [0, 420.0, 120.0], [0, 0, 1.0]])


def textured(seed=0, width=416, height=240):
    """A real photograph (feature-rich); synthetic noise gives too few SIFT features."""
    from pathlib import Path
    names = ["GC007.jpg", "GC001.jpg", "GC004.jpg"]
    root = Path(__file__).resolve().parents[1] / "external/PlayWorld/data/gc/images"
    path = root / names[seed % len(names)]
    if not path.exists():
        pytest.skip("PlayWorld benchmark images not available")
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (width, height))


def yawed(image, degrees):
    """Render the scene after the camera yaws right by `degrees` (pure rotation homography)."""
    angle = -math.radians(degrees)  # points rotate opposite to a camera turning right
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # camera-b = R camera-a in image coordinates
    homography = INTRINSICS @ rotation @ np.linalg.inv(INTRINSICS)
    return cv2.warpPerspective(image, homography, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_yaw_sign_and_magnitude_on_synthetic_rotation():
    image = textured(1)
    for degrees in (8.0, -8.0):
        measured = pair_motion(image, yawed(image, degrees), INTRINSICS)
        assert measured["usable"]
        # Positive yaw is a right turn; allow a few degrees of fit error on synthetic texture.
        assert abs(abs(measured["yaw_degrees"]) - 8.0) < 3.0
    right = pair_motion(image, yawed(image, 8.0), INTRINSICS)["yaw_degrees"]
    assert right > 0  # positive = right turn, matching the real-video check
    left = pair_motion(image, yawed(image, -8.0), INTRINSICS)["yaw_degrees"]
    assert right * left < 0


def test_static_scene_reports_no_motion():
    image = textured(2)
    frames = np.stack([image] * 16)
    measure = chunk_motion(frames, INTRINSICS)
    assert measure["image_motion_px_per_second"] < 0.5
    assert measure.get("yaw_degrees", 0.0) == 0.0 or "yaw_degrees" not in measure


def test_chunk_yaw_accumulates_and_flags_unreliable():
    image = textured(3)
    # 16 frames sweeping 12 degrees in total
    frames = np.stack([yawed(image, 12.0 * i / 15) for i in range(16)])
    measure = chunk_motion(frames, INTRINSICS)
    assert measure["reliable_pairs"] >= 2
    assert abs(measure["yaw_degrees"] - 12.0) < 5.0
    noise = np.random.default_rng(0).integers(0, 255, (16, 240, 416, 3), dtype=np.uint8)
    assert "yaw_degrees" not in chunk_motion(noise, INTRINSICS)


def test_motion_log_summary_marks_unavailable_not_zero():
    log = MotionLog(INTRINSICS)
    noise = np.random.default_rng(1).integers(0, 255, (16, 240, 416, 3), dtype=np.uint8)
    log.add(noise, "RIGHT", 60.0)
    summary = log.summary()
    assert summary["chunks_without_yaw_measurement"] == 1
    assert summary["recent_chunks"][0]["measured_yaw_degrees_this_chunk"] == "unavailable"
    assert "rubric" not in json.dumps(summary).lower()


def test_feedback_note_fields():
    note = feedback_note({"pairs": 4, "reliable_pairs": 3, "yaw_degrees": 51.234, "image_motion_px_per_second": 250.0},
                         "RIGHT", 60.0)
    assert note["measured_yaw_degrees_this_chunk"] == 51.2 and note["reliable_pairs"] == "3/4"


# The player-prompt integration test (MEASURED_MOTION_HINT in reference_player) is not ported: main's Player v6
# differs; see reproduction/INTEGRATION.md. The branch worktree-agent-a4a049746f1d62929 has the original.
