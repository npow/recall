import copy
import json

import pytest

from worldsync.pov_video import STATES, prepare, tile, validate_states, viewpoint_text


def test_both_view_plans_share_revision_boundaries_and_opposite_roles(tmp_path):
    for actor in ("orange", "purple"):
        (tmp_path / f"{actor}.png").write_bytes(b"reference copied without editing")
    plan = prepare(tmp_path)
    orange = json.loads((tmp_path / "orange-director.json").read_text())["cases"]["insight:POV001"]
    purple = json.loads((tmp_path / "purple-director.json").read_text())["cases"]["insight:POV001"]
    assert [s[0] for s in orange["scheduled"]] == [s[0] for s in purple["scheduled"]]
    assert plan["expected_frames_per_view"] == 189
    # A view must describe the other actor as companion and its own sleeves in foreground.
    assert "viewer's own orange floral sleeves" in orange["static"]
    assert "companion sitting opposite wears a purple" in orange["static"]
    assert "viewer's own purple floral sleeves" in purple["static"]
    assert "companion sitting opposite wears an orange" in purple["static"]
    # The final shared owner projects differently into each first-person view.
    assert "companion is holding the single basket" in orange["scheduled"][-1][1]
    assert "viewer's purple-sleeved hands are holding" in purple["scheduled"][-1][1]


@pytest.mark.parametrize("field,value", [("revision", 0), ("chunk", 0), ("holder", "both")])
def test_invalid_shared_timeline_is_rejected(field, value):
    states = copy.deepcopy(STATES)
    states[1][field] = value
    with pytest.raises(ValueError):
        validate_states(states)


def test_tiler_rejects_frame_count_or_rate_mismatch(monkeypatch, tmp_path):
    a = {"nb_frames": "189", "avg_frame_rate": "16/1"}
    for b in ({"nb_frames": "188", "avg_frame_rate": "16/1"},
              {"nb_frames": "189", "avg_frame_rate": "24/1"}):
        replies = iter([(a, 11.8125), (b, 11.8125)])
        monkeypatch.setattr("worldsync.pov_video.probe", lambda path: next(replies))
        with pytest.raises(ValueError, match="must match"):
            tile("orange.mp4", "purple.mp4", tmp_path / "tiled.mp4")
        assert not (tmp_path / "tiled.mp4").exists()


def test_unknown_camera_cannot_silently_get_a_valid_agents_view():
    with pytest.raises(ValueError):
        viewpoint_text("observer", STATES[0])
