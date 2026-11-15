import pytest

from worldsync.canvas import Canvas, CanvasConfig, TileState
from worldsync.poses import CameraRig, actions_to_latent_steps, parse_actions


def latent_poses(text, chunk=4):
    rig = CameraRig()
    poses = [rig.pose.copy()] + [rig.advance(k, s) for k, s in actions_to_latent_steps(parse_actions(text))]
    while len(poses) % chunk:
        poses.append(rig.advance(()))
    return poses


def run(text, config=CanvasConfig()):
    canvas = Canvas(config)
    events = [e for i, pose in enumerate(latent_poses(text)) for e in canvas.update(pose, i // 4)]
    return canvas, [(e.kind, e.tile) for e in events]


def test_turn_away_and_back_forks_then_hands_off_the_original_view():
    canvas, events = run("hold(Left,1800ms); wait(900ms); hold(Right,1800ms); wait(900ms)")
    assert events[0] == ("create", 0)
    fork, handoff = events.index(("fork", 0)), events.index(("handoff", 0))
    assert fork < handoff
    assert canvas.tiles[0].state is TileState.LIVE
    final = latent_poses("hold(Left,1800ms); wait(900ms); hold(Right,1800ms); wait(900ms)")[-1]
    for tile in canvas.tiles:  # states respect the hysteresis band at the final view
        seen = canvas.visibility(tile.pose, final)
        assert seen >= canvas.config.leave_below if tile.state is TileState.LIVE else seen < canvas.config.enter_above


def test_full_turn_builds_a_ring_and_returns_to_the_first_tile():
    canvas, events = run("hold(Right,7200ms); wait(450ms)")
    assert 4 <= len(canvas.tiles) <= 10
    rig = CameraRig()
    for _ in range(72):  # every heading around the ring is mostly covered by stored tiles
        assert canvas.coverage(rig.advance(("RIGHT",), seconds=0.1)) >= canvas.config.create_below - 0.05
    assert ("handoff", 0) in events and canvas.tiles[0].state is TileState.LIVE


def test_walking_forward_reuses_tiles_until_parallax_changes_the_view():
    canvas, _ = run("hold(W,1000ms)")
    assert len(canvas.tiles) == 1
    canvas, _ = run("hold(W,6000ms)")
    assert 1 < len(canvas.tiles) < 6


def test_visibility_is_symmetric_for_pure_rotation():
    canvas = Canvas()
    rig = CameraRig()
    a = rig.pose.copy()
    b = rig.advance(("LEFT",), seconds=0.9)
    assert canvas.visibility(a, a) == 1.0
    assert canvas.visibility(a, b) == pytest.approx(canvas.visibility(b, a), abs=0.07)
    assert 0.3 < canvas.visibility(a, b) < 0.7


def test_hysteresis_is_required():
    with pytest.raises(ValueError):
        CanvasConfig(leave_below=0.3, enter_above=0.2)
