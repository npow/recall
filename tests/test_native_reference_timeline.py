import numpy as np
import pytest

from worldsync.lingbot_controls import CameraPlan
from worldsync.native_reference_timeline import NativeReferenceTimeline, apply_camera_steps


def hold(key, duration):
    return {'type': 'hold', 'keys': [key], 'duration_ms': duration}


def test_actual_released_response_schema_extends_and_handles_no_correction():
    timeline = NativeReferenceTimeline([hold('KeyW', 1000)], '')
    timeline.apply_decision({'release_now': False, 'adaptive_direction': None, 'extend_ms': 1500})
    assert len([s for s in timeline.take(11) if s['action'] == 'W']) == 10


def test_both_corrections_execute_in_order_before_next_phase():
    timeline = NativeReferenceTimeline([hold('KeyW', 1000), hold('KeyS', 1000)], '')
    for direction in ['ArrowLeft', 'ArrowUp']:
        timeline.apply_decision({'release_now': False, 'adaptive_direction': direction, 'extend_ms': 0})
    assert [s['action'] for s in timeline.take(16)] == ['W']*4 + ['LEFT']*4 + ['UP']*4 + ['S']*4


def test_quarter_schedule_has_symmetric_translation_after_neutral_bootstrap():
    timeline = NativeReferenceTimeline([hold('KeyW', 1800), hold('KeyS', 1800)], '')
    camera = CameraPlan()
    apply_camera_steps(camera, [{'action': 'WAIT', 'turn_degrees': 0}]*3)
    for _ in range(4):
        apply_camera_steps(camera, timeline.take())
    # Cumulative nearest-boundary quantization gives7forward and7backward steps.
    assert np.allclose(camera.poses[-1], np.eye(4))


def test_explicit_return_angle_command_does_not_use_hy_rotation_padding():
    timeline = NativeReferenceTimeline([hold('ArrowRight', 1800), hold('ArrowLeft', 1800)],
                                       'Turn90 degrees right and return to the original view')
    camera = CameraPlan()
    apply_camera_steps(camera, [{'action': 'WAIT', 'turn_degrees': 0}]*3)
    headings = []
    for step in timeline.take(14):
        # Test individual transforms through full native chunks below.
        headings.append(step['turn_degrees'] * (1 if step['action'] == 'RIGHT' else -1))
    assert max(np.cumsum(headings)) == pytest.approx(90)
    assert sum(headings) == pytest.approx(0)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, True])
def test_bad_camera_step_rejected_before_mutating_pose(bad):
    camera = CameraPlan()
    with pytest.raises(ValueError):
        apply_camera_steps(camera, [{'action': 'RIGHT', 'turn_degrees': bad}]*3)
    assert camera.poses == []


def test_invalid_released_decision_rejected():
    timeline = NativeReferenceTimeline([hold('KeyW', 1000)], '')
    with pytest.raises(ValueError):
        timeline.apply_decision({'release_now': 'false', 'adaptive_direction': None, 'extend_ms': 0})


def test_fractional_boundary_does_not_change_rectangle_closure():
    timeline = NativeReferenceTimeline([hold('KeyW', 5400), hold('KeyD', 5400),
                                       hold('KeyS', 5400), hold('KeyA', 5400)], '')
    camera = CameraPlan()
    apply_camera_steps(camera, [{'action': 'WAIT', 'turn_degrees': 0}]*3)
    for _ in range(22):
        apply_camera_steps(camera, timeline.take())
    assert np.allclose(camera.poses[-1], np.eye(4))


def test_unspecified_angle_preserves_duration_despite_quantization():
    timeline = NativeReferenceTimeline([hold('KeyW', 1800), hold('ArrowRight', 3600)], '')
    assert sum(s['turn_degrees'] for s in timeline.take(22)) == pytest.approx(180)


def test_early_release_defers_until_minimum_hold_and_never_adds_extension():
    timeline = NativeReferenceTimeline([hold('KeyW', 5000)], '')
    timeline.take(1)
    timeline.apply_decision({'release_now': True, 'adaptive_direction': None, 'extend_ms': 5000})
    assert timeline.current()['extension'] == 0
    assert [s['action'] for s in timeline.take(4)] == ['W', 'W', 'W', 'WAIT']


def test_ambiguous_explicit_angles_are_not_silently_replaced_by_nominal_rate():
    with pytest.raises(ValueError, match='unambiguously'):
        NativeReferenceTimeline([hold('ArrowRight', 1800), hold('ArrowUp', 800), hold('ArrowLeft', 1800)],
                                'Turn90 degrees right, look up40 degrees, then return')


def test_corrected_profile_cannot_launch_unvalidated_scheduler(tmp_path):
    from worldsync.lingbot_released_baseline import preflight, run
    assert preflight('gc', 'GC010')['status'] == 'blocked_controller_integration'
    with pytest.raises(ValueError, match='preflight failed'):
        run('gc', 'GC010', tmp_path / 'blocked')
