from pathlib import Path

import numpy as np

from worldsync.lingbot_released_baseline import BaselinePlayer, load_task
from worldsync.lingbot_controls import CameraPlan
from worldsync.native_reference_timeline import apply_camera_steps


class HoldService:
    def __init__(self):
        self.calls = []

    def decide(self, *args, **kwargs):
        raise AssertionError('Legacy HY scheduler must never execute')

    def check_hold(self, client, task, actions, phase, elapsed, extension, latest):
        self.calls.append((phase, elapsed, extension))
        return {'release_now': False, 'adaptive_direction': None, 'extend_ms': 0}, {'player_transport': 'test'}


def run_player(task, chunks):
    service = HoldService()
    player = BaselinePlayer(service)
    history = []
    camera = CameraPlan()
    scales = []
    states = []
    for _ in range(chunks):
        decision, metadata = player.decide(None, task, Path('initial'), Path('latest'), history)
        apply_camera_steps(camera, metadata['native_camera_steps'])
        history.append(decision)
        scales.append(metadata['translation_scale'])
        states.append(camera.commanded_state())
    return player, service, camera, scales, states


def test_gc010_uses_canonical_timing_and_returns_without_hy_padding():
    player, service, camera, scales, states = run_player(load_task('gc', 'GC010'), 30)
    assert player.timeline.current() is None
    assert max(s['accumulated_yaw_degrees'] for s in states) <= 90 + 1e-6
    assert abs(camera.commanded_state()['accumulated_yaw_degrees']) < 1e-6
    assert np.allclose(camera.poses[-1], np.eye(4))
    assert len(set(scales)) == 1
    assert service.calls == []


def test_gc002_uses_hold_service_with_original_phase_indices():
    player, service, camera, scales, states = run_player(load_task('gc', 'GC002'), 40)
    assert player.timeline.current() is None
    assert service.calls
    assert player.requests == len(service.calls)
    assert all(0 <= phase < len(player.actions) for phase, _, _ in service.calls)
    assert np.allclose(camera.poses[-1], np.eye(4))
    assert len(set(scales)) == 1
