"""END guard: compare the player's intended odometry with the reference plan's own end state.

The human reference plan (`action_sequence_steps`) fully determines an expected final commanded
camera state under our fixed control mapping (4 poses per second, 60 deg/s turns, unit steps).
When the player issues END while its intended odometry is far from that end state, the decision
is re-requested with a neutral note. No rubric text is exposed: only the reference plan the
player already receives and its own odometry.
"""
import math
import re

KEYS = {"w": "W", "s": "S", "a": "A", "d": "D", "left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN"}
POSES_PER_SECOND = 4


def parse_steps(steps):
    """['hold(W,7200ms)', 'wait(900ms)'] -> [('W', 7200), ('WAIT', 900)]; unknown steps are skipped."""
    parsed = []
    for step in steps or []:
        match = re.match(r"\s*(hold|wait)\(\s*(?:([A-Za-z]+)\s*,\s*)?(\d+)\s*ms\s*\)", step)
        if not match:
            continue
        kind, key, milliseconds = match.group(1), (match.group(2) or "").lower(), int(match.group(3))
        action = "WAIT" if kind == "wait" else KEYS.get(key)
        if action:
            parsed.append((action, milliseconds))
    return parsed


def expected_state(steps):
    """Simulate the plan with CameraPlan's per-pose arithmetic; return its final commanded state."""
    import numpy as np
    turn = math.radians(60) / POSES_PER_SECOND
    pose, yaw, path = np.eye(4), 0.0, 0.0
    for action, milliseconds in parse_steps(steps):
        step = np.eye(4)
        if action in {"W", "S", "A", "D"}:
            step[:3, 3] = {"W": [0, 0, 1], "S": [0, 0, -1], "A": [-1, 0, 0], "D": [1, 0, 0]}[action]
        elif action in {"LEFT", "RIGHT"}:
            angle = turn if action == "RIGHT" else -turn
            c, s = math.cos(angle), math.sin(angle)
            step[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        elif action in {"UP", "DOWN"}:
            angle = turn if action == "UP" else -turn
            c, s = math.cos(angle), math.sin(angle)
            step[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        for _ in range(max(1, round(milliseconds / 1000 * POSES_PER_SECOND))):
            pose = pose @ step
            path += float(np.linalg.norm(step[:3, 3]))
            if action in {"LEFT", "RIGHT"}:
                yaw += turn if action == "RIGHT" else -turn
    return {"position_normalized_units": pose[:3, 3].tolist(),
            "heading_degrees": math.degrees(math.atan2(pose[0, 2], pose[2, 2])),
            "accumulated_yaw_degrees": math.degrees(yaw), "path_length": path}


def heading_difference(a, b):
    return abs((a - b + 180) % 360 - 180)


def end_deviation(current, steps, heading_tolerance=45.0, leg_fraction=0.15, position_floor=2.0):
    """Conservative check; return a neutral description of a clear mismatch with the plan, else None.

    Hold durations in the reference plan are tuned to other models' speeds, so rotation plans
    and absolute durations are NOT enforced. Only structural facts are checked:
    * translation-only plans keep the original heading (plans without LEFT/RIGHT);
    * translation plans whose own end state is the start position (paired out-and-back holds)
      end near the start position.
    """
    parsed = parse_steps(steps)
    if not parsed:
        return None
    turns = any(action in {"LEFT", "RIGHT", "UP", "DOWN"} for action, _ in parsed)
    expected = expected_state(steps)
    problems = []
    if not turns:
        difference = heading_difference(current["heading_degrees"], 0.0)
        if difference > heading_tolerance:
            problems.append(f"the reference plan has no turns but intended heading is {difference:.0f} degrees from the start")
        longest = max(ms for action, ms in parsed if action != "WAIT") / 1000 * POSES_PER_SECOND if any(a != "WAIT" for a, _ in parsed) else 0
        if math.dist(expected["position_normalized_units"], (0, 0, 0)) < 0.5:
            away = math.dist(current["position_normalized_units"], (0, 0, 0))
            if away > max(position_floor, leg_fraction * longest):
                problems.append(f"the reference plan ends at its start position but intended position is {away:.1f} units away")
    return "; ".join(problems) or None
