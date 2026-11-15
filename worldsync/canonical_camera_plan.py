"""Build camera plans, with explicit audited bindings for multi-return objectives."""
import hashlib
import json
from pathlib import Path

from worldsync.native_reference_timeline import NativeReferenceTimeline
from worldsync.released_reference_player import canonical_actions

BINDINGS = Path(__file__).resolve().parents[1] / 'reproduction/baseline-gap/explicit-angle-bindings.json'


def build_plan(task):
    actions = canonical_actions(task)
    binding = json.loads(BINDINGS.read_text()).get(task['task_id'])
    angles = None
    if binding:
        payload = {'prompt': task['prompt'], 'actions': actions}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != binding['fingerprint']:
            raise ValueError('Audited angle binding no longer matches the released task')
        angles = {int(index): value for index, value in binding['phase_degrees'].items()}
    return actions, NativeReferenceTimeline(actions, task['prompt'], phase_angles=angles)
