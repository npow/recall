"""Report the frozen full1 matrix; withhold overall scores until every case is judged."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def summarize(directory):
    plan = json.loads((directory / 'generation-plan.json').read_text())
    expected_dimensions = {'gc': 48, 'if': 50, 'insight': 30, 'outsight': 43}
    actual_dimensions = {s: sum(c.startswith(s + ':') for c in plan['cases']) for s in expected_dimensions}
    if actual_dimensions != expected_dimensions or plan['seeds'] != [42]:
        raise ValueError('Frozen case matrix differs from full171 benchmark')
    arms = {}
    for arm in plan['arms']:
        dims = {}
        for split, count in expected_dimensions.items():
            records = []
            missing = []
            for case in plan['cases']:
                if not case.startswith(split + ':'):
                    continue
                task = case.split(':')[1]
                identifier = f'{arm}-{split}-{task}-seed42'
                path = directory / 'episodes' / f'{identifier}.json'
                result = json.loads(path.read_text()) if path.exists() else {}
                if result and (result.get('episode_id') != identifier or result.get('task_id') != task):
                    raise ValueError('Mismatched judgment: ' + identifier)
                value = result.get('scores', {}).get('official')
                if result.get('error') or value is None:
                    missing.append(identifier)
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 5:
                    raise ValueError('Invalid score: ' + identifier)
                records.append(result)
            dims[split] = {'expected': count, 'scored': len(records), 'missing_or_error': missing,
                           'mean': sum(r['scores']['official'] for r in records)/count if not missing else None,
                           'gate_verdicts': {v: sum(r.get('gate', {}).get('verdict') == v for r in records)
                                             for v in ['pass', 'partial', 'fail']}}
        complete = all(d['mean'] is not None for d in dims.values())
        arms[arm] = {'complete': complete, 'dimensions': dims,
                     'overall_equal_dimension_mean': sum(d['mean'] for d in dims.values())/4 if complete else None}
    return {'scope': 'Full171 per arm, equal weighting of four dimension means; released partial-admitting rubric',
            'complete': all(a['complete'] for a in arms.values()), 'arms': arms,
            'comparability': 'Integrated open checkpoint with streamed Haiku player and Gemini3.1pro preview via Vertex; not hosted LingBot score reproduction.',
            'memory_claim_gate': 'An official score alone does not certify successful revisits. Inspect returns and landmark consistency separately.'}


if __name__ == '__main__':
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'reproduction/full1-recovered-judging'
    report = summarize(directory)
    (directory / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': report['complete'], 'arms': {k: {
        'scored': sum(d['scored'] for d in a['dimensions'].values()),
        'overall': a['overall_equal_dimension_mean']} for k, a in report['arms'].items()}}))
