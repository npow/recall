"""Report the frozen full1 matrix; withhold overall scores until every case is judged."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def summarize(directory):
    plan = json.loads((directory / 'generation-plan.json').read_text())
    expected_dimensions = {'gc': 48, 'if': 50, 'insight': 30, 'outsight': 43}
    if len(plan['cases']) != len(set(plan['cases'])):
        raise ValueError('Frozen matrix contains duplicate cases')
    if (len(plan['arms']) != len(set(plan['arms'])) or set(plan['arms']) != {'stock', 'budget'}):
        raise ValueError('Frozen matrix must contain exactly stock and budget arms')
    for split in expected_dimensions:
        dataset = json.loads((ROOT / 'external/PlayWorld/data' / split / 'data.json').read_text())
        expected = {f"{split}:{task['task_id']}" for task in dataset}
        actual = {case for case in plan['cases'] if case.startswith(split + ':')}
        if actual != expected:
            raise ValueError('Frozen task membership differs from released split: ' + split)
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
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 5:
                    raise ValueError('Invalid score: ' + identifier)
                records.append(result)
            dims[split] = {'expected': count, 'scored': len(records), 'missing_or_error': missing,
                           'mean': sum(r['scores']['official'] for r in records)/count if not missing else None,
                           'possible_mean_range': [(sum(r['scores']['official'] for r in records)+bound*len(missing))/count
                                                   for bound in [1,5]],
                           'gate_verdicts': {v: sum(r.get('gate', {}).get('verdict') == v for r in records)
                                             for v in ['pass', 'partial', 'fail']},
                           'judge_return_audit': {
                               'required': sum(r.get('gate', {}).get('revisit_required') == 'yes' for r in records),
                               'required_and_reported_yes': sum(r.get('gate', {}).get('revisit_required') == 'yes'
                                   and r.get('gate', {}).get('revisit_observed') == 'yes' for r in records),
                               'high_score_without_required_return': [r['episode_id'] for r in records
                                   if r['scores']['official'] >= 3 and r.get('gate', {}).get('revisit_required') == 'yes'
                                   and r.get('gate', {}).get('revisit_observed') != 'yes'],
                               'caveat': 'Judge self-report, not independent verified return; no replacement official score.'}}
        complete = all(d['mean'] is not None for d in dims.values())
        arms[arm] = {'complete': complete, 'dimensions': dims,
                     'overall_equal_dimension_mean': sum(d['mean'] for d in dims.values())/4 if complete else None,
                     'possible_overall_range': [sum(d['possible_mean_range'][i] for d in dims.values())/4 for i in [0,1]]}
    comparison = None
    if {'stock','budget'} <= arms.keys():
        stock, budget = arms['stock']['possible_overall_range'], arms['budget']['possible_overall_range']
        comparison = {'budget_minus_stock_possible_range': [budget[0]-stock[1],budget[1]-stock[0]],
                      'budget_could_exceed_stock_on_saved_judgments': budget[1] > stock[0]}
    return {'range_interpretation': 'Arithmetic bounds assigning each pending case any valid score1to5. Not imputed official scores, confidence intervals, or claims about judge accuracy.',
            'comparison_bounds': comparison,
            'numerical_target': 2.12,
            'any_arm_could_exceed_target_with_pending_scores': any(a['possible_overall_range'][1] > 2.12 for a in arms.values()),
            'scope': 'Full171 per arm, equal weighting of four dimension means; released partial-admitting rubric',
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
