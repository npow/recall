"""Audit saved top-level judgments for gate contradictions and high scores without a revisit.

Reports judge statements, not independently measured trajectory success. Does not
change the released score or exclude episodes. No API calls.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for path in sorted((ROOT/'reproduction').rglob('*.json')):
        if any(part in ('cache','judge-cache') for part in path.parts):
            continue
        try:
            item = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(item,dict) or not isinstance(item.get('gate'),dict):
            continue
        gate = item['gate']
        if gate.get('verdict') not in ('pass','partial','fail'):
            continue
        score = (item.get('scores') or {}).get('official')
        rows.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'score':score,'verdict':gate['verdict'],'numeric_gate':gate.get('instruction_following_score'),
                     'gate_fields_consistent':gate.get('instruction_following_score')==(0 if gate['verdict']=='fail' else 1),
                     'revisit_required':gate.get('revisit_required'),'judge_reports_revisit':gate.get('revisit_observed'),
                     'judge_reports_completion':((item.get('rubric') or {}).get('requested_action_completion') or {}).get('completed')})
    flagged = [row for row in rows if isinstance(row['score'],(int,float)) and row['score']>=3
               and row['revisit_required']=='yes' and row['judge_reports_revisit']=='no']
    report = {'scope':'Saved top-level judgment audit; not unique benchmark cases or independent success measurements',
              'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'reports':len(rows),'gate_contradictions':[r for r in rows if not r['gate_fields_consistent']],
              'score_at_least3_without_judge_reported_revisit':flagged,'rows':rows,
              'policy':'Retain released scores. Require independent trajectory and landmark evidence before a memory-success claim.'}
    (ROOT/'reproduction/completion-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'reports':len(rows),'gate_contradictions':len(report['gate_contradictions']),
                      'high_score_no_revisit':len(flagged)},indent=2))


if __name__=='__main__':
    main()
