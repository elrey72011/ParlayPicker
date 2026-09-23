"""Explicit forward evidence job; never activates policy or places wagers."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _plans(database, plan_dir):
    """Use a unique immutable Drive-restored plan when no local file exists."""
    from app_core.prediction_evidence import connect
    from core.sport_policy import SPORTS

    with closing(connect(database)) as db:
        receipts = db.execute('SELECT sport,payload FROM validation_plans').fetchall()
    by_sport = {sport: [] for sport in SPORTS}
    for sport, payload in receipts:
        if sport in by_sport:
            by_sport[sport].append(json.loads(payload))
    plans = {}
    for sport in SPORTS:
        path = Path(plan_dir, sport + '.json')
        if path.exists():
            plans[sport] = json.loads(path.read_text(encoding='utf-8'))
        elif len(by_sport[sport]) == 1:
            plans[sport] = by_sport[sport][0]
        elif len(by_sport[sport]) > 1:
            raise ValueError(f'Multiple frozen plans for {sport}; provide an explicit plan file')
        else:
            plans[sport] = None
    return plans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--database', default='data/prediction_evidence/evidence.sqlite3')
    p.add_argument('--plan-dir', default='data/validation_plans')
    p.add_argument('--output', default='output/sport-validation')
    p.add_argument('--restore', action='store_true')
    p.add_argument('--capture-closes', action='store_true')
    p.add_argument('--grade', action='store_true')
    p.add_argument('--skip-validation', action='store_true')
    p.add_argument('--backup', action='store_true')
    a = p.parse_args()

    def stage(name, operation):
        print(json.dumps({'stage': name, 'state': 'started'}), flush=True)
        started = time.monotonic()
        try:
            value = operation()
        except Exception as exc:
            print(json.dumps({'stage': name, 'state': 'failed', 'error_type': type(exc).__name__,
                              'seconds': round(time.monotonic() - started, 3)}), flush=True)
            raise
        print(json.dumps({'stage': name, 'state': 'completed',
                          'seconds': round(time.monotonic() - started, 3)}), flush=True)
        return value

    result = {}
    if a.restore:
        from app_core.evidence_remote import restore
        result['restored_snapshots'] = stage('restore', lambda: restore(a.database))

    def backup(name):
        from app_core.evidence_remote import remote_status, sync
        if not stage(name, lambda: sync(a.database, incremental=a.restore)):
            raise RuntimeError('Evidence backup was not verified')
        result['remote'] = remote_status()

    if a.capture_closes:
        from app_core.activation_closing import capture_live
        result['closing'] = stage('capture_closes', lambda: capture_live(a.database))
        if a.backup and result['closing']['verified']:
            backup('backup_closes')
    if a.grade:
        from app_core.prediction_evidence import refresh_outcomes
        result['grading'] = stage('refresh_outcomes', lambda: refresh_outcomes(a.database))
        if a.backup and result['grading']['revisions']:
            backup('backup_scores')
        from core.ranking_evidence_rebuild import rebuild
        result['ranking_evidence'] = stage(
            'ranking_evidence', lambda: rebuild(a.database, Path(a.output) / 'ranking-evidence.json'))

    if not a.skip_validation:
        from core.activation_validation import validate
        from core.sport_policy import SPORTS
        plans = stage('load_frozen_plans', lambda: _plans(a.database, a.plan_dir))
        output = Path(a.output)
        output.mkdir(parents=True, exist_ok=True)
        for sport in SPORTS:
            report = stage('validate_' + sport, lambda: validate(a.database, sport, plans[sport]))
            (output / (sport + '.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
            (output / (sport + '.md')).write_text('# ' + sport + '\\n\\n' + json.dumps(report, indent=2), encoding='utf-8')
            result[sport] = {'state': report['deployment_state'], 'blockers': report['blockers']}
    if a.backup:
        backup('backup_final')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
