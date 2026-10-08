"""Reconcile original local schedule inventories, without acquisition or databases."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from app_core.slate_coverage import build_coverage, native_ncaaf, validate_report, MARKETS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, action='append', default=[])
    parser.add_argument('--date', required=True, help='Explicit Eastern YYYY-MM-DD')
    parser.add_argument('--as-of', required=True, help='Explicit aware clock')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--final', type=Path)
    parser.add_argument('--providers', type=Path)
    parser.add_argument('--gate-audit', type=Path)
    parser.add_argument('--provider-health', type=Path)
    parser.add_argument('--policy-exclusions', type=Path)
    parser.add_argument('--required-market', action='append', choices=MARKETS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(path.read_bytes()) if path else []
    def frame(path):
        return pd.read_csv(path) if path else pd.DataFrame()
    sources = [*args.inventory, *(p for p in (args.candidates, args.final, args.providers, args.gate_audit, args.provider_health, args.policy_exclusions) if p)]
    targets = [args.output.with_suffix('.json'), args.output.with_suffix('.csv')]
    if {p.resolve() for p in targets} & {p.resolve() for p in sources}:
        parser.error('Output must not overwrite an original input')
    try:
        inventories = [read(p) for p in args.inventory]
        inventories = [native_ncaaf(i, args.date) if i.get('source') == 'espn_schedule' and i.get('schema_version') == 1 else i for i in inventories]
        report = build_coverage(inventories, selected_date=args.date, as_of=args.as_of, run_id=args.run_id,
            candidates=frame(args.candidates), final=frame(args.final), provider_events=read(args.providers),
            gate_audit=read(args.gate_audit), provider_health=read(args.provider_health),
            policy_exclusions=read(args.policy_exclusions), required_markets=args.required_market or MARKETS)
        validate_report(report)
    except (ValueError, TypeError, KeyError) as exc:
        print('COVERAGE_RECONCILIATION_FAILED: ' + str(exc), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    output = pd.DataFrame(report['decisions'])
    for name in output:
        if output[name].map(lambda v: isinstance(v, (dict, list))).any():
            output[name] = output[name].map(lambda v: json.dumps(v, sort_keys=True, allow_nan=False))
    output.to_csv(targets[1], index=False)
    print(json.dumps({k:report[k] for k in ('inventory_status', 'inventory_scope', 'counts', 'blockers_by_league_market', 'orphan_events', 'reconciliation', 'fully_reconciled')}, sort_keys=True))
    return {'COMPLETE': 0, 'PARTIAL': 2, 'UNAVAILABLE': 3}[report['inventory_status']]


if __name__ == '__main__':
    raise SystemExit(main())
