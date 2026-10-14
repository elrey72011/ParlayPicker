"""Network-free explicit retained-input action and static private read-back."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_core import wnba_scenario_research as scenario, wnba_scenario_replay as replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'run', 'replay'])
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--database', type=Path)
    parser.add_argument('--authorize-subject')
    parser.add_argument('--synthetic-inspection', action='store_true', help='Inspection only; cannot run a synthetic packet')
    args = parser.parse_args(argv)
    try:
        if args.action == 'replay':
            result = replay.read(json.loads(args.receipt.read_text()), path=args.database)
            output = replay.display(result)
        else:
            scenario.require(not args.synthetic_inspection or args.action == 'inspect', 'SYNTHETIC_UPLOAD')
            packet = scenario.load(args.packet.read_bytes(), owner_upload=not args.synthetic_inspection)
            if args.action == 'inspect':
                p = packet['payload']
                output = dict(version=p['version'], subject_sha256=scenario.subject(p), event=p['event'], assumption=p['scenario'],
                              required_findings=list(scenario.FINDINGS), owner_review_recorded=p['owner_review'] is not None,
                              numerical_calls=0, source_acceptance=False)
            else:
                scenario.require(args.authorize_subject == scenario.subject(packet['payload']), 'EXECUTION_NOT_SELECTED')
                output = scenario.run(packet, path=args.database)
        print(scenario.encode(output))
        return 0
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
        print(scenario.encode(dict(status='UNAVAILABLE', reason=scenario.reason(exc), wager_action='PASS', stake=0)))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
