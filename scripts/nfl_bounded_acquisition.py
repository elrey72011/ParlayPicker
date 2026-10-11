"""Default planning only. Execution requires separate controlled bindings."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app_core import nfl_bounded_acquisition as lane


def main():
    p = argparse.ArgumentParser()
    p.add_argument('plan',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--authorization',type=Path)
    p.add_argument('--custody-root',type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if not args.execute:
        print(json.dumps(lane.planning(plan),sort_keys=True)); return 0
    # No catalog activation/import hooks. Controlled deployment is separately reviewed.
    if args.authorization is None or args.custody_root is None: p.error('Execution needs exact authorization and verified custody root')
    try:
        result = lane.acquire(plan,json.loads(args.authorization.read_text(encoding='utf-8')),root=args.custody_root,transport=lane.HttpsTransport())
        print(json.dumps({k:result['payload'][k] for k in ('status','reason','attempted')},sort_keys=True))
        return 0 if result['payload']['status'] == 'CAPTURED_UNREVIEWED' else 1
    except (ValueError,KeyError,TypeError,OSError) as exc:
        code = str(exc) if str(exc).startswith('NFL_ACQUIRE_') else 'NFL_ACQUIRE_REJECTED'
        print(json.dumps(dict(status='REJECTED',reason=code))); return 1


if __name__ == '__main__': raise SystemExit(main())
