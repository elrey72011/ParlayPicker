"""Default offline planning; explicit future execution needs reviewed bindings."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app_core import nfl_scoring_acquisition as lane


def bounded_json(path):
    with path.open('rb') as stream:raw=stream.read(128*1024+1)
    lane.require(len(raw)<=128*1024,'NFL_ACQUIRE_CONFIGURATION_SIZE')
    return json.loads(raw,object_pairs_hook=lane.custody.unique)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--authorization',type=Path)
    p.add_argument('--custody-root',type=Path)
    p.add_argument('--controlled-bindings',type=Path)
    p.add_argument('--bindings-sha256')
    args=p.parse_args(argv)
    try:
        plan=bounded_json(args.plan)
        if not args.execute:
            print(json.dumps(lane.planning(plan),sort_keys=True)); return 0
        if args.authorization is None or args.custody_root is None or args.controlled_bindings is None or args.bindings_sha256 is None:
            p.error('Execution needs exact authorization, verified custody and separately reviewed controlled bindings/hash')
        with args.controlled_bindings.open('rb') as stream:raw=stream.read(128*1024+1)
        auth=bounded_json(args.authorization)
        with lane.controlled(plan,raw,args.bindings_sha256,root=args.custody_root) as approved:
            lane.require(auth==approved,'NFL_ACQUIRE_CONFIGURATION_CONFLICT')
            result=lane.acquire(plan,auth,root=args.custody_root,transport=lane.HttpsTransport())
        print(json.dumps(lane.execution_report(result),sort_keys=True))
        return 0 if result['payload']['status']=='CAPTURED_UNREVIEWED' else 1
    except (ValueError,KeyError,TypeError,OSError) as exc:
        text=str(exc)
        code=text if text.startswith(('NFL_ACQUIRE_','NFL_PRIVATE_')) and text.replace('_','').isalnum() else 'NFL_ACQUIRE_REJECTED'
        print(json.dumps(dict(status='REJECTED',reason=code))); return 1


if __name__=='__main__': raise SystemExit(main())
