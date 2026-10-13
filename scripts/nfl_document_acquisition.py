"""Exact-manifest public-document retrieval; network-free planning is default."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.nfl_odds_acquisition import verify_runtime


def bounded(path, maximum=128*1024):
    from app_core import nfl_document_acquisition as lane
    with Path(path).open('rb') as f:raw=f.read(maximum+1)
    lane.require(0<len(raw)<=maximum,'DOCUMENT_CONFIGURATION')
    return raw


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--controlled-bindings',type=Path)
    parser.add_argument('--bindings-sha256')
    parser.add_argument('--custody-root',type=Path)
    args=parser.parse_args(argv)
    from app_core import nfl_document_acquisition as lane
    try:
        plan=json.loads(bounded(args.plan),object_pairs_hook=lane.custody.unique)
        if not args.execute:
            print(json.dumps(lane.planning(plan),sort_keys=True));return 0
        verify_runtime()  # Reject test/platform wrappers; never undo them.
        lane.require(all(v is not None for v in (args.controlled_bindings,args.bindings_sha256,args.custody_root)),'DOCUMENT_CONFIGURATION')
        with lane.controlled(plan,bounded(args.controlled_bindings),args.bindings_sha256,root=args.custody_root) as authorization:
            result=lane.acquire(plan,authorization,root=args.custody_root)
        print(json.dumps(lane.report(result),sort_keys=True))
        return 0 if result['payload']['status']=='CAPTURED_UNREVIEWED' else 1
    except (ValueError,OSError,KeyError,TypeError) as exc:
        code=str(exc) if str(exc) in lane.CODES | {'DOCUMENT_CONFIGURATION','DOCUMENT_CONFIGURATION_REVIEW','DOCUMENT_CUSTODY','DOCUMENT_INTEGRITY','DOCUMENT_PLAN','DOCUMENT_PREREQUISITES','DOCUMENT_UNTRUSTED_CONFIGURATION','DOCUMENT_SPENT_NO_RESUME','DOCUMENT_PERMISSION','DOCUMENT_PERMISSION_CLOCK','DOCUMENT_AUTHORIZATION','DOCUMENT_URL','DOCUMENT_WINDOW','DOCUMENT_WINDOW_EXPIRED','NFL_ACQUIRE_RUNTIME_UNSUPPORTED'} else 'DOCUMENT_REJECTED'
        print(json.dumps(dict(status='REJECTED',reason=code)));return 1


if __name__=='__main__':raise SystemExit(main())
