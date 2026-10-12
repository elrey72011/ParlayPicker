"""Network-free planning by default. Execute only after exact local approvals."""
import argparse
import getpass
import hashlib
import http.client
import json
from pathlib import Path
import socket
import ssl
import sys

# Capture before importing application/dependency code. Never undo test blockers.
_TRANSPORT = (socket.create_connection, socket.getaddrinfo, socket.socket.connect,
              http.client.HTTPSConnection.connect, ssl.SSLContext.wrap_socket)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def verify_runtime():
    if not (sys.flags.isolated and sys.dont_write_bytecode
            and not any(k in sys.modules for k in ('sitecustomize','usercustomize','scope_git_reader','pytest','tests.conftest'))
            and socket.create_connection.__module__=='socket' and socket.getaddrinfo.__module__=='socket'
            and getattr(socket.socket.connect,'__code__',None) is None
            and _TRANSPORT==(socket.create_connection,socket.getaddrinfo,socket.socket.connect,http.client.HTTPSConnection.connect,ssl.SSLContext.wrap_socket)):
        raise ValueError('NFL_ACQUIRE_RUNTIME_UNSUPPORTED')
    context=ssl.create_default_context()
    if not (context.verify_mode==ssl.CERT_REQUIRED and context.check_hostname):
        raise ValueError('NFL_ACQUIRE_RUNTIME_UNSUPPORTED')


def private_credential():
    if not sys.stdin.isatty(): raise ValueError('NFL_ACQUIRE_PRIVATE_TERMINAL_REQUIRED')
    return getpass.getpass('Odds API key (private input; never stored): ')


def bounded_json(path):
    from app_core import nfl_odds_acquisition as lane
    with path.open('rb') as f: raw=f.read(128*1024+1)
    lane.require(len(raw)<=128*1024,'NFL_ACQUIRE_CONFIGURATION_SIZE')
    return json.loads(raw,object_pairs_hook=lane.custody.unique)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--custody-root',type=Path)
    parser.add_argument('--controlled-bindings',type=Path)
    parser.add_argument('--bindings-sha256')
    args=parser.parse_args(argv)
    try:
        if args.execute: verify_runtime()
        from app_core import nfl_odds_acquisition as lane
        plan=bounded_json(args.plan)
        if not args.execute:
            print(json.dumps(lane.planning(plan),sort_keys=True)); return 0
        if any(v is None for v in (args.authorization,args.custody_root,args.controlled_bindings,args.bindings_sha256)):
            parser.error('Exact authorization, custody, controlled bindings and file hash are required')
        with args.controlled_bindings.open('rb') as f: raw=f.read(128*1024+1)
        auth=bounded_json(args.authorization)
        with lane.controlled(plan,raw,args.bindings_sha256,root=args.custody_root) as approved:
            lane.require(auth==approved,'NFL_ACQUIRE_CONFIGURATION_CONFLICT')
            result=lane.acquire(plan,auth,root=args.custody_root,transport=lane.HttpsTransport(private_credential))
        print(json.dumps(lane.execution_report(result),sort_keys=True))
        return 0 if result['payload']['status']=='CAPTURED_UNREVIEWED' else 1
    except (ValueError,KeyError,TypeError,OSError) as exc:
        text=str(exc)
        code=text if text in {'NFL_ACQUIRE_RUNTIME_UNSUPPORTED','NFL_ACQUIRE_PRIVATE_TERMINAL_REQUIRED'} else lane.rejection_code(exc) if 'lane' in locals() else 'NFL_ACQUIRE_REJECTED'
        print(json.dumps(dict(status='REJECTED',reason=code))); return 1


if __name__=='__main__': raise SystemExit(main())
