"""Network-free version-selected staged planning/preparation; explicit execution only."""
import argparse
import getpass
import hashlib
import http.client
import json
from pathlib import Path
import socket
import ssl
import sys

_TRANSPORT = (socket.create_connection, socket.getaddrinfo, socket.socket.connect, http.client.HTTPSConnection.connect, ssl.SSLContext.wrap_socket)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify_runtime():
    if not (sys.flags.isolated and sys.dont_write_bytecode and not any(k in sys.modules for k in ('pytest', 'sitecustomize', 'usercustomize', 'tests.conftest'))
            and socket.create_connection.__module__ == 'socket' and socket.getaddrinfo.__module__ == 'socket'
            and getattr(socket.socket.connect, '__code__', None) is None
            and _TRANSPORT == (socket.create_connection, socket.getaddrinfo, socket.socket.connect, http.client.HTTPSConnection.connect, ssl.SSLContext.wrap_socket)):
        raise ValueError('NCAAF_STAGED_RUNTIME_UNSUPPORTED')
    c = ssl.create_default_context()
    if c.verify_mode != ssl.CERT_REQUIRED or not c.check_hostname: raise ValueError('NCAAF_STAGED_RUNTIME_UNSUPPORTED')


def bounded(path, limit):
    with Path(path).open('rb') as f: raw = f.read(limit+1)
    if len(raw) > limit: raise ValueError('NCAAF_STAGED_BUNDLE_LIMIT')
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['plan', 'execute', 'prepare', 'join'])
    parser.add_argument('input', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--controlled-bindings', type=Path)
    parser.add_argument('--bindings-sha256')
    parser.add_argument('--custody-root', type=Path)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--prequote', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--transport-diagnostics', action='store_true', help='Retain separate sanitized private phase observations; no admission authority.')
    args = parser.parse_args(argv)
    lane = None
    try:
        if args.operation == 'execute': verify_runtime()
        from app_core import ncaaf_staged_acquisition as lane, ncaaf_staged_evidence as evidence
        def load(path, limit=lane.MAX_BUNDLE):
            return json.loads(bounded(path, limit), object_pairs_hook=lane.custody._unique, parse_constant=lane.custody._constant)
        value = load(args.input)
        if args.operation == 'plan':
            print(json.dumps(lane.planning(value), sort_keys=True)); return 0
        if args.operation == 'prepare':
            if not args.model or not args.output: parser.error('--model and --output required')
            result = evidence.prepare(value, model_packet=load(args.model), prepared_at=lane.transport.utc())
            lane._exclusive(args.output, lane.model.encode(result))
            print(json.dumps(dict(status='PREPARED_UNREVIEWED', sha256=result['sha256'], inference=False))); return 0
        if args.operation == 'join':
            if not all((args.history, args.prequote, args.output)): parser.error('--history, --prequote and --output required')
            result = evidence.join(load(args.history), load(args.prequote), value)
            lane._exclusive(args.output, lane.model.encode(result))
            print(json.dumps(dict(status='BOUND_UNREVIEWED', sha256=result['sha256'], inference=False))); return 0
        if not all((args.authorization, args.controlled_bindings, args.bindings_sha256, args.custody_root)):
            parser.error('Exact authorization/configuration hash/private custody required')
        raw = bounded(args.controlled_bindings, 128*1024)
        with lane.controlled(value, raw, args.bindings_sha256, root=args.custody_root, at=lane.transport.utc()) as approval:
            lane.require(load(args.authorization, 128*1024) == approval, 'NCAAF_STAGED_AUTHORIZATION')
            if not sys.stdin.isatty(): raise ValueError('NCAAF_STAGED_PRIVATE_TERMINAL_REQUIRED')
            provider = 'cfbd' if lane.is_history(value['payload']) else 'odds_api'
            # One private entry per stage; never arguments/environment/files/logs.
            secret = getpass.getpass(provider+' key (private, memory only): ')
            observation = None
            options = {}
            if args.transport_diagnostics:
                from app_core import ncaaf_transport_diagnostics as diagnostics
                observation = diagnostics.Collector()
                options['diagnostics'] = observation
            transport = lane.transport.HttpsTransport(lambda selected: secret if selected == provider else None, **options)
            result = lane.acquire(value, approval, root=args.custody_root, transport=transport,
                history_bundle=load(args.history) if args.history else None, prequote=load(args.prequote) if args.prequote else None, wall=lane.transport.utc, **options)
            secret = None; transport._secret = None
        summary = dict(status=result['payload']['status'], reason=result['payload']['reason'], attempts=result['payload']['attempts'], bundle_sha256=result['sha256'])
        if args.transport_diagnostics:
            diagnostic_raw = bounded(args.custody_root/(value['sha256']+'.transport-diagnostics.json'),32*1024)
            diagnostic_hash = hashlib.sha256(diagnostic_raw).hexdigest()
            diagnostics.read(args.custody_root,result,diagnostic_raw,diagnostic_hash)
            summary.update(transport_diagnostics_file_sha256=diagnostic_hash,transport_diagnostics_readback='PASS')
        print(json.dumps(summary))
        return 0 if result['payload']['status'] == 'CAPTURED_UNREVIEWED' else 1
    except (ValueError, KeyError, TypeError, OSError):
        print(json.dumps(dict(status='REJECTED', reason='NCAAF_STAGED_STOPPED'))); return 1


if __name__ == '__main__': raise SystemExit(main())
