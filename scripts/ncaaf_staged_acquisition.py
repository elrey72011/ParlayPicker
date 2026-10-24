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
    parser.add_argument('operation', choices=['plan', 'execute', 'prepare', 'join', 'readback'])
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
        if args.operation=='execute' and args.controlled_bindings and args.bindings_sha256:
            # Authenticate the selected source bootstrap before application imports.
            control_raw=bounded(args.controlled_bindings,128*1024)
            if hashlib.sha256(control_raw).hexdigest()!=args.bindings_sha256:raise ValueError('NCAAF_STAGED_CONFIGURATION')
            control=json.loads(control_raw)
            if not isinstance(control,dict) or not isinstance(control.get('payload'),dict):
                raise ValueError('NCAAF_STAGED_CONFIGURATION')
            if control.get('payload',{}).get('version') in {'ncaaf-reviewed-controlled-bindings-v3','ncaaf-filtered-controlled-bindings-v4','ncaaf-fresh-filtered-controlled-bindings-v5','ncaaf-continuation-controlled-bindings-v6'}:
                runtime=control['payload']['runtime'];path=Path(__file__).resolve().parent/'ncaaf_reviewed_target_operator.py'
                source=path.read_bytes()
                if hashlib.sha256(source).hexdigest()!=runtime['sources']['scripts/ncaaf_reviewed_target_operator.py']:
                    raise ValueError('NCAAF_TARGET_RUNTIME_CONFLICT')
                scope={'__file__':str(path),'__name__':'NCAAF_REVIEWED_SOURCE_BOOTSTRAP'}
                exec(compile(source,str(path),'exec'),scope)
                scope['verify_checkout'](runtime['code_head'])
        from app_core import ncaaf_staged_acquisition as lane, ncaaf_staged_evidence as evidence
        def load(path, limit=lane.MAX_BUNDLE):
            return json.loads(bounded(path, limit), object_pairs_hook=lane.custody._unique, parse_constant=lane.custody._constant)
        value = load(args.input)
        if args.operation == 'plan':
            print(json.dumps(lane.planning(value), sort_keys=True)); return 0
        if args.operation == 'readback':
            b=lane.read_bundle(value,complete=False)
            if args.custody_root:
                lane.verify_journal(args.custody_root,value)
                from app_core import ncaaf_transport_diagnostics as observations
                dr=bounded(args.custody_root/(b['plan']['sha256']+'.transport-diagnostics.json'),32*1024)
                observations.read(args.custody_root,value,dr,hashlib.sha256(dr).hexdigest())
            print(json.dumps(dict(status=b['status'],attempts=b['attempts'],readback='PASS',inference=False)));return 0
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
            filtered=value['payload']['version'] in {'ncaaf-filtered-history-stage-plan-v4','ncaaf-fresh-filtered-history-stage-plan-v5','ncaaf-history-continuation-stage-plan-v6'}
            if filtered:
                lane.require(args.transport_diagnostics,'NCAAF_STAGED_DIAGNOSTIC_SELECTION')
                from app_core import ncaaf_filtered_history as filtered_history
                filtered_history.verify_parent_files(args.custody_root,value['payload'])
            # One private entry per stage; never arguments/environment/files/logs.
            secret = getpass.getpass(provider+' key (private, memory only): ')
            if filtered:
                # Input time is outside active acquisition, but wall authority
                # still requires a full minute of completion margin.
                now=lane.timestamp(lane.transport.utc())
                cutoff=lane.timestamp(value['payload']['execution_window']['not_after'])
                if (cutoff-now).total_seconds()<=60:
                    secret=None
                    raise ValueError('NCAAF_STAGED_DEADLINE')
            observation = None
            options = {}
            if args.transport_diagnostics:
                from app_core import ncaaf_transport_diagnostics as diagnostics
                observation = diagnostics.Collector()
                options['diagnostics'] = observation
            if filtered:options['filtered_history_policy']=filtered_history.RequestPolicy(filtered_history.policy_contract(value['payload']))
            transport = lane.transport.HttpsTransport(lambda selected: secret if selected == provider else None, **options)
            result = lane.acquire(value, approval, root=args.custody_root, transport=transport,
                history_bundle=load(args.history) if args.history else None, prequote=load(args.prequote) if args.prequote else None, wall=lane.transport.utc, **({k:v for k,v in options.items() if k!='filtered_history_policy'}))
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
