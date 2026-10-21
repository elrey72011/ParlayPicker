"""Authenticated successor; exact original guard code and historical readers retained."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
_CI_STABILITY_BINDING_SHA256='491e79ce31d7157fe79b40ea8183014d99e6a332832c5bc8a4e9888b2a51c62b'
_CI_STABILITY_GUARD_SHA256='83831cf2a80cd80d3266850485a6ae0edaa384974d85efdfa01b9068d5ce3e1d'

def _ci_fixture_module():
    with open(ROOT/'scripts/check_launch_change_scope.py','rb') as stream: source=stream.read().replace(b'\r\n',b'\n')
    if source.count(_CI_STABILITY_GUARD_SHA256.encode())!=1 or hashlib.sha256(source.replace(_CI_STABILITY_GUARD_SHA256.encode(),b'0'*64)).hexdigest()!=_CI_STABILITY_GUARD_SHA256:
        raise ValueError('SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    with open(ROOT/'docs/paid-launch/ci-fixture-stability-bindings-v1.json','rb') as stream: raw=stream.read().replace(b'\r\n',b'\n')
    if hashlib.sha256(raw).hexdigest()!=_CI_STABILITY_BINDING_SHA256:raise ValueError('REVIEWED_BINDINGS_CHANGED')
    binding=json.loads(raw);name='scripts/ci_fixture_stability_scope.py';path=ROOT/name
    with open(path,'rb') as stream: content=stream.read().replace(b'\r\n',b'\n')
    if hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()!=binding['reviewed_blobs'][name]:
        raise ValueError('SUCCESSOR_SCOPE_MODULE_CHANGED')
    module=types.ModuleType('exact_ci_fixture_stability_scope');module.__file__=str(path)
    exec(compile(content,str(path),'exec'),module.__dict__)
    return module,binding

# Only immutable, authenticated predecessor source is supplied during its own
# import-time reconstructions. Restore the exact incoming method even on failure.
_ci_module,_ci_binding=_ci_fixture_module()
with open(ROOT/'scripts/check_launch_change_scope.py','rb') as _ci_stream:
    _ci_source=_ci_module.previous_guard(_ci_stream.read().replace(b'\r\n',b'\n'),_ci_binding)
_ci_native_read=Path.read_bytes
_ci_guard_file=(ROOT/'scripts/check_launch_change_scope.py').resolve()
def _ci_import_read(path):
    return _ci_source if path.resolve()==_ci_guard_file else _ci_native_read(path)
try:
    Path.read_bytes=_ci_import_read
    exec(compile(_ci_source[:_ci_source.rindex(b'\ndef main() -> int:')],str(_ci_guard_file),'exec'),globals())
finally:
    Path.read_bytes=_ci_native_read
del _ci_source

_ci_fixture_previous_base_source=_ncaaf_future_base_source
def _ncaaf_future_base_source(source):
    module,binding=_ci_fixture_module()
    if b'_CI_STABILITY_GUARD_SHA256=' in source:source=module.previous_guard(source,binding)
    return _ci_fixture_previous_base_source(source)

_ci_fixture_previous_guard_match=_ncaaf_future_guard_matches
def _ncaaf_future_guard_matches(source):
    try:
        module,binding=_ci_fixture_module()
        if b'_CI_STABILITY_GUARD_SHA256=' in source:source=module.previous_guard(source,binding)
        return _ci_fixture_previous_guard_match(source)
    except (OSError,ValueError,KeyError,TypeError):return False


def _ci_authenticated_source_input():
    module,binding=_ci_fixture_module()
    with open(ROOT/'scripts/check_launch_change_scope.py','rb') as stream:
        source=stream.read().replace(b'\r\n',b'\n')
    return source,binding


_ci_module.install_legacy_source_bridge(globals(),_ci_authenticated_source_input)

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--base')
    parser.add_argument('--json-output',type=Path)
    args=parser.parse_args()
    try:
        module,binding=_ci_fixture_module()
        code,report=module.run(sys.modules[__name__],args.manifest,args.base,binding)
    except Exception as exc:
        code,report=2,dict(schema_version=85,status='ERROR',reason_codes=['GUARD_EXECUTION_ERROR'],error=str(exc))
    rendered=json.dumps(report,indent=2,sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True,exist_ok=True)
        args.json_output.write_text(rendered+'\n',encoding='utf-8')
    print(rendered)
    return code

if __name__=='__main__':
    raise SystemExit(main())
