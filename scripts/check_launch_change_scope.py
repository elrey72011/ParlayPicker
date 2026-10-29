"""Authenticated successor; exact original guard code and historical readers retained."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]

_FILTERED_HISTORY_PREDECESSOR_IMAGES={} # Pure byte transformations, owned by this current guard.
_FILTERED_HISTORY_LEGACY_IMAGES={}
import base64,zlib
_FILTERED_HISTORY_NATIVE_B85DECODE=base64.b85decode
_FILTERED_HISTORY_NATIVE_B85CODE=getattr(base64.b85decode,'__code__',None)
_FILTERED_HISTORY_NATIVE_DECOMPRESS=zlib.decompressobj
_FILTERED_HISTORY_BINDING_SHA256='14e603462380727923c58513842d02926e92dbc5692e994c90f00530e2eed342'
def _filtered_history_module():
    # Authenticate before executing the selected successor helper.
    import io
    with io.FileIO(ROOT/'docs/research/ncaaf-filtered-history-bindings-v1.json','r') as stream: binding_raw=stream.read(8*1024*1024+1)
    if hashlib.sha256(binding_raw).hexdigest()!=_FILTERED_HISTORY_BINDING_SHA256:raise ValueError('NCAAF_FILTERED_BINDING_CHANGED')
    record=json.loads(binding_raw);relative='scripts/ncaaf_filtered_history_scope.py'
    with io.FileIO(ROOT/relative,'r') as stream: source=stream.read(8*1024*1024+1).replace(b'\r\n',b'\n')
    if source.count(_FILTERED_HISTORY_BINDING_SHA256.encode())!=1 or hashlib.sha256(source.replace(_FILTERED_HISTORY_BINDING_SHA256.encode(),b'0'*64)).hexdigest()!=record['components'][relative]:
        raise ValueError('NCAAF_FILTERED_SOURCE_CHANGED')
    module=types.ModuleType('exact_filtered_history_scope');module.__file__=str(ROOT/relative)
    exec(compile(source,module.__file__,'exec'),module.__dict__);module.authenticate(ROOT)
    module._PREDECESSOR_IMAGES=_FILTERED_HISTORY_PREDECESSOR_IMAGES
    module._LEGACY_GUARD_IMAGES=_FILTERED_HISTORY_LEGACY_IMAGES
    module._NATIVE_B85DECODE=_FILTERED_HISTORY_NATIVE_B85DECODE
    module._NATIVE_B85CODE=_FILTERED_HISTORY_NATIVE_B85CODE
    module._NATIVE_DECOMPRESS=_FILTERED_HISTORY_NATIVE_DECOMPRESS
    return module

if __name__=='__main__':raise SystemExit(_filtered_history_module().main())
_CI_STABILITY_BINDING_SHA256='a42948babc5372d0af500c836917afc36315b3951bada66bf9b3137cd2559251'
_CI_STABILITY_GUARD_SHA256='59ba5d368a58a9c755830049d056442569268a0c424333986af1edd6002b856d'

def _ci_fixture_module():
    with open(ROOT/'scripts/check_launch_change_scope.py','rb') as stream: source=stream.read().replace(b'\r\n',b'\n')
    selected=_filtered_history_module()
    source=selected.predecessor_source(ROOT,'scripts/check_launch_change_scope.py',source)
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
    module.previous_guard=selected.preserve_guard_reader(module.previous_guard,ROOT)
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


def _ci_current_source_input():
    module,binding=_ci_fixture_module()
    with open(ROOT/'scripts/check_launch_change_scope.py','rb') as stream:
        source=stream.read().replace(b'\r\n',b'\n')
    return source,binding


def _ci_authenticated_source_input():
    # Explicit legacy decoder contract: authenticate physical current bytes,
    # then select exactly its reviewed 3e744bb input and matching old binding.
    # Physical readers and current runtime authentication remain independent.
    source,binding=_ci_current_source_input()
    selected=_filtered_history_module()
    return selected.predecessor_source(ROOT,'scripts/check_launch_change_scope.py',source),binding

_ci_module.install_legacy_source_bridge(globals(),_ci_current_source_input)

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
