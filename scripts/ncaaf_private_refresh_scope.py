"""Authenticate the exact private Refresh source graph, without mutable verdicts."""
import hashlib
import io
import json
from pathlib import Path

LEGACY_BINDING='docs/paid-launch/ncaaf-private-refresh-bindings-v1.json'
BINDING='docs/research/ncaaf-filtered-history-bindings-v1.json'
_BINDING_SHA256='be717669ffc2955633ad3b00b05edacea5833e710d4d27a0f84c74e4941c16c6'
_FILTERED_HISTORY_BINDING_SHA256='9a851611f4bad537128719c73ba0f41ed98c3d91c474d2b36f1a15b05d24e600'


def physical(path):
    with io.FileIO(path,'r') as stream:
        return stream.read().replace(b'\r\n',b'\n')


RUNTIME_BINDING='docs/integrity/ncaaf-runtime-binding-v2.json'
_RUNTIME_BINDING_SHA256='e76e6276a5953cda340fe7d0c9e873b09f4a705e9a29ed5b2afbccc8a8cf1200'
SELF='scripts/ncaaf_private_refresh_scope.py'


def runtime_component_bytes(root,relative):
    """Bytes hashed for one production component; this file zeroes its own pin."""
    raw=physical(Path(root)/relative)
    if relative==SELF:
        if raw.count(_RUNTIME_BINDING_SHA256.encode())!=1:
            raise ValueError('NCAAF_RUNTIME_BINDING_CHANGED')
        raw=raw.replace(_RUNTIME_BINDING_SHA256.encode(),b'0'*64)
    return raw


def authenticate_runtime(root):
    """Re-baselined runtime check (2026-11-04): production source and frozen
    evidence only. Tests, workflows and retired seal guards are not runtime."""
    root=Path(root)
    raw=physical(root/RUNTIME_BINDING)
    if hashlib.sha256(raw).hexdigest()!=_RUNTIME_BINDING_SHA256:
        raise ValueError('NCAAF_RUNTIME_BINDING_CHANGED')
    runtime=json.loads(raw)
    if runtime.get('version')!='ncaaf-runtime-binding-v2' or runtime.get('frozen_history_binding')!={'path':BINDING,'sha256':_FILTERED_HISTORY_BINDING_SHA256}:
        raise ValueError('NCAAF_RUNTIME_BINDING_CHANGED')
    for relative,expected in runtime['components'].items():
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts or '\\' in relative:
            raise ValueError('NCAAF_RUNTIME_BINDING_CHANGED')
        if hashlib.sha256(runtime_component_bytes(root,relative)).hexdigest()!=expected:
            raise ValueError('NCAAF_RUNTIME_SOURCE_CHANGED')
    history=physical(root/BINDING)
    if hashlib.sha256(history).hexdigest()!=_FILTERED_HISTORY_BINDING_SHA256:
        raise ValueError('NCAAF_FILTERED_BINDING_CHANGED')
    # Callers read immutable predecessor records from the frozen history binding;
    # 'components' names exactly the files this authentication depended on.
    verified=dict(runtime['components'])
    verified[RUNTIME_BINDING]=_RUNTIME_BINDING_SHA256
    return dict(json.loads(history),components=verified)


def authenticate(root):
    root=Path(root)
    if (root/RUNTIME_BINDING).is_file():
        return authenticate_runtime(root)
    selected=root/BINDING
    if selected.is_file():
        raw=physical(selected)
        if hashlib.sha256(raw).hexdigest()!=_FILTERED_HISTORY_BINDING_SHA256:
            raise ValueError('NCAAF_FILTERED_BINDING_CHANGED')
        binding=json.loads(raw);relative='scripts/ncaaf_filtered_history_scope.py'
        source=physical(root/relative)
        if source.count(_FILTERED_HISTORY_BINDING_SHA256.encode())!=1 or hashlib.sha256(source.replace(_FILTERED_HISTORY_BINDING_SHA256.encode(),b'0'*64)).hexdigest()!=binding['components'][relative]:
            raise ValueError('NCAAF_FILTERED_SOURCE_CHANGED')
        namespace={'__file__':str(root/relative),'__name__':'exact_filtered_refresh_source'}
        exec(compile(source,str(root/relative),'exec'),namespace)
        return namespace['authenticate'](root)
    raw=physical(root/LEGACY_BINDING)
    if hashlib.sha256(raw).hexdigest()!=_BINDING_SHA256:
        raise ValueError('NCAAF_PRIVATE_REFRESH_BINDING_CHANGED')
    binding=json.loads(raw)
    if binding['proposal_sha256']!='5e4d1a9cdff8b3813d2f9ea8156f2f76515f86c6bda7e619b6a9fe4c5474e0b7':
        raise ValueError('NCAAF_PRIVATE_REFRESH_SCOPE_CHANGED')
    for path,expected in binding['components'].items():
        if Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('NCAAF_PRIVATE_REFRESH_SCOPE_CHANGED')
        value=physical(root/path)
        if path=='scripts/ncaaf_private_refresh_scope.py':
            value=value.replace(_BINDING_SHA256.encode(),b'0'*64)
        if hashlib.sha256(value).hexdigest()!=expected:
            raise ValueError('NCAAF_PRIVATE_REFRESH_SOURCE_CHANGED')
    return binding
