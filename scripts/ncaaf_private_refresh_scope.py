"""Authenticate the exact private Refresh source graph, without mutable verdicts."""
import hashlib
import io
import json
from pathlib import Path

LEGACY_BINDING='docs/paid-launch/ncaaf-private-refresh-bindings-v1.json'
BINDING='docs/research/ncaaf-filtered-history-bindings-v1.json'
_BINDING_SHA256='be717669ffc2955633ad3b00b05edacea5833e710d4d27a0f84c74e4941c16c6'
_FILTERED_HISTORY_BINDING_SHA256='ebf3be5fadba750afd2e99465d7d6d0caef579d559bafc522f1d619e581facf7'


def physical(path):
    with io.FileIO(path,'r') as stream:
        return stream.read().replace(b'\r\n',b'\n')


def authenticate(root):
    root=Path(root)
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
