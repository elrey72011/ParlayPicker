"""Authenticate the exact private Refresh source graph, without mutable verdicts."""
import hashlib
import io
import json
from pathlib import Path

BINDING='docs/paid-launch/ncaaf-private-refresh-bindings-v1.json'
_BINDING_SHA256='28a979160808aa8b5aff8c54dcff675d4e55fb407b55d5c3a1fc763d8018ca32'


def physical(path):
    with io.FileIO(path,'r') as stream:
        return stream.read().replace(b'\r\n',b'\n')


def authenticate(root):
    root=Path(root)
    raw=physical(root/BINDING)
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
