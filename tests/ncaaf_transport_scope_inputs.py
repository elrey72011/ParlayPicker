"""Exact immutable #2431 inputs for inherited scope tests, never live gates."""
import base64,hashlib,json,sys,types
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def frozen_predecessor(root):
    from scripts import check_launch_change_scope as live
    from scripts import ncaaf_transport_scope as scope
    root=Path(root)
    with (root/scope.BINDING).open('rb') as f:binding=json.loads(f.read())
    with (root/live.GUARD_PATH).open('rb') as f:source=f.read().replace(b'\r\n',b'\n')
    source=scope.predecessor_guard(source,live,binding)
    old=types.ModuleType('exact_ncaaf_transport_predecessor_scope');old.__file__=str(root/live.GUARD_PATH)
    from unittest.mock import patch
    read=Path.read_bytes
    sys.modules[old.__name__]=old
    with patch.object(Path,'read_bytes',lambda p:source if p.resolve()==(root/live.GUARD_PATH).resolve() else read(p)):
        exec(compile(source,old.__file__,'exec'),old.__dict__)
    return old,binding


def install(request,patch):
    name=request.module.__name__.split('.')[-1]
    # Explicit inherited scope contracts only. New diagnostics tests and actual
    # subprocess CLI/validation do not receive a projected runtime or result.
    if 'scope' not in name or name.startswith('test_ncaaf_transport'):return
    from scripts import check_launch_change_scope as live
    from scripts import ncaaf_transport_scope as scope
    root=Path(__file__).resolve().parents[1]
    try:Path(request.module.__file__).resolve().relative_to(root/'tests')
    except (ValueError,AttributeError):return
    with (root/scope.BINDING).open('rb') as f:binding=json.loads(f.read())
    if 'transport_diagnostics' not in binding:return
    old,binding=frozen_predecessor(str(root))
    from ncaaf_future_scope_inputs import historical_test_consumers
    for consumer,key in historical_test_consumers(request.module,live):patch.setattr(consumer,key,old)
    import scripts
    patch.setattr(scripts,'check_launch_change_scope',old)
    patch.setitem(sys.modules,'scripts.check_launch_change_scope',old)
    read=Path.read_bytes
    def previous(path):
        value=read(path)
        try:relative=path.resolve().relative_to(root).as_posix()
        except ValueError:return value
        normalized=value.replace(b'\r\n',b'\n')
        if relative==live.GUARD_PATH:
            return scope.predecessor_guard(normalized,live,binding)
        if relative==scope.BINDING or relative in binding['transport_diagnostics']['source_reconstructions']:
            return scope.previous_source(relative,normalized,live,binding)
        return value
    patch.setattr(Path,'read_bytes',previous)
