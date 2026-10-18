"""Exact #2432 inputs for its unchanged scope assertions only."""
import base64,hashlib,json,sys,types
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def frozen_predecessor(root):
    from scripts import check_launch_change_scope as live
    from scripts import ci_scheduling_fixture_scope as scope
    root=Path(root)
    with (root/scope.BINDING).open('rb') as f:binding=json.loads(f.read())
    with (root/live.GUARD_PATH).open('rb') as f:source=f.read().replace(b'\r\n',b'\n')
    source=scope.predecessor_guard(source,live,binding)
    old=types.ModuleType('exact_ci_template_predecessor_scope');old.__file__=str(root/live.GUARD_PATH)
    from unittest.mock import patch
    read=Path.read_bytes;sys.modules[old.__name__]=old
    with patch.object(Path,'read_bytes',lambda p:source if p.resolve()==(root/live.GUARD_PATH).resolve() else read(p)):
        exec(compile(source,old.__file__,'exec'),old.__dict__)
    return old,binding

def install(request,patch):
    name=request.module.__name__.split('.')[-1]
    if name=='test_ci_scheduling_fixture_scope':return True
    if name!='test_ncaaf_transport_scope':return False
    from scripts import check_launch_change_scope as live
    from scripts import ci_scheduling_fixture_scope as scope
    root=Path(__file__).resolve().parents[1]
    try:Path(request.module.__file__).resolve().relative_to(root/'tests')
    except (ValueError,AttributeError):return False
    # This inherited assertion checks native fixture isolation, not predecessor bytes.
    if getattr(getattr(request,'function',None),'__name__',None)=='test_external_fixture_children_keep_their_actual_live_context':return True
    with (root/scope.BINDING).open('rb') as f:binding=json.loads(f.read())
    if 'ci_fixture_copy' not in binding:return False
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
        restored=scope.previous_source(relative,normalized,live,binding)
        return restored if restored!=normalized else value
    patch.setattr(Path,'read_bytes',previous)
    return True

