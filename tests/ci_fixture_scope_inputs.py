"""Exact #2432 inputs for its unchanged scope assertions only."""
import base64,hashlib,json,sys,types
from pathlib import Path


def frozen_predecessor(root):
    from tests.historical_scope_input_boundary import canonical_guard,physical_bytes
    live=canonical_guard()
    from scripts import ci_scheduling_fixture_scope as scope
    root=Path(root)
    binding=json.loads(physical_bytes(root/scope.BINDING))
    module,control=live._ci_fixture_module()
    source=module.previous_guard(physical_bytes(root/live.GUARD_PATH).replace(b'\r\n',b'\n'),control)
    source=scope.predecessor_guard(source,live,binding)
    old=types.ModuleType('exact_ci_template_predecessor_scope');old.__file__=str(root/live.GUARD_PATH)
    from unittest.mock import patch
    read=Path.read_bytes
    with patch.dict(sys.modules,{old.__name__:old}), patch.object(Path,'read_bytes',lambda p:source if p.resolve()==(root/live.GUARD_PATH).resolve() else read(p)):
        exec(compile(source,old.__file__,'exec'),old.__dict__)
    from tests.historical_scope_input_boundary import register_owned_module
    register_owned_module(old)
    return old,binding


# Compatibility with callers of the former root cache. There is no mutable
# cache left to clear; every construction authenticates and builds a new module.
frozen_predecessor.cache_clear=lambda:None

def _install_selected_fixture_inputs(request,patch):
    name=request.module.__name__.split('.')[-1]
    if name=='test_ci_scheduling_fixture_scope':return True
    if name!='test_ncaaf_transport_scope':return False
    from tests.historical_scope_input_boundary import canonical_guard,physical_bytes
    live=canonical_guard()
    from scripts import ci_scheduling_fixture_scope as scope
    root=Path(__file__).resolve().parents[1]
    try:Path(request.module.__file__).resolve().relative_to(root/'tests')
    except (ValueError,AttributeError):return False
    # This inherited assertion checks native fixture isolation, not predecessor bytes.
    if getattr(getattr(request,'function',None),'__name__',None)=='test_external_fixture_children_keep_their_actual_live_context':return True
    binding=json.loads(physical_bytes(root/scope.BINDING))
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



def install(request,patch):
    from tests.historical_scope_input_boundary import resolve_context,Role,owned_api_patch
    plan=resolve_context(request)
    if plan.role is not Role.HISTORICAL:return plan.role is Role.NATIVE
    return _install_selected_fixture_inputs(request,owned_api_patch(patch,plan))
