"""Exact immutable #2431 inputs for inherited scope tests, never live gates."""
import base64,hashlib,json,sys,types
from pathlib import Path



def frozen_predecessor(root):
    from tests.historical_scope_input_boundary import canonical_guard,physical_bytes
    live=canonical_guard()
    from scripts import ncaaf_transport_scope as scope
    root=Path(root)
    binding=json.loads(physical_bytes(root/scope.BINDING))
    module,control=live._ci_fixture_module()
    source=module.previous_guard(physical_bytes(root/live.GUARD_PATH).replace(b'\r\n',b'\n'),control)
    if 'ci_fixture_copy' in binding:
        from scripts import ci_scheduling_fixture_scope as correction
        d=binding['ci_fixture_copy'];source=correction.predecessor_guard(source,live,binding)
        prior=json.loads(base64.b64decode(d['previous_binding_b64'],validate=True))
        td=prior['transport_diagnostics']
        source=source.replace(d['previous_binding_sha256'].encode(),td['previous_binding_sha256'].encode())
        source=source.replace(d['previous_guard_identity'].encode(),td['previous_guard_identity'].encode())
        live._require(hashlib.sha256(source).hexdigest()==td['previous_guard_sha256'],'PREVIOUS_GUARD_LOGIC_CHANGED')
    else:
        source=scope.predecessor_guard(source,live,binding)
    old=types.ModuleType('exact_ncaaf_transport_predecessor_scope');old.__file__=str(root/live.GUARD_PATH)
    from unittest.mock import patch
    read=Path.read_bytes
    with patch.dict(sys.modules,{old.__name__:old}), patch.object(Path,'read_bytes',lambda p:source if p.resolve()==(root/live.GUARD_PATH).resolve() else read(p)):
        exec(compile(source,old.__file__,'exec'),old.__dict__)
    from tests.historical_scope_input_boundary import register_owned_module
    register_owned_module(old)
    return old,binding


frozen_predecessor.cache_clear=lambda:None


def _install_selected_transport_inputs(request,patch,plan):
    from ci_fixture_scope_inputs import install as selected
    if selected(request,patch):return
    name=request.module.__name__.split('.')[-1]
    # Explicit inherited scope contracts only. New diagnostics tests and actual
    # subprocess CLI/validation do not receive a projected runtime or result.
    if plan.generation not in {'transport','transport-future'}:return
    from tests.historical_scope_input_boundary import canonical_guard,physical_bytes
    live=canonical_guard()
    from scripts import ncaaf_transport_scope as scope
    root=Path(__file__).resolve().parents[1]
    try:Path(request.module.__file__).resolve().relative_to(root/'tests')
    except (ValueError,AttributeError):return
    binding=json.loads(physical_bytes(root/scope.BINDING))
    if 'transport_diagnostics' not in binding:return
    if plan.generation=='transport':
        old,_=frozen_predecessor(str(root))
        from tests.ncaaf_future_scope_inputs import historical_test_consumers
        for consumer,key in historical_test_consumers(request.module,live):patch.setattr(consumer,key,old)
        import scripts
        patch.setattr(scripts,'check_launch_change_scope',old)
        patch.setitem(sys.modules,'scripts.check_launch_change_scope',old)
    # This route ends at the selected FUTURE generation. Reconstruct its
    # immutable transport inputs without executing an immediately discarded
    # intermediate predecessor module or changing the canonical import twice.
    read=Path.read_bytes
    def previous(path):
        value=read(path)
        try:relative=path.resolve().relative_to(root).as_posix()
        except ValueError:return value
        normalized=value.replace(b'\r\n',b'\n')
        if relative==live.GUARD_PATH:
            return scope.predecessor_guard(normalized,live,binding)
        if relative==scope.BINDING:return scope.previous_source(relative,normalized,live,binding)
        if 'ci_fixture_copy' in binding:
            from scripts import ci_scheduling_fixture_scope as correction
            normalized=correction.previous_source(relative,normalized,live,binding)
        if relative in binding['transport_diagnostics']['source_reconstructions']:
            return scope.previous_source(relative,normalized,live,binding)
        return normalized if normalized!=value.replace(b'\r\n',b'\n') else value
    patch.setattr(Path,'read_bytes',previous)


def install(request,patch):
    from tests.historical_scope_input_boundary import resolve_context,Role,owned_api_patch
    plan=resolve_context(request)
    if plan.role is not Role.HISTORICAL:return
    return _install_selected_transport_inputs(request,owned_api_patch(patch,plan),plan)
