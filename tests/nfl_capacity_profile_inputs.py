"""Explicit f6a7506 profile generation for two unchanged historical assertions."""
from pathlib import Path
import types
import pytest


@pytest.fixture
def historical_filtered_profile_generation(tmp_path, monkeypatch, request):
    from scripts import ncaaf_filtered_history_scope as current
    root=Path(__file__).resolve().parents[1]
    binding=current.authenticate(root)
    prior_raw=current.capacity_parent_binding(binding)
    import json
    prior=json.loads(prior_raw)
    historical=tmp_path/'SYNTHETIC-authenticated-f6a7506-inputs'
    for relative in set(prior['components'])|{current.BINDING}:
        raw=prior_raw if relative==current.BINDING else current.capacity_predecessor(binding,relative,current.physical(root/relative))
        path=historical/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    relative='scripts/ncaaf_filtered_history_scope.py'
    module=types.ModuleType('exact_capacity_parent_history');module.__file__=str(historical/relative)
    exec(compile((historical/relative).read_bytes(),module.__file__,'exec'),module.__dict__)
    module.authenticate(historical)  # Original, unchanged predecessor authentication.
    if request.node.name=='test_legacy_fixture_adapter_is_exact_and_live_inputs_stay_current':
        from tests import historical_scope_input_boundary as boundary
        # The original guard imports its independently pinned older bindings.
        # Copy supporting source documents, never create a projected live guard.
        supporting=list((root/'docs').rglob('*.json'))+list((root/'scripts').glob('*.py'))
        for source in supporting:
            relative=source.relative_to(root).as_posix()
            path=historical/relative
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(current.physical(source))
        ci=json.loads((historical/'docs/paid-launch/ci-fixture-stability-bindings-v1.json').read_bytes())
        for relative in ci['historical_input_boundary'].get('helper_preimages',{}):
            path=historical/relative
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(current.physical(root/relative))
        guard=types.ModuleType('exact_capacity_parent_guard');guard.__file__=str(historical/'scripts/check_launch_change_scope.py')
        import sys
        monkeypatch.setitem(sys.modules,guard.__name__,guard)
        exec(compile(Path(guard.__file__).read_bytes(),guard.__file__,'exec'),guard.__dict__)
        monkeypatch.setattr(boundary,'ROOT',historical)
        monkeypatch.setattr(boundary,'_LIVE_GUARD',guard)
    monkeypatch.setattr(request.module,'ROOT',historical)
    monkeypatch.setattr(request.module,'scope',module)
    yield
