"""Exact test-fixture successor. Authenticate current state; never cache validation."""
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys
import zlib
import functools
import types
import ast

PROPOSAL = '261bb913276725605a4b142458b0be536002cac9af327ec17187da2dde1b359d'
CONSOLIDATED_AMENDMENT = 'b7175b51176334e8b31967bac836e38ebea307a0f90d5e1f7af24b850b36fd7e'
HISTORICAL_INPUT_AMENDMENT = '12e3046831d8fde24748e5193a2e6a54311e1be318e703d3739fe6e1c41da7b2'
SHARED_BOUNDARY_AMENDMENT = '8ad3749ffee5ebcb9f2688e9bb1b10d99efb5b5dc476c85d956dfd004c8a731f'
MEASURED_CAPACITY_AMENDMENT = 'dd6e6859971a5b7d3938d9a92a648bed04e94f14b1c2ffdc535e143b54cee9b8'
CAPACITY_PREDECESSOR = '3bfb62123bf550ee5727bd039b6f63c7cb1625d5'
CAPACITY_PREDECESSOR_TREE = '7ec3783d8d8aa071f107909216ffbfaa6d1d288e'
CAPACITY_LEDGER_SHA256 = 'a6a1143c0f264136cecaf4ed543b8ddb04d9b52d4969ffad89152e8259a0fd65'
READ_CONSISTENCY_CAPACITY_AMENDMENT = 'd1e57722658d927bd202b21d9fcd3ace85b051609b2bad6ed07577ad0b8ddae1'
PRESERVED_CANDIDATE = '97ed96247939165c2f73026525e31e1c3dba9302'
PRESERVED_TREE = '945cab79d9674ee15e4d3d81f088e9c5beab969e'
COMPLETE_LEDGER_SHA256 = '5eec77cd6d43fbd41b3839d7b99c135bfd0cb2603e92b794874385f8432261b7'
BASE = '8b79b0a0eead4f63a041a551283a4b6be2980d72'
BASE_TREE = '028d3daf56d71596663a10714495a7b48d53c7ab'
SHARED_REPAIR_FOLLOWUP = dict(predecessor='b6af3a7ab41eeb8f6e4387a30b83945704698536',
    predecessor_tree='41734a2c62e2845c030d6b44341bed0c62032eee',
    clock_path='tests/test_prospective_legacy_view.py',
    clock_before_sha256='3e30ca3b1a1ea3988b129a28575c6c3fe77c6e608fbcf0bc2f16b8d5fd778eac')
GUARD = 'scripts/check_launch_change_scope.py'
BINDING = 'docs/paid-launch/ci-fixture-stability-bindings-v1.json'
POLICY = 'docs/paid-launch/launch-scope-policy-ci-fixture-stability-v1.json'
MAX_SOURCE = 8 * 1024 * 1024
GUARD_TEMPLATE_SHA256 = 'e9069c74d604f1d5f5cf7d0b665f8d14e03e91db7ed0ebdca170d2000a2a1a6e'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def blob(value):
    return hashlib.sha1(b'blob ' + str(len(value)).encode() + b'\0' + value).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def bytes_at(root, relative):
    with (root / relative).open('rb') as stream:
        return stream.read()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def previous_guard(source, binding, *, predecessor=None):
    d = binding['predecessor_guard']
    if sha(source) == d['sha256']:
        return source
    require(type(source) is bytes and len(source) <= MAX_SOURCE, 'SUCCESSOR_GUARD_LIMIT_CHANGED')
    identity = re.search(rb"_CI_STABILITY_GUARD_SHA256='([0-9a-f]{64})'", source)
    stamp = re.search(rb"_CI_STABILITY_BINDING_SHA256='([0-9a-f]{64})'", source)
    require(identity is not None and stamp is not None, 'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    identity, stamp = identity.group(1), stamp.group(1)
    require(source.count(identity) == 1 and sha(source.replace(identity, b'0'*64)) == identity.decode(), 'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    require(stamp.decode() == sha(canonical(binding)), 'REVIEWED_BINDINGS_CHANGED')
    template = source.replace(identity, b'0'*64).replace(stamp, b'0'*64)
    require(sha(template) == GUARD_TEMPLATE_SHA256, 'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    if predecessor is None:
        z = zlib.decompressobj()
        value = z.decompress(base64.b85decode(d['source_b85']), MAX_SOURCE + 1)
        require(z.eof and not z.unused_data, 'PREVIOUS_GUARD_LOGIC_CHANGED')
    else:
        value = predecessor
    require(type(value) is bytes and len(value) == d['bytes'] <= MAX_SOURCE
            and sha(value) == d['sha256'], 'PREVIOUS_GUARD_LOGIC_CHANGED')
    return value


LEGACY_SOURCE_LOADERS = {
    '_slate_predecessor': '_slate_audit_previous_guard_source',
    '_ci_scheduling_predecessor': '_ci_scheduling_previous_guard_source',
    '_ncaaf_compatibility_predecessor': '_ncaaf_compatibility_previous_guard_source',
    '_readiness_dashboard_predecessor': '_readiness_dashboard_previous_guard_source',
    '_ncaaf_normal_predecessor': '_ncaaf_normal_parent_guard_source',
    '_preview_binding_predecessor': '_preview_binding_parent_guard_source',
    '_ncaaf_chronology_predecessor': '_ncaaf_chronology_parent_guard_source',
    '_ncaaf_custody_predecessor': '_ncaaf_custody_parent_guard_source',
    '_ncaaf_pilot_predecessor': '_ncaaf_pilot_parent_guard_source',
    '_discovery_predecessor': '_discovery_parent_guard_source',
    '_cloud_predecessor': '_cloud_peel_exact_guard',
    '_owner_predecessor': '_owner_base_source',
    '_nfl_owner_predecessor': '_nfl_owner_base_source',
}


def install_legacy_source_bridge(namespace, authenticate, _immutable=None):
    """One authenticated immutable source generation; no validation cache.

    Physical source/binding authentication runs on every call, including a
    warmed loader. Only the already verified predecessor bytes are reused.
    No Path override or mutable predecessor namespace is retained here.
    """
    identity = lambda source, binding: (sha(source), sha(canonical(binding)),
                                      binding['predecessor_guard']['sha256'])
    physical, binding = authenticate()
    key = identity(physical, binding)
    if _immutable is None:
        predecessor = previous_guard(physical, binding)
        immutable = (key, predecessor)
    else:
        require(type(_immutable) is tuple and len(_immutable) == 2 and key == _immutable[0], 'LEGACY_SOURCE_GENERATION_CHANGED')
        require(type(_immutable[1]) is bytes and len(_immutable[1]) == binding['predecessor_guard']['bytes'] <= MAX_SOURCE
                and sha(_immutable[1]) == binding['predecessor_guard']['sha256'], 'PREVIOUS_GUARD_LOGIC_CHANGED')
        immutable = _immutable
    check = require
    digest = sha
    def verified():
        source, current = authenticate()
        check(identity(source, current) == immutable[0], 'LEGACY_SOURCE_GENERATION_CHANGED')
        return source, current
    def bind(original, wrapper):
        # Preserve the original global dictionary as well as public metadata.
        result = types.FunctionType(wrapper.__code__, original.__globals__,
            original.__name__, wrapper.__defaults__, wrapper.__closure__)
        result.__kwdefaults__ = wrapper.__kwdefaults__
        return functools.update_wrapper(result, original)
    installer = install_legacy_source_bridge
    module_type = types.ModuleType
    pending = [namespace]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        require(len(seen) <= 128, 'LEGACY_SOURCE_NAMESPACE_LIMIT')
        for value in tuple(current.values()):
            if isinstance(value, types.FunctionType) and value.__code__.co_filename == current.get('__file__'):
                if value.__globals__ is not current:
                    pending.append(value.__globals__)
            elif isinstance(value, module_type) and value.__name__.startswith('exact_') and getattr(value, '__file__', None) == current.get('__file__'):
                pending.append(vars(value))
        for loader, reader in LEGACY_SOURCE_LOADERS.items():
            original = current.get(reader)
            if callable(original) and not getattr(original, '_authenticated_source_bridge', False):
                def source_input(source, *args, _original=original, **kwargs):
                    physical, current_binding = verified()
                    check(digest(immutable[1]) == current_binding['predecessor_guard']['sha256'], 'PREVIOUS_GUARD_LOGIC_CHANGED')
                    if type(source) is bytes and source.replace(b'\r\n', b'\n') == physical:
                        source = immutable[1]
                    return _original(source, *args, **kwargs)
                wrapped = bind(original, source_input)
                wrapped._authenticated_source_bridge = True
                current[reader] = wrapped
            original = current.get(loader)
            if callable(original) and not getattr(original, '_authenticated_loader_bridge', False):
                def loaded(*args, _original=original, **kwargs):
                    verified()
                    result = _original(*args, **kwargs)
                    if isinstance(result, module_type):
                        installer(vars(result), authenticate, immutable)
                    return result
                wrapped = bind(original, loaded)
                wrapped._authenticated_loader_bridge = True
                current[loader] = wrapped


FIXTURE_INITIALIZER_CORRECTION = {'parent': '85d8a3d71327035e2466938855884f9faf936cd7', 'parent_tree': '2a8f8b283ddfc76f7b727c1f5487e96800a16dd1', 'path': 'tests/ncaaf_future_scope_inputs.py', 'before_sha256': '3a4644e5d638d2ab83dcd02aaf19240552d5923659f950c5af1e77f4dd3d0dc4', 'before_git_blob': 'ca01875c5bbaece2ff49188ecdc49978a6235a70', 'after_sha256': 'e4c4955bc3829ad4b649a865e2a94c0199ceb77a6f9c66a60dc6b32eba2bb874', 'initializer': '_scope_reconstruction_for_validation', 'unchanged_source_memo_blob': 'efa69e2f228f0c44f2a5084e3becdd181a6e831d', 'max_entries': 8, 'max_retained_bytes': 33554432, 'max_context_fields': 512}


PRIVATE_DELIVERY_REPAIR = {'parent': '896cb7b10811ccfae220ea31510637da99a88ac6', 'parent_tree': 'c4aa5fd78800a778c97f7e930dd18d0e5a9aae6f', 'authority': 'Latest owner private-delivery engineering steering; original shared-boundary authority retained.', 'source_edits': {'tests/ncaaf_future_scope_inputs.py': {'before_sha256': 'e4c4955bc3829ad4b649a865e2a94c0199ceb77a6f9c66a60dc6b32eba2bb874', 'before_git_blob': '5e52fe7ba891b5d87eff81f08220e017e1c32558', 'after_sha256': '5186a7a546c2443615b43da35aa59106c761ae186c21a20fe32bca75bfe0af03', 'bytes': 8097, 'edits': [["def _new_released_scope_inputs(live):\n    from tests.historical_scope_input_boundary import physical_bytes\n    root=Path(__file__).resolve().parents[1]\n    scope,binding=live._ci_fixture_module()\n    physical=physical_bytes(root/live.GUARD_PATH).replace(b'\\r\\n',b'\\n')\n    current=scope.previous_guard(physical,binding)\n    source=live._ncaaf_future_base_source(current)\n    reviewed=json.loads(physical_bytes(root/'docs/paid-launch/ncaaf-reviewed-target-binding-v1.json'))\n    assert hashlib.sha256(physical_bytes(root/'docs/paid-launch/ncaaf-reviewed-target-binding-v1.json')).hexdigest()==binding['historical_input_boundary']['source_generations']['docs/paid-launch/ncaaf-reviewed-target-binding-v1.json']\n    assert hashlib.sha256(source).hexdigest()==reviewed['historical_future']['prior_base_sha256']\n    frozen=types.ModuleType('exact_ncaaf_future_released_scope');frozen.__file__=str(root/live.GUARD_PATH)\n    from unittest.mock import patch\n    read=Path.read_bytes\n    with patch.dict(sys.modules,{frozen.__name__:frozen}), patch.object(Path,'read_bytes',lambda path:source if path.resolve()==(root/live.GUARD_PATH).resolve() else read(path)):\n        exec(compile(source,frozen.__file__,'exec'),frozen.__dict__)\n        frozen._scope_reconstruction_for_validation()\n    return frozen,json.loads(live.NCAAF_FUTURE_RECONSTRUCTION_JSON)", "def _new_released_scope_inputs(live):\n    from tests.historical_scope_input_boundary import physical_bytes, selected_reader_generation, historical_reader_source, authenticated_predecessor\n    root=Path(__file__).resolve().parents[1]\n    scope,binding=live._ci_fixture_module()\n    reader_generation=selected_reader_generation(binding)\n    if reader_generation=='released-future-v1':\n        current=authenticated_predecessor(scope,binding)\n        source=live._ncaaf_future_base_source(current)\n        reviewed=json.loads(physical_bytes(root/'docs/paid-launch/ncaaf-reviewed-target-binding-v1.json'))\n        assert hashlib.sha256(physical_bytes(root/'docs/paid-launch/ncaaf-reviewed-target-binding-v1.json')).hexdigest()==binding['historical_input_boundary']['source_generations']['docs/paid-launch/ncaaf-reviewed-target-binding-v1.json']\n        assert hashlib.sha256(source).hexdigest()==reviewed['historical_future']['prior_base_sha256']\n    else:\n        source=historical_reader_source(live,binding,reader_generation)\n    frozen=types.ModuleType('exact_ncaaf_future_released_scope');frozen.__file__=str(root/live.GUARD_PATH)\n    from unittest.mock import patch\n    read=Path.read_bytes\n    with patch.dict(sys.modules,{frozen.__name__:frozen}), patch.object(Path,'read_bytes',lambda path:source if path.resolve()==(root/live.GUARD_PATH).resolve() else read(path)):\n        if reader_generation=='released-future-v1':\n            from tests.historical_scope_input_boundary import execute_released_bootstrap\n            execute_released_bootstrap(source,frozen,live,binding)\n            frozen._scope_reconstruction_for_validation()\n        else:\n            exec(compile(source,frozen.__file__,'exec'),frozen.__dict__)\n    return frozen,json.loads(live.NCAAF_FUTURE_RECONSTRUCTION_JSON)", 1], ["def _install_selected_future_inputs(request,patch,live,plan):\n    # The authenticated generation decides before alias discovery or loading.\n    # A consumer importing the guard later has the same namespace contract.\n    if plan.generation not in {'future','transport-future'}:\n        return\n    # Selection is the authenticated physical contract, not alias identity or\n    # whichever generation happened to populate a session fixture first.\n    aliases=[];pending=[request.module];seen=set()\n    target=Path(__file__).resolve().parents[1]/live.GUARD_PATH\n    while pending:\n        current=pending.pop()\n        if id(current) in seen:continue\n        seen.add(id(current));assert len(seen)<=128\n        for name,value in tuple(vars(current).items()):\n            if isinstance(value,types.ModuleType):\n                file=getattr(value,'__file__',None)\n                if file and Path(file).resolve()==target.resolve() and hasattr(value,'_ncaaf_future_base_source'):\n                    aliases.append((current,name))\n                elif value.__name__.split('.')[-1].startswith('test_'):\n                    pending.append(value)\n    frozen,records=_new_released_scope_inputs(live)\n    root=Path(__file__).resolve().parents[1];read=Path.read_bytes\n    from tests.historical_scope_input_boundary import physical_bytes,generation_input\n    scope,binding=live._ci_fixture_module()\n    source_inputs=binding['historical_input_boundary']['consumers'][plan.consumer].get('source_inputs',{})\n    assert len(source_inputs)<=64 and all(isinstance(p,str) and isinstance(h,str) and len(h)==64 for p,h in source_inputs.items())\n    physical_guard=physical_bytes(root/live.GUARD_PATH).replace(b'\\r\\n',b'\\n')\n    main_guard=scope.previous_guard(physical_guard,binding)\n    guard_source=live._ncaaf_future_base_source(main_guard)\n    allowed_guard_inputs={hashlib.sha256(v).hexdigest() for v in (physical_guard,main_guard,guard_source)}\n    original=json.loads(physical_bytes(root/'docs/paid-launch/ncaaf-future-spread-binding-v1.json'))\n    for data in (original.get('ci_fixture_copy',{}),original.get('transport_diagnostics',{})):\n        if 'previous_guard_sha256' in data:allowed_guard_inputs.add(data['previous_guard_sha256'])\n    def historical(path):\n        try:relative=path.resolve().relative_to(root).as_posix()\n        except ValueError:return read(path)\n        value=read(path)\n        if relative==live.GUARD_PATH:\n            return guard_source if hashlib.sha256(value.replace(b'\\r\\n',b'\\n')).hexdigest() in allowed_guard_inputs else value\n        exact=generation_input(plan,relative,value) if relative in source_inputs else None\n        if exact is not None:return exact\n        normalized=value.replace(b'\\r\\n',b'\\n')\n        record=records.get(relative)\n        if record is None:return value\n        return live._ncaaf_future_previous_source(relative,normalized)\n    for consumer,name in aliases:patch.setattr(consumer,name,frozen)\n    import scripts\n    patch.setattr(scripts,'check_launch_change_scope',frozen)\n    patch.setitem(sys.modules,'scripts.check_launch_change_scope',frozen)\n    patch.setitem(sys.modules,frozen.__name__,frozen)\n    patch.setattr(Path,'read_bytes',historical)", "def _install_selected_future_inputs(request,patch,live,plan):\n    # The authenticated generation decides before alias discovery or loading.\n    # A consumer importing the guard later has the same namespace contract.\n    if plan.generation not in {'future','transport-future'}:\n        return\n    # Selection is the authenticated physical contract, not alias identity or\n    # whichever generation happened to populate a session fixture first.\n    aliases=[];pending=[request.module];seen=set()\n    target=Path(__file__).resolve().parents[1]/live.GUARD_PATH\n    while pending:\n        current=pending.pop()\n        if id(current) in seen:continue\n        seen.add(id(current));assert len(seen)<=128\n        for name,value in tuple(vars(current).items()):\n            if isinstance(value,types.ModuleType):\n                file=getattr(value,'__file__',None)\n                if file and Path(file).resolve()==target.resolve():\n                    aliases.append((current,name))\n                elif value.__name__.split('.')[-1].startswith('test_'):\n                    pending.append(value)\n    frozen,records=_new_released_scope_inputs(live)\n    root=Path(__file__).resolve().parents[1];read=Path.read_bytes\n    from tests.historical_scope_input_boundary import physical_bytes,generation_input\n    scope,binding=live._ci_fixture_module()\n    source_inputs=binding['historical_input_boundary']['consumers'][plan.consumer].get('source_inputs',{})\n    assert len(source_inputs)<=64 and all(isinstance(p,str) and isinstance(h,str) and len(h)==64 for p,h in source_inputs.items())\n    physical_guard=physical_bytes(root/live.GUARD_PATH).replace(b'\\r\\n',b'\\n')\n    from tests.historical_scope_input_boundary import selected_reader_generation, historical_reader_source, authenticated_predecessor\n    reader_generation=selected_reader_generation(binding,plan)\n    if reader_generation=='released-future-v1':\n        main_guard=authenticated_predecessor(scope,binding)\n        guard_source=live._ncaaf_future_base_source(main_guard)\n        guard_inputs=(physical_guard,main_guard,guard_source)\n    else:\n        guard_source=historical_reader_source(live,binding,reader_generation)\n        guard_inputs=(physical_guard,guard_source)\n    allowed_guard_inputs={hashlib.sha256(v).hexdigest() for v in guard_inputs}\n    from tests.historical_scope_input_boundary import _BOOTSTRAP_SOURCE\n    allowed_guard_inputs.add(_BOOTSTRAP_SOURCE)\n    original=json.loads(physical_bytes(root/'docs/paid-launch/ncaaf-future-spread-binding-v1.json'))\n    for data in (original.get('ci_fixture_copy',{}),original.get('transport_diagnostics',{})):\n        if 'previous_guard_sha256' in data:allowed_guard_inputs.add(data['previous_guard_sha256'])\n    def historical(path):\n        try:relative=path.resolve().relative_to(root).as_posix()\n        except ValueError:return read(path)\n        value=read(path)\n        if relative==live.GUARD_PATH:\n            return guard_source if hashlib.sha256(value.replace(b'\\r\\n',b'\\n')).hexdigest() in allowed_guard_inputs else value\n        exact=generation_input(plan,relative,value) if relative in source_inputs else None\n        if exact is not None:return exact\n        normalized=value.replace(b'\\r\\n',b'\\n')\n        record=records.get(relative)\n        if record is None:return value\n        return live._ncaaf_future_previous_source(relative,normalized)\n    for consumer,name in aliases:patch.setattr(consumer,name,frozen)\n    import scripts\n    patch.setattr(scripts,'check_launch_change_scope',frozen)\n    patch.setitem(sys.modules,'scripts.check_launch_change_scope',frozen)\n    patch.setitem(sys.modules,frozen.__name__,frozen)\n    # The shared boundary owns the final open/read_bytes agreement after these\n    # exact transformations. Preserve unrelated incoming tampering here.\n    patch.setattr(Path,'read_bytes',historical)", 1]]}, 'tests/test_locked_picks.py': {'before_sha256': '8265e11c4bab2b0f1ab66e72bc9674d9db5d7815752eff1b8a86526ef3882d1e', 'before_git_blob': 'ae5152343f6f2ae2ef3a2b260f4082a6fbe2d502', 'after_sha256': 'cdc5d494b8d59ed4c91fda6741625063a2de287eb3ecfe1429566ca9a97b4a1c', 'bytes': 13667, 'edits': [['import pytest\n', 'import pytest\nfrom tests.apptest_owned_script import app_from_owned_script\n', 1], ['def test_explicit_button_only_and_rerun_idempotence(store,monkeypatch):', 'def test_explicit_button_only_and_rerun_idempotence(store,monkeypatch,tmp_path):', 1], ['def test_lock_ui_explains_missing_quotes_without_requesting_pointless_refresh(monkeypatch):', 'def test_lock_ui_explains_missing_quotes_without_requesting_pointless_refresh(monkeypatch,tmp_path):', 1], ['def test_existing_alias_duplicates_do_not_hide_new_selector(store, monkeypatch):', 'def test_existing_alias_duplicates_do_not_hide_new_selector(store, monkeypatch, tmp_path):', 1], ['AppTest.from_string(code)', 'app_from_owned_script(tmp_path,code)', 3]]}, 'tests/test_mlb_prospective.py': {'before_sha256': '0cf77a41967d64aa90c15706eeea70e5d0e934d71415cee4c0d2410850d1eb34', 'before_git_blob': 'f2c0e22232d91a6d959a581785c0f2d3e1b5b0a0', 'after_sha256': '354c72225d7218f566863449a377147615ff44c66467e46b4ecd2ca43d0c407c', 'bytes': 4625, 'edits': [['import pytest\n', 'import pytest\nfrom tests.apptest_owned_script import app_from_owned_script\n', 1], ['AppTest.from_string(', 'app_from_owned_script(tmp_path,', 1]]}}, 'added_paths': ['tests/apptest_owned_script.py', 'tests/test_apptest_owned_script.py', 'tests/test_locked_picks.py', 'tests/test_mlb_prospective.py'], 'source_sha256': {'tests/apptest_owned_script.py': '713ea44cfb9c6e44f36fd4b41dbceb431dc8794d2fe33bf702dd14bc149a293f', 'tests/test_apptest_owned_script.py': '849dd7e99980905011330e4ce73985ad93a4d27960a00d23fa930e06aad9db27', 'tests/test_locked_picks.py': 'cdc5d494b8d59ed4c91fda6741625063a2de287eb3ecfe1429566ca9a97b4a1c', 'tests/test_mlb_prospective.py': '354c72225d7218f566863449a377147615ff44c66467e46b4ecd2ca43d0c407c', 'tests/historical_scope_input_boundary.py': '4b238fae5b60055cf3d1793865493c5dae97bb61de3d758a1e4f70c1441e6b2f', 'tests/test_historical_scope_input_boundary.py': '332f8cc9c6e4a51940d056930f1440cb96572ee37ce59413df3997169efb488c', 'tests/test_ci_fixture_stability_scope.py': 'cf4008cba0a45b2f5dd7fb40884cd7424c45bb97c1e0d1b685502b105d750db6'}, 'bootstrap_source_sha256': '3720e55ec37baae7dcf53d5eb3b3436fcc69ae41df820a110900965945adb306', 'immutable_image_limit': 33554432, 'native_reconstruction_blob': 'efa69e2f228f0c44f2a5084e3becdd181a6e831d', 'original_AppTest_default_timeout': 3, 'MLB_run_timeout': 20, 'production_changes': False, 'source_parent': '55ecf5aa115462e8b61d6761d120d3226abc07ff', 'source_parent_tree': 'd8c9424f5313f1c35c3077e3e60b7f926a3eaab6', 'bootstrap_generations': [['_ncaaf_custody_predecessor', '_ncaaf_custody_parent_guard_source', '_ncaaf_pilot_parent_guard_source', '35ac29313e9509efd68894a86e4a98b109a58459605fe589cc6f86159e0dbc64', '0772744a3b07e7a3aa67d889efff45743e4562e4fe8b781e2d2b3d5111424d9d', 1], ['_ncaaf_pilot_predecessor', '_ncaaf_pilot_parent_guard_source', '_discovery_parent_guard_source', '6e82f1a4e0e14a34d62b3ead8eba8f44f382e40093dcfec447029d22027a219c', '35ac29313e9509efd68894a86e4a98b109a58459605fe589cc6f86159e0dbc64', 1], ['_discovery_predecessor', '_discovery_parent_guard_source', '_cloud_peel_exact_guard', '8df5b4b619e220d2884f8f395562ab356d41e247e3729a72774d4e0b79cc4d00', '6e82f1a4e0e14a34d62b3ead8eba8f44f382e40093dcfec447029d22027a219c', 2], ['_cloud_predecessor', '_cloud_peel_exact_guard', '_owner_base_source', '1639ccad41c0943799b30aaf6f8c0f08c08aaef96293408d147a919537a6406b', '8df5b4b619e220d2884f8f395562ab356d41e247e3729a72774d4e0b79cc4d00', 1], ['_owner_predecessor', '_owner_base_source', '_nfl_owner_base_source', '0ab2edd3f3a246fde7aa25c81a6e00bceb4a41621f2fb2ea276392725e29cb6b', '1639ccad41c0943799b30aaf6f8c0f08c08aaef96293408d147a919537a6406b', 1]], 'compiled_plan': 'Immutable code and byte transformations only; fresh dictionaries every execution; combined32MiB retention bound; physical authentication before every warm hit.', 'fixture_template_required_native_input': {'path': 'scripts/scope_reconstruction.py', 'blob': 'efa69e2f228f0c44f2a5084e3becdd181a6e831d', 'consumer': 'tests/test_ci_fixture_stability_scope.py::sealed_template', 'semantics': 'Copy the unchanged authenticated native dependency into the synthetic baseline before committing it; preserve missing/corrupt-object rejection.'}, 'reader_generation_contracts': {'tests/ci_fixture_scope_inputs.py': 'released-future-v1', 'tests/ncaaf_future_scope_inputs.py': 'released-future-v1', 'tests/ncaaf_transport_scope_inputs.py': 'released-future-v1', 'tests/test_canonical_download_scope.py': 'released-future-v1', 'tests/test_ci_fixture_stability_scope.py': 'released-future-v1', 'tests/test_ci_scheduling_fixture_copy.py': 'released-future-v1', 'tests/test_ci_scheduling_fixture_scope.py': 'released-future-v1', 'tests/test_ci_scheduling_scope.py': 'released-future-v1', 'tests/test_ci_six_shards.py': 'released-future-v1', 'tests/test_ci_timing_profile.py': 'released-future-v1', 'tests/test_ci_timing_scope.py': 'released-future-v1', 'tests/test_clock_scope_exception_proposal.py': 'released-future-v1', 'tests/test_dfs_scope_policy.py': 'legacy-drive-after-v1', 'tests/test_drive_history_scope_policy.py': 'released-future-v1', 'tests/test_estimate_scope_policy.py': 'released-future-v1', 'tests/test_export_scope.py': 'released-future-v1', 'tests/test_football_catalog_scope.py': 'released-future-v1', 'tests/test_football_document_ci_closure.py': 'released-future-v1', 'tests/test_football_document_ci_closure_scope.py': 'released-future-v1', 'tests/test_football_document_ci_performance.py': 'released-future-v1', 'tests/test_football_document_ci_performance_scope.py': 'released-future-v1', 'tests/test_football_document_fixture_report_scope.py': 'released-future-v1', 'tests/test_football_document_html_fixture_scope.py': 'released-future-v1', 'tests/test_football_document_html_scope.py': 'released-future-v1', 'tests/test_football_document_io_scope.py': 'released-future-v1', 'tests/test_football_document_literals_scope.py': 'released-future-v1', 'tests/test_football_document_metadata_scope.py': 'released-future-v1', 'tests/test_football_document_predecessor_fixture.py': 'released-future-v1', 'tests/test_football_document_predecessor_fixture_scope.py': 'released-future-v1', 'tests/test_football_document_remote_fixture_scope.py': 'released-future-v1', 'tests/test_football_document_six_capacity_scope.py': 'released-future-v1', 'tests/test_football_document_socket_scope.py': 'released-future-v1', 'tests/test_football_research_scope.py': 'released-future-v1', 'tests/test_home_runline_scope_policy.py': 'released-future-v1', 'tests/test_launch_scope_integration.py': 'released-future-v1', 'tests/test_mlb_nhl_research_scope.py': 'released-future-v1', 'tests/test_ncaaf_chronology_capacity.py': 'released-future-v1', 'tests/test_ncaaf_chronology_fixture_copy.py': 'released-future-v1', 'tests/test_ncaaf_chronology_fixture_scope.py': 'released-future-v1', 'tests/test_ncaaf_chronology_scope.py': 'released-future-v1', 'tests/test_ncaaf_cloud_backup_scope.py': 'released-future-v1', 'tests/test_ncaaf_compatibility_scope.py': 'released-future-v1', 'tests/test_ncaaf_coverage_scope_policy.py': 'released-future-v1', 'tests/test_ncaaf_custody_scope.py': 'released-future-v1', 'tests/test_ncaaf_future_fixture_copy.py': 'released-future-v1', 'tests/test_ncaaf_future_spread_scope.py': 'released-future-v1', 'tests/test_ncaaf_normal_scope.py': 'released-future-v1', 'tests/test_ncaaf_owner_scope.py': 'released-future-v1', 'tests/test_ncaaf_pilot_discovery_scope.py': 'released-future-v1', 'tests/test_ncaaf_pilot_scope.py': 'released-future-v1', 'tests/test_ncaaf_postmerge_scope.py': 'released-future-v1', 'tests/test_ncaaf_research_scope.py': 'released-future-v1', 'tests/test_ncaaf_reviewed_target_scope.py': 'released-future-v1', 'tests/test_ncaaf_reviewed_target_timing.py': 'released-future-v1', 'tests/test_ncaaf_schedule_scope_policy.py': 'released-future-v1', 'tests/test_ncaaf_socket_ci_costs.py': 'released-future-v1', 'tests/test_ncaaf_socket_ci_scope.py': 'released-future-v1', 'tests/test_ncaaf_socket_scope.py': 'released-future-v1', 'tests/test_ncaaf_staged_capacity_scope.py': 'released-future-v1', 'tests/test_ncaaf_staged_ci_capacity.py': 'released-future-v1', 'tests/test_ncaaf_staged_scope.py': 'released-future-v1', 'tests/test_ncaaf_terms_scope.py': 'released-future-v1', 'tests/test_ncaaf_transport_scope.py': 'released-future-v1', 'tests/test_nfl_admission_scope.py': 'released-future-v1', 'tests/test_nfl_calibration_scope.py': 'released-future-v1', 'tests/test_nfl_diagnostics_scope.py': 'released-future-v1', 'tests/test_nfl_document_scope.py': 'released-future-v1', 'tests/test_nfl_inputs_scope.py': 'released-future-v1', 'tests/test_nfl_interpretation_scope.py': 'released-future-v1', 'tests/test_nfl_listing_capacity_scope.py': 'released-future-v1', 'tests/test_nfl_listing_cost_scope.py': 'released-future-v1', 'tests/test_nfl_listing_immutable_scope.py': 'released-future-v1', 'tests/test_nfl_listing_metadata_scope.py': 'released-future-v1', 'tests/test_nfl_native_scope.py': 'released-future-v1', 'tests/test_nfl_observed_scope.py': 'released-future-v1', 'tests/test_nfl_odds_listing_scope.py': 'released-future-v1', 'tests/test_nfl_odds_scope.py': 'released-future-v1', 'tests/test_nfl_odds_socket_scope.py': 'released-future-v1', 'tests/test_nfl_owner_scope.py': 'released-future-v1', 'tests/test_nfl_pdf_ci_scope.py': 'released-future-v1', 'tests/test_nfl_pdf_fixture_scope.py': 'released-future-v1', 'tests/test_nfl_pdf_margin_scope.py': 'released-future-v1', 'tests/test_nfl_pdf_reconcile_scope.py': 'released-future-v1', 'tests/test_nfl_pdf_scope.py': 'released-future-v1', 'tests/test_nfl_pdf_timing_costs.py': 'released-future-v1', 'tests/test_nfl_pdf_timing_scope.py': 'released-future-v1', 'tests/test_nfl_provenance_scope.py': 'released-future-v1', 'tests/test_nfl_reconcile_performance_scope.py': 'released-future-v1', 'tests/test_nfl_reconcile_projection.py': 'released-future-v1', 'tests/test_nfl_scenario_scope.py': 'released-future-v1', 'tests/test_nfl_scoring_scope.py': 'released-future-v1', 'tests/test_nfl_seven_predecessor_inputs.py': 'released-future-v1', 'tests/test_nfl_ui_scope.py': 'released-future-v1', 'tests/test_nhl_puck_line_scope.py': 'released-future-v1', 'tests/test_performance_recap_scope.py': 'released-future-v1', 'tests/test_pick_board_scope.py': 'released-future-v1', 'tests/test_preview_binding_scope.py': 'released-future-v1', 'tests/test_producer_provenance_scope.py': 'released-future-v1', 'tests/test_provider_health_scope_policy.py': 'released-future-v1', 'tests/test_readiness_dashboard_scope.py': 'released-future-v1', 'tests/test_remote_canonical_scope.py': 'released-future-v1', 'tests/test_remote_continuation_scope.py': 'released-future-v1', 'tests/test_scope_fixture_isolation.py': 'released-future-v1', 'tests/test_scope_git_batch.py': 'released-future-v1', 'tests/test_scope_git_reads.py': 'released-future-v1', 'tests/test_scope_guard_bootstrap.py': 'released-future-v1', 'tests/test_scope_guard_scan.py': 'released-future-v1', 'tests/test_scope_reconstruction_memo.py': 'released-future-v1', 'tests/test_shared_git_fixture_lifecycle.py': 'released-future-v1', 'tests/test_slate_audit_scope.py': 'released-future-v1', 'tests/test_source_contract_scope.py': 'released-future-v1', 'tests/test_source_intake_scope.py': 'released-future-v1', 'tests/test_sweet_spot_export_scope.py': 'released-future-v1', 'tests/test_theover_fade_scope.py': 'released-future-v1'}, 'reader_source_specs': {'legacy-drive-after-v1': {'bytes': 170111, 'git_blob': '30d04ccf7764a9f07bf9b8c98d3dd8a6556976b4', 'sha256': '2116a73036a6243da92120a29c5a4dcf242d21fa735e31beeeceedd0174ef386', 'source_generation': 'legacy-drive-after-v1', 'source_provenance': {'expected_hash_preserved': True, 'original_binding': 'ESTIMATE_BINDINGS.previous_guard_sha256', 'original_reader': '_estimate_previous_guard_source', 'parent_source_sha256': '3720e55ec37baae7dcf53d5eb3b3436fcc69ae41df820a110900965945adb306'}, 'version': 'authenticated-historical-reader-source-v1'}}, 'reader_contract_semantics': 'Authenticate physical consumer, exact binding and declared guard generation before constructor; retain frozen original expected hash; no validation verdict or mutable module cache; Path.open/read_bytes/read_text agree and restore incoming ownership.', 'read_api_ownership': 'One owned reentrant adapter exposes the existing exact selected byte view through Path.read_bytes/open/read_text for declared repository inputs. No extra source authority, cache or expected hash; restore depth on rejection/interruption.', 'predecessor_reuse': 'Reuse the existing authenticated_predecessor immutable-byte mechanism in the two future fixture call sites; preserve physical guard/scope/binding authentication and output validation on every warm call. Legacy reader does not construct the unused future predecessor.', 'input_generation_overrides': {'tests/test_ci_fixture_stability_scope.py': {'generation': 'stability'}, 'tests/test_ncaaf_reviewed_target_scope.py': {'call_generations': {'test_current_sources_have_exact_preimages_and_unknown_bytes_stay_unknown': {'generation': 'recorded-source-afterimages', 'function_line': 78}}}}}


def trees(g, revision):
    return {path: meta.split()[2] for row in g.git('ls-tree', '-r', revision).splitlines()
            for meta, path in [row.split('\t', 1)]}


def make_policy(g, binding, implementation):
    before, after = trees(g, binding['base']), trees(g, implementation)
    return dict(schema_version=85, policy_version='ci-fixture-stability-v1',
        proposal_sha256=PROPOSAL, shared_historical_boundary_amendment_sha256=binding.get('shared_historical_boundary_amendment_sha256'),
        measured_capacity_amendment_sha256=binding.get('measured_capacity_amendment_sha256'),
        read_consistency_capacity_amendment_sha256=binding.get('read_consistency_capacity_amendment_sha256'),
        advisory_amendment_sha256='c7597d04525eca38a52c7207f75f45829560bd6fa1ab0ce1f5ba2a1bfa7dcf45',
        second_advisory_amendment_sha256='6cf28cd1db29cd0ba101cce67b8ac3bcc5d6857b61ba555b6aa382298fda7295',
        consolidated_advisory_amendment_sha256=CONSOLIDATED_AMENDMENT,historical_input_amendment_sha256=HISTORICAL_INPUT_AMENDMENT,complete_profile_ledger_sha256=COMPLETE_LEDGER_SHA256, base_sha=binding['base'], base_tree=binding['base_tree'],
        implementation_commit=implementation, implementation_tree=g.git('rev-parse', implementation+'^{tree}'),
        implementation_changes={p: dict(before_blob=before.get(p), after_blob=after.get(p)) for p in binding['paths']},
        unchanged_bindings={p: v for p, v in before.items() if p not in binding['paths']})


def validate(g, manifest_path, comparison_base, binding):
    require(binding['proposal_sha256'] == PROPOSAL, 'PROPOSAL_NOT_APPROVED')
    require(binding.get('consolidated_advisory_amendment_sha256') == CONSOLIDATED_AMENDMENT
            and binding.get('complete_profile_ledger_sha256') == COMPLETE_LEDGER_SHA256, 'COMPLETE_PROFILE_PROPOSAL_NOT_APPROVED')
    require(binding.get('historical_input_amendment_sha256') == HISTORICAL_INPUT_AMENDMENT, 'HISTORICAL_INPUT_PROPOSAL_NOT_APPROVED')
    require(binding['base'] == BASE and binding['base_tree'] == BASE_TREE, 'STARTING_MAIN_NOT_APPROVED')
    require(manifest_path.resolve() == (g.ROOT/g.MANIFEST_PATH).resolve(), 'MANIFEST_PATH_NOT_APPROVED')
    raw = g.git_bytes('show', BASE+':'+g.MANIFEST_PATH)
    require(sha(raw) == binding['manifest_sha256'], 'BASELINE_CHANGED')
    # The recorded comparison base is usable only after the original manifest is authenticated.
    require(comparison_base in (None, BASE, json.loads(raw)['base_sha']), 'COMPARISON_BASE_NOT_APPROVED')
    require(g.git('rev-parse', BASE+'^{tree}') == BASE_TREE, 'STARTING_MAIN_TREE_CHANGED')
    parents = [line.split()[1] for line in g.git('cat-file', '-p', BASE).splitlines() if line.startswith('parent ')]
    require(parents == binding['ordered_base_parents'], 'PREDECESSOR_PARENTS_CHANGED')
    require(not g.git('diff', '--name-only') and not g.git('diff', '--cached', '--name-only'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    policy_raw = g.git_bytes('show', 'HEAD:'+POLICY)
    policy = json.loads(policy_raw)
    implementation = policy['implementation_commit']
    shared = binding.get('shared_historical_boundary_amendment_sha256') == SHARED_BOUNDARY_AMENDMENT
    # Original synthetic baseline/seal contracts still start at their supplied
    # exact BASE. Only the actual reviewed checkout may extend the preserved seal.
    successor = shared and BASE == '8b79b0a0eead4f63a041a551283a4b6be2980d72'
    capacity = binding.get('measured_capacity_amendment_sha256') == MEASURED_CAPACITY_AMENDMENT
    if binding.get('measured_capacity_amendment_sha256') is not None:
        require(capacity and binding['measured_capacity']['ledger_sha256']==CAPACITY_LEDGER_SHA256,
                'MEASURED_CAPACITY_PROPOSAL_NOT_APPROVED')
    if binding.get('read_consistency_capacity_amendment_sha256') is not None:
        require(binding['read_consistency_capacity_amendment_sha256']==READ_CONSISTENCY_CAPACITY_AMENDMENT,
                'READ_CONSISTENCY_CAPACITY_PROPOSAL_NOT_APPROVED')
    parent = (CAPACITY_PREDECESSOR if capacity else PRESERVED_CANDIDATE) if successor else BASE
    delivery=binding.get('private_delivery_repair')
    if delivery is not None:
        require(delivery==PRIVATE_DELIVERY_REPAIR and capacity,'PRIVATE_DELIVERY_REPAIR_CHANGED')
        if successor:
            parent=delivery['parent']
            require(g.git('rev-parse',parent+'^{tree}')==delivery['parent_tree'],'PRIVATE_DELIVERY_PARENT_CHANGED')
    require(g.git('show', '-s', '--format=%P', implementation).split() == [parent], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(not g.exists_at(BASE, POLICY), 'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    if successor:
        require(binding.get('preserved_candidate') == PRESERVED_CANDIDATE and binding.get('preserved_tree') == PRESERVED_TREE,
                'PRESERVED_CANDIDATE_CHANGED')
        require(g.git('rev-parse', PRESERVED_CANDIDATE+'^{tree}') == PRESERVED_TREE, 'PRESERVED_CANDIDATE_CHANGED')
        if capacity:
            require(binding['measured_capacity']['predecessor']==CAPACITY_PREDECESSOR
                    and binding['measured_capacity']['predecessor_tree']==CAPACITY_PREDECESSOR_TREE
                    and g.git('rev-parse',CAPACITY_PREDECESSOR+'^{tree}')==CAPACITY_PREDECESSOR_TREE, 'CAPACITY_PREDECESSOR_CHANGED')
        require(g.git_bytes('show', implementation+':'+POLICY) == g.git_bytes('show', parent+':'+POLICY), 'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    else:
        require(not g.exists_at(implementation, POLICY), 'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    changes = set(g.git('diff', '--name-only', BASE, implementation).splitlines())
    require(changes - ({POLICY} if successor else set()) == set(binding['paths']), 'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    followup=binding.get('shared_repair_followup')
    if followup is not None:
        require(followup==SHARED_REPAIR_FOLLOWUP and capacity, 'SHARED_REPAIR_FOLLOWUP_CHANGED')
    require(len(binding['paths']) == len(set(binding['paths'])) == (48 if delivery is not None else 44 if followup else 43 if capacity else 38 if shared else 33), 'IMPLEMENTATION_PATH_SET_CHANGED')
    require(policy == make_policy(g, binding, implementation), 'EXACT_POLICY_BINDINGS_CHANGED')
    impl = trees(g, implementation)
    require(set(binding['reviewed_blobs']) == set(binding['paths'])-{GUARD, BINDING}, 'REVIEWED_BINDING_SET_CHANGED')
    require(all(impl.get(p) == value for p, value in binding['reviewed_blobs'].items()), 'REVIEWED_IMPLEMENTATION_BYTES_CHANGED')
    require(g.git_bytes('show', implementation+':'+BINDING) == canonical(binding), 'REVIEWED_BINDINGS_CHANGED')
    current_guard = g.git_bytes('show', implementation+':'+GUARD)
    require(previous_guard(current_guard, binding) == g.git_bytes('show', BASE+':'+GUARD), 'PREVIOUS_GUARD_LOGIC_CHANGED')
    require(sha(g.git_bytes('show', BASE+':'+GUARD)) == binding['predecessor_guard']['sha256'], 'PREVIOUS_GUARD_CHANGED')
    # Verify every exact predecessor input, not just the historical test's asserted result.
    helper_path = 'tests/ci_fixture_stability_inputs.py'
    helper_raw = g.git_bytes('show', implementation+':'+helper_path)
    require(blob(helper_raw) == binding['reviewed_blobs'][helper_path], 'PREDECESSOR_INPUT_MODULE_CHANGED')
    namespace = {'__file__': str(g.ROOT/helper_path), '__name__': 'exact_ci_fixture_preimages'}
    exec(compile(helper_raw, namespace['__file__'], 'exec'), namespace)
    require(namespace['BASE'] == BASE and len(namespace['RECORDS']) == 25, 'PREDECESSOR_INPUT_SET_CHANGED')
    for path, record in namespace['RECORDS'].items():
        before = g.git_bytes('show', BASE+':'+path)
        after = g.git_bytes('show', implementation+':'+path)
        require(sha(before) == record['before_sha256'] and blob(before) == record['before_git_blob'], 'PREDECESSOR_INPUT_CHANGED')
        authenticated_after=namespace['capacity_predecessor_source'](path,after) if capacity else after
        require(sha(authenticated_after) == record['after_sha256'] and namespace['original_source'](path, after) == before, 'PREDECESSOR_INPUT_CHANGED')
    if binding.get('fixture_initializer_correction') is not None:
        require(binding['fixture_initializer_correction']==FIXTURE_INITIALIZER_CORRECTION,'FIXTURE_INITIALIZER_BINDING_CHANGED')
        meta=FIXTURE_INITIALIZER_CORRECTION
        rows=namespace['FIXTURE_INITIALIZER_SOURCE_RECORDS']
        require(set(rows)=={meta['path']},'FIXTURE_INITIALIZER_INPUT_SET_CHANGED')
        row=rows[meta['path']]
        current=g.git_bytes('show',implementation+':'+meta['path'])
        current=namespace['private_delivery_predecessor_source'](meta['path'],current)
        prior=namespace['fixture_initializer_predecessor_source'](meta['path'],current)
        if successor:
            require(prior==g.git_bytes('show',meta['parent']+':'+meta['path']),'FIXTURE_INITIALIZER_PREIMAGE_CHANGED')
        require(sha(prior)==meta['before_sha256'] and blob(prior)==meta['before_git_blob']
                and sha(current)==meta['after_sha256']
                and namespace['fixture_initializer_predecessor_source'](meta['path'],current)==prior,
                'FIXTURE_INITIALIZER_PREIMAGE_CHANGED')
        original=b"        exec(compile(source,frozen.__file__,'exec'),frozen.__dict__)\n    return frozen,json.loads(live.NCAAF_FUTURE_RECONSTRUCTION_JSON)\n\n\n@pytest.fixture"
        corrected=b"        exec(compile(source,frozen.__file__,'exec'),frozen.__dict__)\n        frozen._scope_reconstruction_for_validation()\n    return frozen,json.loads(live.NCAAF_FUTURE_RECONSTRUCTION_JSON)\n\n@pytest.fixture"
        require(prior.count(original)==1 and prior.replace(original,corrected)==current,'FIXTURE_INITIALIZER_SEMANTICS_CHANGED')
        if successor:
            require(blob(g.git_bytes('show',implementation+':scripts/scope_reconstruction.py'))==meta['unchanged_source_memo_blob'],
                    'RECONSTRUCTION_LIMIT_OR_LOGIC_CHANGED')
    if delivery is not None:
        contracts=delivery['reader_generation_contracts']
        consumers=binding['historical_input_boundary']['consumers']
        overrides=delivery['input_generation_overrides']
        require(all(all(consumers[p].get(k)==v for k,v in fields.items())for p,fields in overrides.items())
                and all(not row.get('call_generations')or path in overrides for path,row in consumers.items()),
                'HISTORICAL_CALL_GENERATION_CHANGED')
        require(set(contracts)==set(consumers) and all(consumers[p].get('guard_source_generation')==key
                for p,key in contracts.items()),'HISTORICAL_READER_CONTRACT_CHANGED')
        require(set(contracts.values())=={'released-future-v1','legacy-drive-after-v1'}
                and {p for p,key in contracts.items()if key=='legacy-drive-after-v1'}=={'tests/test_dfs_scope_policy.py'},
                'HISTORICAL_READER_CONTRACT_CHANGED')
        specs=delivery['reader_source_specs'];sources=binding.get('historical_reader_sources',{})
        require(set(sources)==set(specs)=={'legacy-drive-after-v1'},'HISTORICAL_READER_SOURCE_CHANGED')
        for key,row in sources.items():
            require(set(row)==set(specs[key])|{'compressed_b85'}
                    and {k:v for k,v in row.items()if k!='compressed_b85'}==specs[key],
                    'HISTORICAL_READER_SOURCE_CHANGED')
            try:
                require(isinstance(row['compressed_b85'],str) and len(row['compressed_b85'])<=1024*1024,
                        'HISTORICAL_READER_SOURCE_CHANGED')
                decoder=zlib.decompressobj();value=decoder.decompress(base64.b85decode(row['compressed_b85']),row['bytes']+1)
            except (ValueError,TypeError,zlib.error):
                require(False,'HISTORICAL_READER_SOURCE_CHANGED')
            require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
                    and len(value)==row['bytes'] and sha(value)==row['sha256'] and blob(value)==row['git_blob'],
                    'HISTORICAL_READER_SOURCE_CHANGED')
        rows=namespace['PRIVATE_DELIVERY_SOURCE_RECORDS']
        require(set(rows)==set(delivery['source_edits']),'PRIVATE_DELIVERY_INPUT_SET_CHANGED')
        for path,meta in delivery['source_edits'].items():
            current=g.git_bytes('show',implementation+':'+path)
            previous=namespace['private_delivery_predecessor_source'](path,current)
            require(sha(previous)==meta['before_sha256'] and blob(previous)==meta['before_git_blob']
                    and sha(current)==meta['after_sha256'],'PRIVATE_DELIVERY_PREIMAGE_CHANGED')
            if successor:
                require(previous==g.git_bytes('show',delivery['source_parent']+':'+path),'PRIVATE_DELIVERY_PREIMAGE_CHANGED')
            expected=previous
            for old,new,count in meta['edits']:
                require(expected.count(old.encode())==count,'PRIVATE_DELIVERY_SEMANTICS_CHANGED')
                expected=expected.replace(old.encode(),new.encode())
            require(expected==current,'PRIVATE_DELIVERY_SEMANTICS_CHANGED')
        for path,expected in delivery['source_sha256'].items():
            require(sha(g.git_bytes('show',implementation+':'+path))==expected,'PRIVATE_DELIVERY_SOURCE_CHANGED')
        require(blob(g.git_bytes('show',implementation+':scripts/scope_reconstruction.py'))==delivery['native_reconstruction_blob'],
                'RECONSTRUCTION_LIMIT_OR_LOGIC_CHANGED')
    if successor:
        helpers=binding['historical_input_boundary']['helper_preimages']
        require(set(helpers)=={'tests/ci_fixture_scope_inputs.py','tests/ncaaf_transport_scope_inputs.py','tests/ncaaf_staged_capacity_fixtures.py'}, 'HISTORICAL_HELPER_INPUT_SET_CHANGED')
        for path,record in helpers.items():
            before=g.git_bytes('show',BASE+':'+path);after=g.git_bytes('show',implementation+':'+path)
            decoder=zlib.decompressobj();value=decoder.decompress(base64.b85decode(record['compressed']),MAX_SOURCE+1)
            require(decoder.eof and not decoder.unused_data and value==before
                    and len(value)==record['bytes']<=MAX_SOURCE and sha(value)==record['before_sha256']
                    and blob(value)==record['before_git_blob'] and sha(after)==record['after_sha256'], 'HISTORICAL_HELPER_PREIMAGE_CHANGED')
    # The fixture edits and the one harness adjustment have exact, reversible semantics.
    family = binding['fixture_paths']
    require(len(family) == len(set(family)) == 22, 'FIXTURE_FAMILY_CHANGED')
    for path in family:
        before = g.git_bytes('show', BASE+':'+path)
        after = g.git_bytes('show', implementation+':'+path)
        edits = binding['fixture_insertions'][path]
        value = after
        for insertion in reversed(edits):
            at = insertion['offset']; added = insertion['bytes'].encode()
            require(value[at:at+len(added)] == added, 'FIXTURE_SAFEGUARD_CHANGED')
            value = value[:at] + value[at+len(added):]
        require(value == before and len(edits) == 3, 'PRIOR_ASSERTIONS_CHANGED')
    ui = 'tests/test_drive_history_loading.py'
    before = g.git_bytes('show', BASE+':'+ui)
    after = g.git_bytes('show', implementation+':'+ui)
    require(after.replace(b'source+"\\nmain()", default_timeout=10', b'source+"\\nmain()"') == before, 'PRIOR_ASSERTIONS_CHANGED')
    if followup and successor:
        before=g.git_bytes('show',BASE+':'+followup['clock_path'])
        after=g.git_bytes('show',implementation+':'+followup['clock_path'])
        require(sha(before)==followup['clock_before_sha256'],'PRIOR_ASSERTIONS_CHANGED')
        old={n.name:n for n in ast.parse(before).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
        new={n.name:n for n in ast.parse(after).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
        for name,function in old.items():
            require(name in new,'PRIOR_ASSERTIONS_CHANGED')
            assertions=[ast.dump(n,include_attributes=False) for n in ast.walk(function) if isinstance(n,ast.Assert)]
            preserved=[ast.dump(n,include_attributes=False) for n in ast.walk(new[name]) if isinstance(n,ast.Assert)]
            require(assertions==preserved,'PRIOR_ASSERTIONS_CHANGED')
    parents = g.git('show', '-s', '--format=%P', 'HEAD').split()
    candidate = g.git('rev-parse', 'HEAD')
    if len(parents) == 2:
        require(parents[0] == BASE, 'CI_BASE_PARENT_NOT_APPROVED')
        candidate = parents[1]
        require(g.git('rev-parse', 'HEAD^{tree}') == g.git('rev-parse', candidate+'^{tree}'), 'CI_MERGE_TREE_CHANGED')
    require(g.git('show', '-s', '--format=%P', candidate).split() == [implementation], 'CANDIDATE_NOT_POLICY_SEAL')
    require(g.git('diff', '--name-status', implementation, candidate).splitlines() == [('M\t' if successor else 'A\t')+POLICY], 'SEAL_CHANGE_SET_NOT_APPROVED')
    head = trees(g, 'HEAD')
    require(set(head) == set(policy['unchanged_bindings'])|set(binding['paths'])|{POLICY}, 'UNREVIEWED_PATH')
    require(all(head.get(p) == v for p, v in policy['unchanged_bindings'].items()), 'IMMUTABLE_FILE_CHANGED')
    for path, value in head.items():
        content = bytes_at(g.ROOT, path)
        require(blob(content) == value or blob(content.replace(b'\r\n', b'\n')) == value, 'UNAUTHORIZED_CHECKOUT_CHANGE')
    for path in g.ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.relative_to(g.ROOT).parts:
            continue
        relative = path.relative_to(g.ROOT).as_posix()
        require(path.name not in {'sitecustomize.py', 'usercustomize.py'} and path.suffix != '.pth', 'PROTECTED_RUNTIME_SHADOWING_RISK')
        require(not any(relative.endswith('/'+p) for p in json.loads(raw)['protected_files']), 'PROTECTED_RUNTIME_SHADOWING_RISK')
    return policy


def run(g, manifest_path, base, binding):
    try:
        policy = validate(g, manifest_path, base, binding)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
        return 1, dict(schema_version=85, status='FAIL', policy_valid=False, reason_codes=[str(exc)])
    # The successor compares exact snapshots against its authenticated starting
    # main. Do not require older Git ancestry merely to render the legacy report
    # in a shallow checkout. Its actual comparison context is explicit below.
    _, original = g.run(manifest_path, binding['base'])
    return 0, dict(schema_version=85, status='PASS', policy_valid=True, reason_codes=[],
        base_sha=binding['base'], head_sha=g.git('rev-parse', 'HEAD'), original_guard_report=original,
        original_guard_report_comparison_base=binding['base'], requested_workflow_comparison_base=base,
        approved_integration_changes=policy['implementation_changes'], unchanged_bindings=policy['unchanged_bindings'],
        protected_changes=[], existing_test_changes=[],
        approved_fixture_inputs='Authenticated exact predecessor sources; assertions and current validation unchanged.',
        approved_test_budget='Only the completed-analysis deferred-history AppTest uses ten seconds; production deadlines unchanged.',
        historical_git_deleting_process='UNKNOWN', historical_Streamlit_timeout_cause='UNKNOWN')
