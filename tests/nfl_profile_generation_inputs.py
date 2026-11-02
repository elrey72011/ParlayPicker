"""Authenticated, module-wide input view for the frozen 9,135-case contract.

The native scheduler still reads the current profile. Only the explicitly
bound historical consumer gets predecessor bytes; no individual case dispatch.
"""
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import types
import zlib

import pytest
from tests.nfl_capacity_import_owner import predecessor_imports

CONSUMER = 'tests/test_nfl_capacity_profile.py'
APPEND = b'\n\nfrom tests.nfl_profile_generation_inputs import historical_profile_generation\n'
PARENT = 'ef2c2b14098e0eef8bc2d9626c4d748b1d565d7b961ff311f7ea5c1ee612fa5f'
LIMIT = 8 * 1024 * 1024


def require(value):
    if not value: raise ValueError('NFL_PROFILE_GENERATION_CHANGED')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record_bytes(record):
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b85decode(record['compressed_b85']), LIMIT + 1)
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail)
    require(len(raw) == record['bytes'] <= LIMIT and sha(raw) == record['sha256'])
    return raw


def parent_binding(binding, metadata):
    startup = metadata.get('tls_trust_successor', {}).get('clock_fixture_successor', {}).get(
        'ncaaf_saved_refresh_successor', {}).get('sidebar_startup_successor')
    if startup is not None:
        # Restore the authenticated outer edge before reconstructing this older
        # generation. In particular, it must not inherit newly added UI edges.
        from scripts.ncaaf_sidebar_startup_scope import parent_binding as sidebar_parent
        binding = json.loads(sidebar_parent(binding))
    prior = json.loads(json.dumps(binding))
    del prior['qualification_release_repair']['build_preflight']['receipt_successor']['first_driver_successor']['capacity_context_successor']
    record = metadata['historical_parent_binding']
    prior['components'] = record['components']
    prior['nfl_capacity_predecessors'] = {p: r for p, r in prior['nfl_capacity_predecessors'].items()
        if p in record['predecessor_after_sha256']}
    for path, digest in record['predecessor_after_sha256'].items():
        prior['nfl_capacity_predecessors'][path]['after_sha256'] = digest
    raw = (json.dumps(prior, sort_keys=True, indent=2) + '\n').encode()
    require(len(raw) == record['bytes'] and sha(raw) == record['sha256'] == PARENT)
    return raw


def load_module(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


@contextmanager
def historical_view(root, consumer, destination):
    from scripts import ncaaf_filtered_history_scope as current
    root, consumer, destination = Path(root).resolve(), Path(consumer).resolve(), Path(destination)
    with predecessor_imports(destination):
        binding = current.authenticate(root)
        physical_binding = binding
        account = binding.get('nfl_account_reset_successor', {})
        if account:
            from scripts.nfl_account_reset_scope import parent_binding as account_parent
            binding = json.loads(account_parent(binding))
        metadata = binding['qualification_release_repair']['build_preflight']['receipt_successor']['first_driver_successor']['capacity_context_successor']
        require(consumer == root / CONSUMER)
        sources = metadata['historical_parent_sources']
        before = record_bytes(sources[CONSUMER])
        require(current.physical(consumer) == before + APPEND)
        prior_raw = parent_binding(binding, metadata)
        prior = json.loads(prior_raw)
        stamp = current._BINDING_SHA256.encode()
        for path in set(prior['components']) | set(sources) | {current.BINDING}:
            if path == current.BINDING: raw = prior_raw
            elif path in sources: raw = record_bytes(sources[path])
            else:
                raw = current.physical(root / path)
                if path in account.get('historical_sources', {}):
                    require(sha(raw) == physical_binding['components'][path])
                    raw = record_bytes(account['historical_sources'][path])
                successor = metadata.get('tls_trust_successor', {})
                clock = successor.get('clock_fixture_successor', {})
                saved = clock.get('ncaaf_saved_refresh_successor', {})
                startup = saved.get('sidebar_startup_successor', {})
                if path in startup.get('historical_sources', {}):
                    require(sha(raw) == binding['components'][path])
                    raw = record_bytes(startup['historical_sources'][path])
                if path in saved.get('historical_sources', {}):
                    require(sha(raw) == startup.get('prior_component_sha256', {}).get(path, binding['components'][path]))
                    raw = record_bytes(saved['historical_sources'][path])
                if path in clock.get('historical_sources', {}):
                    require(sha(raw) == saved.get('prior_component_sha256', {}).get(path, binding['components'][path]))
                    raw = record_bytes(clock['historical_sources'][path])
                if path in successor.get('historical_sources', {}):
                    require(sha(raw) == clock.get('prior_component_sha256', {}).get(path, binding['components'][path]))
                    raw = record_bytes(successor['historical_sources'][path])
                if path in current.NORMALIZED:
                    if account:
                        # The outer binding restores its exact parent source as
                        # well as its metadata; keep the original compiler entry.
                        from scripts.nfl_account_reset_scope import reader_entry_parent
                        raw = reader_entry_parent(path, raw)
                    if path == 'scripts/ncaaf_filtered_history_scope.py' and startup:
                        # The older reader is independently pinned. Remove only
                        # this successor's exact, authenticated parent adapter.
                        from scripts.ncaaf_sidebar_startup_scope import HISTORY_PARENT_ADAPTER
                        require(raw.count(HISTORY_PARENT_ADAPTER) == 1)
                        raw = raw.replace(HISTORY_PARENT_ADAPTER, b'', 1)
                    require(raw.count(stamp) == 1)
                    raw = raw.replace(stamp, PARENT.encode())
            if path in prior['components']:
                checked = raw.replace(PARENT.encode(), b'0' * 64) if path in current.NORMALIZED else raw
                require(sha(checked) == prior['components'][path])
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        history = load_module(destination / 'scripts/ncaaf_filtered_history_scope.py', 'private_profile_parent_history')
        history.authenticate(destination)
        scope = load_module(destination / 'scripts/nfl_capacity_profile_scope.py', 'private_profile_parent_scope')
        runner = load_module(destination / 'scripts/run_ci_tests.py', 'private_profile_parent_runner')
        reviewed = load_module(destination / 'tests/qualification_ops/reviewed_profile_inputs.py', 'private_profile_parent_workflow')
        reviewed.history = history
        yield dict(ROOT=destination, history=history, scope=scope, runner=runner,
                   reviewed_workflow_bytes=reviewed.reviewed_workflow_bytes)


@pytest.fixture(autouse=True)
def historical_profile_generation(request):
    root = Path(__file__).resolve().parents[1]
    destination = request.getfixturevalue('tmp_path') / 'SYNTHETIC-authenticated-95d227-profile'
    with historical_view(root, request.module.__file__, destination) as values, pytest.MonkeyPatch.context() as patch:
        for name, value in values.items(): patch.setattr(request.module, name, value)
        yield


CONTEXT_CONSUMER = 'tests/test_nfl_capacity_context_profile.py'
CONTEXT_RECORD = 'docs/paid-launch/nfl-capacity-context-generation-v3.json'
CONTEXT_BINDING = 'ac2abfe85d86f20c9cbbb10be72fdba9a4a61c6c54ab588e4a12bed45fb0167b'
CONTEXT_APPEND = b'\n\nfrom tests.nfl_profile_generation_inputs import historical_context_profile_generation\n'


@contextmanager
def context_view(root, consumer, destination):
    """Exact 9,238-generation source, never a projection of current costs."""
    from scripts import ncaaf_filtered_history_scope as current
    root, consumer, destination = Path(root).resolve(), Path(consumer).resolve(), Path(destination)
    with predecessor_imports(destination):
        binding = current.authenticate(root)
        raw = current.physical(root / CONTEXT_RECORD)
        require(sha(raw) == binding['components'][CONTEXT_RECORD])
        record = json.loads(raw)
        require(record['version'] == 'authenticated-f33-capacity-context-v3'
                and record['commit'] == 'f33fe07a2870efa606e314eb1441eb0405ce3735'
                and consumer == root / CONTEXT_CONSUMER)
        before = record_bytes(record['consumer'])
        require(current.physical(consumer) == before + CONTEXT_APPEND)
        prior_raw = record_bytes(record['binding'])
        require(sha(prior_raw) == CONTEXT_BINDING)
        prior = json.loads(prior_raw)
        require(set(record['sources']) == set(prior['components']) | set(record['fixture_context_blobs'])
                | {CONTEXT_CONSUMER, 'scripts/run_ci_tests.py'})
        # All current inputs are read and checked anew, including inputs whose
        # old bytes are restored. No approval outcome or mutable view is cached.
        for path, declaration in record['sources'].items():
            raw = current.physical(root / path)
            require(sha(current.normalized(path, raw)) == declaration['physical_sha256'])
            if 'before' in declaration:
                raw = record_bytes(declaration['before'])
            elif path in current.NORMALIZED:
                raw = raw.replace(current._BINDING_SHA256.encode(), CONTEXT_BINDING.encode())
            checked = raw.replace(CONTEXT_BINDING.encode(), b'0' * 64) if path in current.NORMALIZED else raw
            require(sha(checked) == declaration['reconstructed_sha256'])
            if path in prior['components']:
                require(sha(checked) == prior['components'][path])
            if path in record['fixture_context_blobs']:
                require(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
                        == record['fixture_context_blobs'][path])
            if path == CONTEXT_CONSUMER: require(raw == before)
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        target = destination / current.BINDING
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(prior_raw)
        history = load_module(destination / 'scripts/ncaaf_filtered_history_scope.py', 'exact_context_history')
        history.authenticate(destination)
        inputs = load_module(destination / 'tests/nfl_profile_generation_inputs.py', 'exact_context_inputs')
        runner = load_module(destination / 'scripts/run_ci_tests.py', 'exact_context_runner')
        yield dict(ROOT=destination, history=history, inputs=inputs, runner=runner)


@pytest.fixture(autouse=True)
def historical_context_profile_generation(request):
    from scripts import ncaaf_filtered_history_scope as current
    import scripts
    import sys
    root = Path(__file__).resolve().parents[1]
    destination = request.getfixturevalue('tmp_path') / 'SYNTHETIC-authenticated-f33-context'
    with context_view(root, request.module.__file__, destination) as values, pytest.MonkeyPatch.context() as patch:
        # The predecessor's own nested view imports its original reader by name.
        # These aliases are owned and restored, even if setup or the body fails.
        patch.setitem(sys.modules, 'scripts.ncaaf_filtered_history_scope', values['history'])
        patch.setattr(scripts, 'ncaaf_filtered_history_scope', values['history'])
        for name, value in values.items(): patch.setattr(request.module, name, value)
        yield
