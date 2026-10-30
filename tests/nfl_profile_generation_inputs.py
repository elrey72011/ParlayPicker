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
                successor = metadata.get('tls_trust_successor', {})
                clock = successor.get('clock_fixture_successor', {})
                saved = clock.get('ncaaf_saved_refresh_successor', {})
                if path in saved.get('historical_sources', {}):
                    require(sha(raw) == binding['components'][path])
                    raw = record_bytes(saved['historical_sources'][path])
                if path in clock.get('historical_sources', {}):
                    require(sha(raw) == saved.get('prior_component_sha256', {}).get(path, binding['components'][path]))
                    raw = record_bytes(clock['historical_sources'][path])
                if path in successor.get('historical_sources', {}):
                    require(sha(raw) == clock.get('prior_component_sha256', {}).get(path, binding['components'][path]))
                    raw = record_bytes(successor['historical_sources'][path])
                if path in current.NORMALIZED:
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
