"""Offline exact-byte reconstruction; no admission or validation result cache."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
from scripts import check_launch_change_scope as guard
from scripts.benchmark_drive_history_loading import blocked_network

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def predecessor():
    binding = json.loads((ROOT/'docs/paid-launch/football-document-ci-performance-binding-v1.json').read_text())
    with blocked_network():
        raw = subprocess.check_output(['git', 'show', binding['base']+':'+guard.GUARD_PATH], cwd=ROOT)
    tree = ast.parse(raw.decode())
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_listing_capacity_base_source'][-1]
    environment = dict(guard.__dict__)
    exec(compile(ast.get_source_segment(raw.decode(), node), '<frozen exact predecessor>', 'exec'), environment)
    return raw, environment['_listing_capacity_base_source'], binding


def test_one_join_matches_frozen_reconstruction(predecessor):
    raw, original, binding = predecessor
    assert hashlib.sha256(raw).hexdigest() == binding['projection_derivation']['input_sha256']
    expected = original(raw)
    assert guard._listing_capacity_base_source(raw) == expected
    current = (ROOT/guard.GUARD_PATH).read_bytes().replace(b'\r\n', b'\n')
    assert guard._docperf_base_source(current) == raw
    assert guard._listing_capacity_base_source(current) == expected


@pytest.mark.parametrize('attack', ['body', 'marker', 'identity', 'short', 'long'])
def test_changed_sources_keep_original_rejections(predecessor, attack):
    raw, original, _ = predecessor
    if attack == 'body': changed = b'X'+raw[1:]
    elif attack == 'marker': changed = raw.replace(b'FOOTBALL_DOCREMOTE_BOOTSTRAP = True', b'FOOTBALL_DOCREMOTE_BOOTSTRAP = None', 1)
    elif attack == 'identity': changed = raw.replace(guard.FOOTBALL_DOCREMOTE_REVIEWED_GUARD_SHA256.encode(), b'1'*64)
    elif attack == 'short': changed = raw[:-1]
    else: changed = raw+b'\nSYNTHETIC_UNREVIEWED=True\n'
    assert changed != raw
    with pytest.raises((ValueError, RuntimeError)) as before: original(changed)
    with pytest.raises(type(before.value)) as after: guard._listing_capacity_base_source(changed)
    assert str(before.value) == str(after.value)


def test_live_parent_and_delegate_checks_are_not_skipped(predecessor, monkeypatch):
    raw, original, _ = predecessor
    with monkeypatch.context() as m:
        m.setattr(guard, 'NFL_INTERPRETATION_PREVIOUS_GUARD_SHA256', '0'*64)
        # The frozen reference also reads the changed live declaration.
        original.__globals__['NFL_INTERPRETATION_PREVIOUS_GUARD_SHA256'] = '0'*64
        with pytest.raises(ValueError) as before: original(raw)
        with pytest.raises(ValueError) as after: guard._listing_capacity_base_source(raw)
        assert str(before.value) == str(after.value)
    original.__globals__['NFL_INTERPRETATION_PREVIOUS_GUARD_SHA256'] = guard.NFL_INTERPRETATION_PREVIOUS_GUARD_SHA256
    def reject(source): raise ValueError('SYNTHETIC_LIVE_DELEGATE_CHANGED')
    with monkeypatch.context() as m:
        m.setattr(guard, '_interpret_parent_capacity_base', reject)
        with pytest.raises(ValueError, match='SYNTHETIC_LIVE_DELEGATE_CHANGED'):
            guard._listing_capacity_base_source(raw)


def test_cross_repository_context_and_original_memo_bounds(predecessor, monkeypatch, tmp_path):
    raw, original, _ = predecessor
    expected = original(raw)
    with monkeypatch.context() as m:
        m.setattr(guard, 'ROOT', tmp_path)
        assert guard._listing_capacity_base_source(raw) == expected
    from scripts import scope_reconstruction as helper
    assert helper.NAMES == ('_nfl_odds_timing_base_source', '_nfl_odds_base_source')
    assert (helper.MAX_SOURCE_BYTES, helper.MAX_RETAINED_BYTES, helper.MAX_ENTRIES, helper.MAX_CONTEXT_FIELDS) == (8*1024*1024, 32*1024*1024, 8, 512)


def test_only_completed_exact_file_costs_and_frozen_metadata():
    profile = json.loads((ROOT/'scripts/ci_test_file_costs_v1.json').read_text())
    assert profile['revision'] == 3
    rows = profile['football_document_ci_cost_provenance']
    assert {r['path'] for r in rows} == {'tests/test_football_catalog_scope.py', 'tests/test_mlb_nhl_research_scope.py', 'tests/test_source_intake_scope.py'}
    for record in rows:
        row = next(r for r in profile['files'] if r['path'] == record['path'])
        assert row['seconds'] == pytest.approx(record['completed_testcase_seconds']*1.15)
        assert row['latest_measurement_complete'] is True
        assert row['profiled_tests'] == record['count']
    source = profile['sources']['document-ci-completed-56-shard6']
    assert source['complete_for_target_files'] is True
    assert source['attempt'] == 1
