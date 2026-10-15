"""Advisory revision only; all execution identities and assertions remain intact."""
from collections import Counter
import json
from types import SimpleNamespace
from scripts import run_ci_tests as runner


def test_revision_provenance_and_honest_incomplete_observations(historical_nfl_pdf_capacity_profile, monkeypatch):
    monkeypatch.setattr(runner, 'DEFAULT_PROFILE', historical_nfl_pdf_capacity_profile/'scripts/ci_test_file_costs_v1.json')
    data = json.loads(runner.DEFAULT_PROFILE.read_text())
    assert data['revision'] == 3
    assert data['predecessor_profile_sha256'] == '7e91ba1167ef610a4c93290b0ae3fabb8fe2b0cdf6d7d7a7cebdb880723a6c00'
    assert data['retained_windows_predecessor_sha256'] == '23d55c6f65e13e3a79aa269e6c89dad7ab8ddaa92eef10f259c808b0a5c12e7e'
    assert data['fallback'] == runner.FALLBACK
    for source in ('main-1', 'main-3', 'pr-2', 'pr-3'):
        receipt = data['sources']['oct11-' + source]
        assert receipt['kind'] == 'completed_junit'
        assert receipt['artifact_id'] and len(receipt['junit_sha256']) == 64
        assert len(receipt['assignment_manifest_sha256']) == 64
        assert len(receipt['checkout_commit']) == 40
    rows = {r['path']: r for r in data['files']}
    for name in ('nfl_owner_research', 'nfl_scoring_acquisition', 'nfl_scoring_profile',
                 'nfl_observed_research', 'nfl_bounded_acquisition', 'nfl_diagnostics_scope'):
        row = rows[f'tests/test_{name}.py']
        assert row['completed_measurements'] and row['kind'] == 'estimated'
        assert row['advisory_basis_seconds'] >= row['original_measured_seconds']
        assert row['seconds'] == round(row['advisory_basis_seconds'] * 1.15, 6)
    diagnostic = rows['tests/test_nfl_acquisition_diagnostics.py']
    assert diagnostic['kind'] == 'estimated'
    # The original cancelled basis remains incomplete; newer complete XML is
    # separately attributable and does not upgrade any cancelled observation.
    assert diagnostic['measurement_complete'] is False
    assert diagnostic['latest_measurement_complete'] is True
    assert diagnostic['completed_measurements'][-1]['source'] == 'refresh1-completed-3'
    assert all(data['sources'][r]['completed_junit_available'] is False
               for r in diagnostic['references'] if 'completed_junit_available' in data['sources'][r])
    cancelled = data['sources']['refresh1-cancelled-1']
    assert cancelled['completed_junit_available'] is False
    assert cancelled['observed_nodeids'] == 2318
    assert cancelled['terminal_teardown_and_reporting'] == 'UNKNOWN'
    assert all(row['kind'] == 'estimated' for row in rows.values()
               if 'refresh1-cancelled-1' in row['references'])


def test_retained_collection_assigns_every_identity_once_without_reordering():
    # Labelled synthetic identities/counts, exercising the unchanged scheduler.
    profile = runner.load_profile(runner.DEFAULT_PROFILE)
    entries = json.loads(runner.DEFAULT_PROFILE.read_text())['files']
    items = [SimpleNamespace(path=r['path'], nodeid=f"{r['path']}::SYNTHETIC_{i}")
             for r in entries for i in range(max(1, r['profiled_tests']))]
    items += [SimpleNamespace(path='tests/test_SYNTHETIC_new.py',
                              nodeid=f'tests/test_SYNTHETIC_new.py::case_{i}') for i in range(11)]
    rows, files, loads = runner.assignment(items, 3, profile, runner.ROOT)
    assert rows == runner.assignment(items, 3, profile, runner.ROOT)[0]
    assert [r['nodeid'] for r in rows] == [i.nodeid for i in items]
    assert len({r['nodeid'] for r in rows}) == len(items)
    partition = [[r['nodeid'] for r in rows if r['shard'] == i] for i in (1, 2, 3)]
    assert Counter(n for shard in partition for n in shard) == Counter(i.nodeid for i in items)
    assert all(partition) and max(loads) - min(loads) < 30
    assert all(len({r['shard'] for r in rows if r['path'] == f['path']}) == 1 for f in files)
    new = next(f for f in files if f['path'] == 'tests/test_SYNTHETIC_new.py')
    assert new['scheduling_seconds'] == 38.5 and new['profile_timing_kind'] == 'missing'


from tests.nfl_pdf_capacity_fixtures import historical_nfl_pdf_capacity_profile
