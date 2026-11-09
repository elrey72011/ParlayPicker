"""Read-only market readiness report (roadmap C0). SYNTHETIC client; no Drive, no network."""
import pytest

from scripts import market_readiness_report as mr


class FakeClient:
    def __init__(self):
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)

    def run_parallel(self, operation, items, progress=None):
        return [operation(self, item) for item in items]  # a real client would hand out writable workers

    def get_object(self, *, Key):
        return dict(Key=Key)

    def read_objects(self, *, Prefix):
        return [('uncached', Prefix)]

    def read_cached_objects(self, *, Prefix, cache_dir):
        return [('cached', Prefix, cache_dir)]


def test_every_write_is_refused_including_through_parallel_workers():
    client = FakeClient()
    guarded = mr.ReadOnlyDrive(client)
    assert guarded.get_object(Key='SYNTHETIC-k') == dict(Key='SYNTHETIC-k')  # reads pass through
    with pytest.raises(mr.ReadOnlyViolation, match='READ_ONLY_VIOLATION:put_object'):
        guarded.put_object(Key='SYNTHETIC-k', Body=b'', IfNoneMatch='*')
    with pytest.raises(mr.ReadOnlyViolation):
        guarded.run_parallel(lambda worker, item: worker.put_object(Key=item), ['SYNTHETIC-a'])
    assert client.puts == [] and guarded.writes_refused == 2


def test_report_covers_all_twelve_markets_from_the_restored_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, 'restore', lambda root, client, folder, with_sources=False: dict(canonical='SYNTHETIC'))
    result = mr.report(tmp_path / 'restore', client=FakeClient(), folder='SYNTHETIC-folder')
    assert len(result['markets']) == 12 and result['remote']['writes_refused'] == 0
    assert {r['remote_evidence_status'] for r in result['markets']} == {'RESTORED_READ_ONLY'}
    assert {r['next_blocker'] for r in result['markets']} == {'NO_PROSPECTIVE_EVENTS'}
    table = mr.markdown(result)
    assert table.count('\n') == 14 and '| NBA | SPREAD |' in table


def test_a_non_empty_restore_directory_is_refused(tmp_path):
    (tmp_path / 'leftover.sqlite3').write_bytes(b'')
    with pytest.raises(SystemExit, match='must start empty'):
        mr.report(tmp_path, client=FakeClient(), folder='SYNTHETIC-folder')


def test_reads_use_the_checksum_verified_cache_only_when_a_cache_dir_is_given(tmp_path):
    assert mr.ReadOnlyDrive(FakeClient()).read_objects(Prefix='SYNTHETIC/') == [('uncached', 'SYNTHETIC/')]
    cached = mr.ReadOnlyDrive(FakeClient(), tmp_path).read_objects(Prefix='SYNTHETIC/')
    assert cached == [('cached', 'SYNTHETIC/', tmp_path)]


def test_table_counts_split_rows_by_sport_without_writing(tmp_path):
    import sqlite3
    path = tmp_path / 'SYNTHETIC-canonical.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE prospective_event (event_id TEXT, sport TEXT)')
        db.executemany('INSERT INTO prospective_event VALUES (?, ?)', [('a', 'NBA'), ('b', 'NBA'), ('c', 'NHL')])
        db.execute('CREATE TABLE prospective_validation_plan (id TEXT)')
        db.execute("INSERT INTO prospective_validation_plan VALUES ('p')")
    before = path.read_bytes()
    assert mr.table_counts(path) == {'prospective_event': {'NBA': 2, 'NHL': 1}, 'prospective_validation_plan': 1}
    assert path.read_bytes() == before and mr.table_counts(tmp_path / 'absent.sqlite3') == {}


def test_plan_evaluations_report_errors_instead_of_guessing(tmp_path):
    assert mr.plan_evaluations(tmp_path / 'absent.sqlite3') == []
    assert mr._scalars({'n': 3, 'rows': [1, 2], 'status': 'X'}) == {'n': 3, 'status': 'X'}
    assert mr._scalars(['MISSING_MODEL', 'NO_CLOSES']) == ['MISSING_MODEL', 'NO_CLOSES'] and mr._scalars([{}]) == 1


def test_pinnacle_replay_covers_each_scope_twice_and_reports_errors(tmp_path, monkeypatch):
    from app_core import pinnacle_model
    assert mr.pinnacle_replay(tmp_path / 'SYNTHETIC-absent.sqlite3') == []
    store = tmp_path / 'SYNTHETIC-store.sqlite3'
    store.write_bytes(b'')
    calls = []

    def estimate(path, sport, family, *, now, lookback_days, books):
        calls.append((sport, family, lookback_days, books))
        if (sport, family) == ('NBA', 'TOTAL'):
            raise ValueError('SYNTHETIC_FAILURE')
        if lookback_days == 1:
            return {'sport': sport, 'market_family': family, 'games_started': 0, 'status': 'NO_GAMES'}
        return {'sport': sport, 'market_family': family, 'status': 'ESTIMATED', 'games_started': 40, 'plan_games': 12,
                'junk_quotes_dropped': 3, 'variants': [
                    {'entry_rule': 'EARLIEST_SHARED_LINE', 'probability': 'PINNACLE_NO_VIG',
                     'by_edge': {'1%': {'bets': 0, 'bets_per_day': 0.0}, '2%': {'bets': 0, 'bets_per_day': 0.0}}},
                    {'entry_rule': 'FIRST_EDGE_CAPTURE', 'probability': 'PINNACLE_NO_VIG',
                     'by_edge': {'1%': {'bets': 5, 'bets_per_day': 1.25}, '2%': {'bets': 2, 'bets_per_day': 0.5}}}]}
    monkeypatch.setattr(pinnacle_model, 'replay_estimate', estimate)
    rows = mr.pinnacle_replay(store)
    assert len(rows) == 8 and {c[2] for c in calls} == {1, 14}
    assert all(c[3] == pinnacle_model.PLAN_BOOKS for c in calls)
    assert {r['window'] for r in rows} == {'last 24 hours', 'last 14 days'}
    failed = [r for r in rows if 'evaluation_error' in r]
    assert len(failed) == 2 and all('SYNTHETIC_FAILURE' in r['evaluation_error'] for r in failed)
    text = '\n'.join(mr._replay_markdown(rows))
    assert '| NCAAB SPREAD | last 14 days | 40 | 12 | 3 | 0 (0.0/day) | 0 (0.0/day) | 5 (1.25/day) | 2 (0.5/day) |' in text
    assert '| NCAAB SPREAD | last 24 hours | 0 |' in text and 'SYNTHETIC_FAILURE' in text
    assert mr._replay_markdown([]) == []
