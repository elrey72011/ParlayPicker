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


def test_pinnacle_replay_adds_the_extra_pool_scopes_only_when_asked(tmp_path, monkeypatch):
    from app_core import pinnacle_model
    store = tmp_path / 'SYNTHETIC-store.sqlite3'
    store.write_bytes(b'')
    monkeypatch.setattr(pinnacle_model, 'replay_estimate', lambda path, sport, family, **k: {
        'sport': sport, 'market_family': family, 'games_started': 0, 'status': 'NO_GAMES'})
    assert len(mr.pinnacle_replay(store)) == 8
    rows = mr.pinnacle_replay(store, extra=mr.EXTRA_REPLAY_SCOPES)
    assert len(rows) == 12
    assert {(r['sport'], r['market_family']) for r in rows if r['sport'] == 'NHL'} == {('NHL', 'PUCK_LINE'), ('NHL', 'TOTAL')}
    assert '| NHL PUCK_LINE | last 24 hours | 0 |' in '\n'.join(mr._replay_markdown(rows))


def test_nightly_record_pace_recommendation_and_capture_warnings_are_reported():
    import scripts.market_readiness_report as report
    replay = [dict(sport='NBA', market_family='SPREAD', window='last 24 hours', status='ESTIMATED', games_started=8, plan_games=3),
              dict(sport='NBA', market_family='SPREAD', window='last 14 days', status='ESTIMATED', games_started=30, plan_games=3),
              dict(sport='NCAAB', market_family='SPREAD', window='last 24 hours', status='ESTIMATED', games_started=2, plan_games=0)]
    paces = [dict(sport='NHL', market_family='TOTAL', games_started=5,
                  paces=[dict(pace='as captured', captures_per_game=1, bets={})]),
             dict(sport='NBA', market_family='SPREAD', games_started=8,
                  paces=[dict(pace='as captured', captures_per_game=12, bets={})])]
    flags = report.capture_health(replay, paces)
    assert [(f['flag'], f['sport']) for f in flags] == [('WATCH_NOT_CAPTURING', 'NHL'), ('PINNACLE_LINE_LOW', 'NBA')]
    result = dict(
        pinnacle_replay=replay, pace_comparison=paces,
        nightly_record=[dict(sport='NBA', market_family='SPREAD', nights=[
            dict(night='2026-11-09', games_started=7, captures_per_game=10, bets={'1%': 5, '2%': 1}),
            dict(night='2026-11-10', games_started=6, captures_per_game=30, bets={'1%': 1, '2%': 0})])],
        pace_recommendation=[dict(sport='NBA', current_minutes=15, minutes=30, status='RECOMMENDED', bets_as_captured=12.0,
                                  captures_per_game=30, kept={'15 min': 1.0, '30 min': 0.9, '60 min': 0.6}),
                             dict(sport='NHL', evaluation_error='ValueError: x')])
    text = '\n'.join(report._pipeline_markdown(result))
    assert '| NBA SPREAD | 5 / 1 [7] | 1 / 0 [6] | 1 of 2 |' in text
    assert '| NBA | 15 min | 30 min | RECOMMENDED | 12 | 30 | 100% / 90% / 60% |' in text
    assert '| WATCH_NOT_CAPTURING | NHL TOTAL | 5 games started with one capture each |' in text
    assert "| PINNACLE_LINE_LOW | NBA SPREAD | 3 of 8 games at Pinnacle's line |" in text
    assert report._pipeline_markdown(dict(pace_comparison=paces[1:], pinnacle_replay=[]))[-1].endswith('none.')
    assert report._pipeline_markdown({}) == []


def test_replay_clv_is_reported_per_market_window_and_edge(tmp_path, monkeypatch):
    import scripts.market_readiness_report as report
    from app_core import pinnacle_model
    store = tmp_path / 'SYNTHETIC.sqlite3'
    store.write_bytes(b'')
    calls = []

    def fake(path, sport, family, *, now, lookback_days, books):
        calls.append((sport, family, lookback_days))
        if (sport, family) == ('NBA', 'TOTAL'):
            raise ValueError('bad scope')
        return dict(sport=sport, market_family=family, games_started=8, status='MEASURED', by_edge={
            '1%': dict(bets=6, close_fetched=5, comparable=4, coverage=4 / 6, mean_price_clv=0.012,
                       price_clv_se=0.004, t=3.0, beat_close_share=0.75),
            '2%': dict(bets=0, close_fetched=0, comparable=0, coverage=None, mean_price_clv=None,
                       price_clv_se=None, t=None, beat_close_share=None)})
    monkeypatch.setattr(pinnacle_model, 'replay_clv', fake)
    rows = report.replay_clv(store, extra={'NHL': ('TOTAL',)})
    assert len(rows) == 2 * (sum(len(f) for f in pinnacle_model.FAMILIES.values()) + 1)
    assert ('NHL', 'TOTAL', 7) in calls and {days for _, _, days in calls} == {1, 7}
    text = '\n'.join(report._clv_markdown(rows))
    assert '| NBA SPREAD | last 24 hours | 1% | 6 | 5 | 4 (67%) | +1.2% (0.4%) | 3.0 | 75% |' in text
    assert '| NBA SPREAD | last 24 hours | 2% |' not in text                  # an edge with no bets has no row
    assert 'ValueError: bad scope' in text
    assert report._clv_markdown([]) == [] and report.replay_clv(tmp_path / 'missing.sqlite3') == []
