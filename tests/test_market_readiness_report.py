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
                       price_clv_se=0.004, t=3.0, beat_close_share=0.75,
                       not_comparable=dict(CLOSE_NOT_FETCHED=1, BOOK_NOT_IN_CLOSE=0, LINE_MOVED=1, CLOSE_BEFORE_BET=0)),
            '2%': dict(bets=0, close_fetched=0, comparable=0, coverage=None, mean_price_clv=None,
                       price_clv_se=None, t=None, beat_close_share=None)})
    monkeypatch.setattr(pinnacle_model, 'replay_clv', fake)
    rows = report.replay_clv(store, extra={'NHL': ('TOTAL',)})
    assert len(rows) == 2 * (sum(len(f) for f in pinnacle_model.FAMILIES.values()) + 1)
    assert ('NHL', 'TOTAL', 7) in calls and {days for _, _, days in calls} == {1, 7}
    text = '\n'.join(report._clv_markdown(rows))
    assert '| NBA SPREAD | last 24 hours | 1% | 6 | 5 | 4 (67%) | +1.2% (0.4%) | 3.0 | 75% | 1 | 0 | 1 | 0 |' in text
    assert '| NBA SPREAD | last 24 hours | 2% |' not in text                  # an edge with no bets has no row
    assert 'ValueError: bad scope' in text
    assert report._clv_markdown([]) == [] and report.replay_clv(tmp_path / 'missing.sqlite3') == []


def test_the_later_entry_table_is_separate_and_leaves_the_any_time_table_as_it_was(tmp_path, monkeypatch):
    import scripts.market_readiness_report as report
    from app_core import pinnacle_model
    store = tmp_path / 'SYNTHETIC.sqlite3'
    store.write_bytes(b'')
    seen = []

    def fake(path, sport, family, *, now, lookback_days, books, entry_within_minutes=None):
        seen.append(entry_within_minutes)
        bets = 6 if entry_within_minutes is None else 3
        return dict(sport=sport, market_family=family, games_started=8, status='MEASURED', by_edge={
            '1%': dict(bets=bets, close_fetched=bets, comparable=3, coverage=3 / bets, mean_price_clv=0.004,
                       price_clv_se=0.002, t=2.0, beat_close_share=2 / 3,
                       not_comparable=dict(CLOSE_NOT_FETCHED=0, BOOK_NOT_IN_CLOSE=0, LINE_MOVED=bets - 3, CLOSE_BEFORE_BET=0))})
    monkeypatch.setattr(pinnacle_model, 'replay_clv', fake)
    any_time, late = report.replay_clv(store), report.late_entry_clv(store)
    assert len(any_time) == len(late) and set(seen) == {None, 60} and pinnacle_model.LATE_ENTRY_MINUTES == 60
    text = report.markdown(dict(markets=[], replay_clv=any_time, late_entry_clv=late))
    first, second = text.index('Replay closing-line value'), text.index('Later entry: the same measurement')
    assert first < second
    assert '| NBA SPREAD | last 24 hours | 1% | 6 | 6 | 3 (50%) | +0.4% (0.2%) | 2.0 | 67% | 0 | 0 | 3 | 0 |' in text[first:second]
    assert '| NBA SPREAD | last 24 hours | 1% | 3 | 3 | 3 (100%) | +0.4% (0.2%) | 2.0 | 67% | 0 | 0 | 0 | 0 |' in text[second:]
    assert report.late_entry_clv(tmp_path / 'missing.sqlite3') == []


def test_the_closing_line_table_is_also_split_by_stake_book_for_seven_days_at_one_percent():
    import scripts.market_readiness_report as report

    def row(window, bets, mean):
        book = dict(bets=bets, close_fetched=bets, comparable=bets - 1, coverage=(bets - 1) / bets, mean_price_clv=mean,
                    price_clv_se=0.005, t=2.0, beat_close_share=0.75)
        return dict(sport='NCAAB', market_family='SPREAD', window=window, by_edge={
            '1%': dict(bets=bets, by_book={'Novig': book, 'FanDuel': dict(book, bets=0)}),
            '2%': dict(bets=1, by_book={'Novig': dict(book, mean_price_clv=0.5)})})
    any_time = [row('last 24 hours', 9, 0.09), row('last 7 days', 5, 0.016), dict(sport='NBA', market_family='TOTAL',
                                                                                 window='last 7 days', evaluation_error='x')]
    late = [row('last 7 days', 4, 0.009)]
    text = '\n'.join(report._clv_by_book_markdown(any_time, late))
    assert '| NCAAB SPREAD | any time | Novig | 5 | 5 | 4 (80%) | +1.6% (0.5%) | 75% |' in text
    assert '| NCAAB SPREAD | last 60 min | Novig | 4 | 4 | 3 (75%) | +0.9% (0.5%) | 75% |' in text
    assert 'FanDuel' not in text and '+9.0%' not in text and '+50.0%' not in text     # no bets; other window; other edge
    assert report._clv_by_book_markdown([], []) == []
    assert report._clv_by_book_markdown([dict(sport='NBA', market_family='SPREAD', window='last 7 days',
                                              by_edge={'1%': dict(bets=3)})], []) == []   # older rows without the split


def test_the_line_adjusted_table_counts_the_moved_lines_and_names_its_assumption():
    import scripts.market_readiness_report as report

    def row(sport, window, adjusted, bets=10, comparable=6):
        return dict(sport=sport, market_family='SPREAD', window=window, by_edge={
            '1%': dict(bets=bets, comparable=comparable, line_adjusted=adjusted),
            '2%': dict(bets=4, comparable=1, line_adjusted=dict(adjusted, mean_clv=0.5))})
    scored = dict(moved_with_close=3, moved_in_favour=2, moved_against=1, mean_points_in_favour=0.5, point_value=0.036,
                  scored=9, mean_clv=0.012, clv_se=0.004, t=3.0, beat_close_share=0.6, moved_mean_clv=0.021)
    hockey = dict(moved_with_close=1, moved_in_favour=0, moved_against=1, mean_points_in_favour=-1.0, point_value=None,
                  scored=2, mean_clv=None, clv_se=None, t=None, beat_close_share=None)
    any_time = [row('NCAAB', 'last 7 days', scored), row('NCAAB', 'last 24 hours', dict(scored, mean_clv=0.09)),
                row('NHL', 'last 7 days', hockey, bets=3, comparable=2),
                dict(sport='NBA', market_family='TOTAL', window='last 7 days', evaluation_error='x')]
    late = [row('NCAAB', 'last 7 days', dict(scored, scored=8, mean_clv=0.008), bets=8)]
    text = '\n'.join(report._clv_line_adjusted_markdown(any_time, late))
    assert '| NCAAB SPREAD | any time | 10 | 6 | 3 | 2 / 1 | +0.50 | 9 (90%) | +1.2% (0.4%) | 3.0 | +2.1% | 3.6% |' in text
    assert '| NCAAB SPREAD | last 60 min | 8 | 6 | 3 | 2 / 1 | +0.50 | 8 (100%) | +0.8% (0.4%) | 3.0 | +2.1% | 3.6% |' in text
    assert '| NHL SPREAD | any time | 3 | 2 | 1 | 0 / 1 | -1.00 | 2 (67%) | no point value assumed | | |  |' in text
    assert 'ASSUMED value per point' in text and 'A plan is not held to this measure' in text
    assert '+9.0%' not in text and '+50.0%' not in text                       # the other window; the other edge
    assert report._clv_line_adjusted_markdown([], []) == []
    assert report._clv_line_adjusted_markdown([dict(sport='NBA', market_family='SPREAD', window='last 7 days',
                                                    by_edge={'1%': dict(bets=3, comparable=1)})], []) == []   # older rows
