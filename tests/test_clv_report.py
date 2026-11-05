"""SYNTHETIC CLV report from real closing-observation records (roadmap E1.4)."""
import hashlib
import json
import sqlite3

import pytest

from app_core.activation_closing import record_close
from app_core.evaluation import clv

KICKOFF = '2026-11-08T18:00:00+00:00'


def candidate(n, market_type='spread_home', line=-2.5, odds=-110, sport='NFL'):
    return dict(snapshot_id='SYNTHETIC-snap', candidate_id=f'SYNTHETIC-c{n}', game_id=f'SYNTHETIC-g{n}', sport=sport,
                market_type=market_type, quote_bookmaker='DraftKings', provider_namespace='odds_api',
                provider_event_id=f'SYNTHETIC-e{n}', game_start_utc=KICKOFF, market_line_used=line, odds_american=odds)


def close(db, cand, *, line, price, recorded='2026-11-08T17:40:00+00:00', captured='2026-11-08T17:45:00+00:00'):
    quote = dict(game_id=cand['game_id'], sport=cand['sport'], market_type=cand['market_type'], sportsbook='DraftKings',
                 provider_event_id=cand['provider_event_id'], provider_namespace='odds_api',
                 quote_recorded_at=recorded, line=line, price=price)
    return record_close(db, cand, quote, captured_at=captured)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / 'SYNTHETIC-evidence.sqlite3'
    close(path, candidate(1), line=-3.5, price=-110)                       # beat by a point
    close(path, candidate(2), line=-1.5, price=-110)                       # lost a point
    close(path, candidate(3, 'total_over', 44.5), line=45.5, price=-110)   # over: higher close is favorable
    c4 = candidate(4, odds=-110)
    close(path, c4, line=-2.5, price=-130, recorded='2026-11-08T17:31:00+00:00', captured='2026-11-08T17:32:00+00:00')
    close(path, c4, line=-2.5, price=-125)                                 # later close for the same pick wins
    return path


def test_report_uses_the_latest_close_per_pick_and_signed_clv(db):
    observations = clv.read_observations(db)
    assert len(observations) == 5
    closes = {c['candidate_id']: c for c in clv.latest_closes(observations)}
    assert len(closes) == 4 and closes['SYNTHETIC-c4']['quote']['price'] == -125
    assert [closes[f'SYNTHETIC-c{i}']['line_clv'] for i in (1, 2, 3)] == [1.0, -1.0, 1.0]
    report = clv.report(observations)
    overall = report['overall']
    assert overall['n'] == 4 and overall['decided'] == 4 and overall['beat_close_rate'] == 0.75
    assert overall['mean_line_clv'] == pytest.approx(0.25)
    assert set(report['by_league_market']) == {'NFL SPREAD', 'NFL TOTAL'}
    assert '### Closing-line value' in clv.markdown(report)


def test_reading_never_writes_to_the_evidence_database(db):
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    clv.read_observations(db)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_a_changed_observation_is_rejected(db):
    with sqlite3.connect(db) as con:  # bypass the database's append-only triggers on purpose
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='trigger' "
                                   "AND tbl_name='closing_observations'").fetchall():
            con.execute(f'DROP TRIGGER "{name}"')
        key, raw = con.execute('SELECT observation_id,payload FROM closing_observations LIMIT 1').fetchone()
        value = json.loads(raw)
        value['line_clv'] = 9.0
        con.execute('UPDATE closing_observations SET payload=? WHERE observation_id=?', (json.dumps(value), key))
    with pytest.raises(ValueError, match='CLOSING_OBSERVATION_CHANGED'):
        clv.read_observations(db)


def test_database_without_closes_reports_zero(tmp_path):
    path = tmp_path / 'SYNTHETIC-empty.sqlite3'
    sqlite3.connect(path).close()
    assert clv.report(clv.read_observations(path))['overall'] == dict(n=0)


def test_scoreboard_cli_appends_clv_from_an_evidence_database(db, tmp_path):
    from scripts import scoreboard as cli
    out = tmp_path / 'board.md'
    assert cli.main(['--markdown', str(out), '--evidence-db', str(db)]) == 0
    assert '### Closing-line value' in out.read_text() and '| All | 4 |' in out.read_text()
