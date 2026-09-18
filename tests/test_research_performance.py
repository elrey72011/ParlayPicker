from copy import deepcopy
from datetime import datetime, timezone
from core.research_performance import build

NOW = datetime(2026,9,18,tzinfo=timezone.utc)
def row(**kw):
    return dict(dict(sport='MLB',matchup_id='one',selected_as_best_pick=True,
        prediction_generated_at='2026-09-16T12:00:00Z',capture_recorded_at='2026-09-16T12:01:00Z',
        game_start_utc='2026-09-16T23:00:00Z',odds_recorded_at='2026-09-16T11:59:00Z',
        outcome_recorded_at='2026-09-17T03:00:00Z',result_source='ESPN',result_provider_event_id='123',
        quote_binding_verified=True,best_pick='Under 8.5',total_line=8.5,market_type='total_under',consensus_agreement='Neutral',
        candidate_outcome='WIN'),**kw)

def test_research_needs_no_model_or_wager_authority_and_does_not_mutate():
    rows=[row()]; before=deepcopy(rows)
    report=build(rows,NOW)
    assert report['eligible_games']==1
    assert report['wager_authority'] is report['ranking_authority'] is False
    assert rows==before

def test_latest_pending_cannot_fall_back_to_old_winner():
    old=row(); new=row(capture_recorded_at='2026-09-16T12:02:00Z',candidate_outcome='N/A')
    assert build([old,new],NOW)['eligible_games']==0

def test_ambiguous_duplicates_sport_isolation_push_and_original_consensus():
    r=row()
    assert build([r,r],NOW)['eligible_games']==1
    assert build([r,row(candidate_outcome='LOSS')],NOW)['eligible_games']==0
    result=build([r,row(sport='WNBA',candidate_outcome='PUSH')],NOW)
    assert result['eligible_games']==2
    assert len(result['buckets'])==2
    assert build([row(consensus_agreement=None)],NOW)['eligible_games']==0

def test_pregame_provider_and_quote_requirements():
    for kw in [dict(capture_recorded_at='2026-09-17T01:00:00Z'),dict(result_source=None),
               dict(result_provider_event_id=None),dict(quote_binding_verified=False),
               dict(odds_recorded_at='2026-09-16T13:00:00Z'),
               dict(outcome_recorded_at='2026-09-19T03:00:00Z')]:
        assert build([row(**kw)],NOW)['eligible_games']==0

def test_rebuild_uses_recorded_capture_time_and_verified_materialization(monkeypatch,tmp_path):
    import pandas as pd
    from app_core.prediction_evidence import connect
    from core.research_performance import rebuild
    from contextlib import closing
    dbpath=tmp_path/'evidence.sqlite3'
    with closing(connect(dbpath)) as db, db:
        db.execute('INSERT INTO bundles VALUES (?,?,?)',('v','2026-09-15T00:00:00Z','{}'))
        db.execute('INSERT INTO snapshots VALUES (?,?,?,?,?,?,?)',('s','v','2026-09-17T01:00:00Z','','','',''))
    # Even a pregame timestamp in the candidate cannot repair a late saved snapshot.
    monkeypatch.setattr('app_core.prediction_evidence.materialize', lambda _: (pd.DataFrame([dict(row(),snapshot_id='s')]),pd.DataFrame()))
    result=rebuild(dbpath)
    assert result['eligible_games']==0
    assert result['exclusions']['unverified_pregame_capture_or_future_game']==1
