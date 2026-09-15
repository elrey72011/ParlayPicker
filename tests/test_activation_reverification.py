import pandas as pd
import pytest
from app_core.candidate_evidence_schema import project, pool_status, FIELDS
from core.activation_studies import studies

pytestmark = pytest.mark.activation_acceptance


def test_per_game_pool_preserves_expected_missing_candidates():
    rows=pd.DataFrame([dict(snapshot_id='s',matchup_id=g,league='NFL',market_type=m,best_pick=m,odds_american=-110,best_available_candidate_count=2) for g,m in [('a','spread_home'),('a','spread_away'),('b','total_over')]])
    saved=project(rows)
    assert set(FIELDS)<=set(saved)
    assert pool_status(saved[saved.matchup_id.eq('a')])
    assert not pool_status(saved[saved.matchup_id.eq('b')])
    assert saved.slate_id.isna().all()
    assert saved.model_trained_through.isna().all()


def test_different_line_candidates_have_distinct_identity():
    rows=pd.DataFrame([dict(snapshot_id='s',matchup_id='g',market_type='total_over',best_pick='Over',total_line=v,odds_american=-110) for v in [8,8.5]])
    assert project(rows).candidate_id.nunique()==2


def test_missing_study_inputs_do_not_pass():
    row=dict(sport='NFL',market_type='spread_home',candidate_outcome='WIN',conservative_probability=.6)
    result=studies([row],{})
    assert 'missing_probability_intervals' in result['uncertainty']['blockers']
    assert result['prior_validation']['blockers']
    assert result['ml_ablation']['with_ml']['blocker']
    assert result['premium']['n']==0


def test_study_reports_band_coverage_without_binary_interval_claim():
    rows=[dict(sport='NFL',market_type='spread_home',candidate_outcome=o,conservative_probability=.6,probability_interval_lower=.5,probability_interval_upper=.9) for o in ['WIN','WIN','LOSS']]
    r=studies(rows,{'uncertainty_quantile':.1})
    assert r['uncertainty']['lower_coverage']==1
    assert 'not_individual_binary' in r['uncertainty']['method']


def test_recommendation_after_commit_cannot_erase_exposure(tmp_path):
    from activation_fixture import setup, NOW
    from core.exposure_ledger import append, snapshot
    ledger=tmp_path/'l.db';setup(ledger)
    e={'status':'COMMITTED','bet_id':'x','source_snapshot_id':'s','sportsbook':'DraftKings','stake_dollars':2.,'legs':[{'sport':'NFL','game_id':'g','team_ids':['a','b'],'market':'spread_home','selection':'a -2.5','line':-2.5,'odds':-110}]}
    append(ledger,e,confirmed=True,now=NOW)
    append(ledger,dict(e,status='RECOMMENDED'),confirmed=True,now=NOW)
    assert snapshot(ledger,now=NOW)['committed']['total']==.002
    append(ledger,{'status':'SETTLED','bet_id':'x'},confirmed=True,now=NOW)
    assert snapshot(ledger,now=NOW)['committed']['total']==0


def test_changed_actual_line_does_not_reuse_probability():
    from core.owner_wager_records import placement_record
    c={'sport':'NFL','game_id':'g','market_type':'spread_home','selection':'a -2.5','line':-2.5,'odds':-110,'conservative_probability':.6}
    r=placement_record(c,sportsbook='DraftKings',line=-4.5,odds=-110,stake=1,bet_id='b',snapshot_id='s',team_ids=['a','b'])
    assert r['actual_conservative_ev'] is None
    assert r['value_warning']


def test_automatic_close_uses_provider_namespace_and_all_directions(tmp_path,monkeypatch):
    from datetime import timedelta
    from activation_fixture import NOW
    from app_core import prediction_evidence as pe
    from app_core.activation_closing import capture_live,observations
    import json
    row={'snapshot_id':'s','candidate_id':'c','sport':'MLB','matchup_id':'g','game_id':'g','market_type':'spread_home','game_start_utc':(NOW+timedelta(minutes=10)).isoformat(),'provider_namespace':'odds_api','provider_event_id':'p','quote_bookmaker':'novig','market_line_used':-1.5,'odds_american':-110}
    monkeypatch.setattr(pe,'load_snapshots',lambda _: [('s',pd.DataFrame([row]),pd.DataFrame())])
    q={'provider_namespace':'odds_api','provider_event_id':'p','book':'novig','market_type':'spread_home','point':-1.5,'price':-120,'recorded_at':NOW.isoformat()}
    fetch=lambda _:pd.DataFrame([{'matchup_id':'g','provider_quotes':json.dumps([q])}])
    db=tmp_path/'c.db'
    assert capture_live(db,fetch=fetch,now=NOW)['verified']==1
    q['provider_namespace']='espn'
    assert capture_live(db,fetch=fetch,now=NOW)['unavailable']==1
    assert len(observations(db))==1


def test_outcome_refresh_uses_deterministic_match_and_preserves_provider(tmp_path,monkeypatch):
    from app_core import prediction_evidence as pe
    row={'snapshot_id':'s','matchup_id':'g','sport':'MLB','league':'MLB','home_team':'Chicago Cubs','away_team':'Pittsburgh Pirates','game_start_utc':'2026-09-01T17:00:00+00:00','market_type':'spread_away','best_pick':'Pittsburgh Pirates +1.5','candidate_outcome':'PENDING'}
    monkeypatch.setattr(pe,'materialize',lambda _: (pd.DataFrame([row]),pd.DataFrame()))
    captured=[]
    monkeypatch.setattr(pe,'record_scores',lambda frame,**kw:captured.extend(frame.to_dict('records')) or len(frame))
    score={'sport':'MLB','home':'Chicago Cubs','away':'Pittsburgh Pirates','start':'2026-09-01T19:00:00+00:00','home_score':2,'away_score':1,'result_source':'ESPN','provider_event_id':'x','event_id':'x','completed':True}
    result=pe.refresh_outcomes(tmp_path/'x.db',fetch=lambda day,sports:{'recorded_at':'2026-09-02T00:00:00+00:00','scores':[score],'events':[score]})
    assert result['revisions']==1
    assert captured[0]['result_source']=='ESPN'
    assert captured[0]['result_provider_event_id']=='x'


def test_prospective_uncertainty_cannot_use_same_slate_or_other_sport():
    from core.prospective_uncertainty import forecast
    c={'sport':'NFL','season':2026,'slate_id':'NFL:2026:WEEK_03','market_type':'spread_home','calibrated_probability':.6,'prediction_generated_at':'2026-09-14T12:00:00Z','model_version':'m','calibration_version':'c'}
    r=dict(c,game_id='g',candidate_id='id',candidate_outcome='WIN',outcome_recorded_at='2026-09-13T22:00:00Z')
    settings={'historical_prior_decay':.8,'historical_effective_sample_cap':20,'current_season_weight':1,'parent_weight':.5,'uncertainty_quantile':.1}
    assert forecast(c,[r],settings)['uncertainty_status']=='NO_PRIOR_ADMISSIBLE_SLATES'
    assert forecast(c,[dict(r,sport='MLB',slate_id='past')],settings)['uncertainty_status']=='NO_PRIOR_ADMISSIBLE_SLATES'
    value=forecast(c,[dict(r,slate_id='NFL:2026:WEEK_02')],settings)
    assert value['conservative_probability']<=.6
    assert value['current_season_weight']==1
    assert value['uncertainty_status']=='PROSPECTIVE_ESTIMATE_NOT_DEPLOYMENT_AUTHORITY'


@pytest.fixture
def strict_close_snapshot(tmp_path, monkeypatch, request):
    """Persist through capture_run and load through the real immutable reader."""
    from datetime import timedelta
    from activation_fixture import NOW
    from app_core import prediction_evidence as pe
    root = tmp_path / 'repo'
    root.mkdir()
    monkeypatch.setattr(pe, 'now_utc', lambda: (NOW-timedelta(minutes=20)).isoformat())
    database = tmp_path / 'closing.db'
    context = pe.begin_run({}, path=database, root=root)
    book = getattr(request, 'param', 'DraftKings')
    row = dict(candidate_id='close-candidate', game_id='game-123', matchup_id='game-123',
        sport='NFL', league='NFL', home_team='Indianapolis Colts', away_team='Baltimore Ravens',
        game_date=NOW.date().isoformat(), game_start_utc=(NOW+timedelta(minutes=10)).isoformat(),
        market_type='spread_home', best_pick='Indianapolis Colts -2.5', spread_line=-2.5,
        market_line_used=-2.5, odds_american=-110, quote_bookmaker=book,
        quote_binding_verified=True, odds_recorded_at=(NOW-timedelta(minutes=20)).isoformat(),
        provider_namespace='odds_api', provider_event_id='provider:123',
        best_available_selected=True, best_available_candidate_count=1, wager_approved=False)
    frame = pd.DataFrame([row])
    pe.capture_run(context, frame, frame, frame, path=database, authoritative_candidates=True)
    candidate = pe.load_snapshots(database)[0][1].iloc[0].to_dict()
    assert candidate['quote_bookmaker'] == book
    return database, candidate


def test_strict_close_saved_canonical_raw_provider_e2e(strict_close_snapshot):
    import json
    from activation_fixture import NOW
    from app_core.activation_closing import capture_live, observations
    from app_core.prediction_evidence import load_snapshots
    database, candidate = strict_close_snapshot
    before = load_snapshots(database)[0][1].copy(deep=True)
    q = dict(book='draftkings', provider_namespace='odds_api', provider_event_id='provider:123',
        market_type='spread_home', point=-3., price=-120, recorded_at=NOW.isoformat())
    live = pd.DataFrame([dict(matchup_id='game-123', provider_quotes=json.dumps([q]))])
    raw_before = live.copy(deep=True)
    result = capture_live(database, fetch=lambda sports: live, now=NOW)
    assert result == {'verified':1, 'unavailable':0, 'reasons':{}}, result
    stored, = observations(database)
    assert stored['candidate_id'] == candidate['candidate_id']
    assert stored['snapshot_id'] == candidate['snapshot_id']
    assert stored['quote']['sportsbook'] == 'DraftKings'
    assert stored['quote']['provider_namespace'] == 'odds_api'
    assert stored['quote']['provider_event_id'] == 'provider:123'
    assert stored['quote']['quote_recorded_at'] == q['recorded_at']
    assert stored['quote_verified'] is True
    pd.testing.assert_frame_equal(live, raw_before)
    pd.testing.assert_frame_equal(load_snapshots(database)[0][1], before)


@pytest.mark.parametrize('strict_close_snapshot,raw', [
    ('DraftKings','draftkings'), ('FanDuel','fanduel'), ('BetMGM','betmgm'),
    ('Novig','novig'), ('Novig','novig_us'), ('DraftKings','DraftKings'),
], indirect=['strict_close_snapshot'])
def test_strict_closing_book_representations(strict_close_snapshot, raw):
    import json
    from activation_fixture import NOW
    from app_core.activation_closing import capture_live, observations
    database, c = strict_close_snapshot
    q = dict(book=raw, provider_namespace=c['provider_namespace'], provider_event_id=c['provider_event_id'],
        market_type=c['market_type'], point=-3., price=-120, recorded_at=NOW.isoformat())
    live = pd.DataFrame([dict(matchup_id=c['matchup_id'], provider_quotes=json.dumps([q]))])
    original = live.copy(deep=True)
    assert capture_live(database, fetch=lambda _: live, now=NOW) == {'verified':1,'unavailable':0,'reasons':{}}
    stored, = observations(database)
    assert stored['quote']['sportsbook'] == c['quote_bookmaker']
    assert stored['quote']['provider_event_id'] == c['provider_event_id']
    pd.testing.assert_frame_equal(original, live)


@pytest.mark.parametrize('change,reason', [
    ({'book':'fanduel'}, 'missing_or_ambiguous_exact_quote'),
    ({'book':'Caesars'}, 'missing_or_ambiguous_exact_quote'),
    ({'book':'DK'}, 'missing_or_ambiguous_exact_quote'),
    ({'provider_namespace':'espn'}, 'missing_or_ambiguous_exact_quote'),
    ({'provider_event_id':'different'}, 'missing_or_ambiguous_exact_quote'),
    ({'market_type':'spread_away'}, 'missing_or_ambiguous_exact_quote'),
    ({'wrong_matchup':True}, 'missing_or_ambiguous_exact_quote'),
    ({'missing':True}, 'missing_or_ambiguous_exact_quote'),
    ({'ambiguous':True}, 'missing_or_ambiguous_exact_quote'),
    ({'minutes':-31}, 'invalid_or_stale_close'),
    ({'minutes':10}, 'invalid_or_stale_close'),
    ({'recorded_at':'invalid'}, 'invalid_or_stale_close'),
])
def test_strict_close_rejects_non_book_failures(strict_close_snapshot, change, reason):
    import json
    from datetime import timedelta
    from activation_fixture import NOW
    from app_core.activation_closing import capture_live, observations
    database, c = strict_close_snapshot
    q = dict(book='draftkings', provider_namespace=c['provider_namespace'], provider_event_id=c['provider_event_id'],
        market_type=c['market_type'], point=-3., price=-120, recorded_at=NOW.isoformat())
    q.update(change)
    if 'minutes' in change:
        q['recorded_at'] = (NOW+timedelta(minutes=change['minutes'])).isoformat()
    quotes = [] if change.get('missing') else [q]
    if change.get('ambiguous'):
        quotes.append(dict(q, book='DraftKings', price=-125))
    live = pd.DataFrame([dict(matchup_id='wrong' if change.get('wrong_matchup') else c['matchup_id'], provider_quotes=json.dumps(quotes))])
    assert capture_live(database, fetch=lambda _: live, now=NOW) == {'verified':0,'unavailable':1,'reasons':{reason:1}}
    assert observations(database) == []


def test_record_close_normalizes_without_mutation_and_preserves_clv(strict_close_snapshot):
    from copy import deepcopy
    from activation_fixture import NOW
    from app_core.activation_closing import record_close, observations
    from core.clv import line_clv, price_clv
    database, c = strict_close_snapshot
    original = deepcopy(c)
    q = dict(game_id=c['game_id'], sport=c['sport'], market_type=c['market_type'],
        sportsbook='draftkings', provider_namespace=c['provider_namespace'], provider_event_id=c['provider_event_id'],
        quote_recorded_at=NOW.isoformat(), line=-2.5, price=-120)
    raw = deepcopy(q)
    key = record_close(database, c, q, captured_at=NOW.isoformat())
    # Equivalent representation produces the same immutable record, not a duplicate.
    assert record_close(database, c, dict(q,sportsbook='DraftKings'), captured_at=NOW.isoformat()) == key
    first, = observations(database)
    assert first['quote']['sportsbook'] == 'DraftKings'
    assert first['line_clv'] == line_clv(c['market_type'],-2.5,-2.5) == 0
    assert first['price_clv'] == price_clv(-110,-120)
    assert first['beat_close'] is True
    # Also accept a historical raw candidate identity without rewriting that candidate.
    record_close(database, dict(c,quote_bookmaker='draftkings'), dict(q,line=-3.), captured_at=NOW.isoformat())
    stored = observations(database)
    assert len(stored) == 2 and stored[0] == first
    assert stored[1]['line_clv'] == line_clv(c['market_type'],-2.5,-3.) == .5
    assert stored[1]['price_clv'] is None and stored[1]['beat_close'] is True
    assert q == raw
    assert c.keys() == original.keys()
    assert c['quote_bookmaker'] == original['quote_bookmaker']


@pytest.mark.parametrize('field,value', [
    ('sportsbook','FanDuel'), ('sportsbook','DK'), ('game_id','other'), ('sport','MLB'),
    ('market_type','spread_away'), ('provider_event_id','other'), ('provider_namespace','espn'),
])
def test_record_close_identity_guards(strict_close_snapshot, field, value):
    from activation_fixture import NOW
    from app_core.activation_closing import record_close, observations
    database, c = strict_close_snapshot
    q = dict(game_id=c['game_id'], sport=c['sport'], market_type=c['market_type'],
        sportsbook='draftkings', provider_namespace=c['provider_namespace'], provider_event_id=c['provider_event_id'],
        quote_recorded_at=NOW.isoformat(), line=-3., price=-120)
    q[field] = value
    with pytest.raises(ValueError, match='Closing provider namespace|Closing identity/book mismatch'):
        record_close(database, c, q, captured_at=NOW.isoformat())
    assert observations(database) == []


def test_closing_observations_are_append_only_and_digest_checked(strict_close_snapshot):
    import sqlite3
    from contextlib import closing
    from activation_fixture import NOW
    from app_core.activation_closing import record_close, observations
    from app_core.prediction_evidence import connect
    database, c = strict_close_snapshot
    q = dict(game_id=c['game_id'], sport=c['sport'], market_type=c['market_type'],
        sportsbook='draftkings', provider_namespace=c['provider_namespace'], provider_event_id=c['provider_event_id'],
        quote_recorded_at=NOW.isoformat(), line=-3., price=-120)
    record_close(database, c, q, captured_at=NOW.isoformat())
    with closing(connect(database)) as db:
        for sql in ('DELETE FROM closing_observations', "UPDATE closing_observations SET payload='{}'"):
            with pytest.raises(sqlite3.IntegrityError, match='append-only'):
                db.execute(sql)
        # A corrupt newly inserted payload cannot pass the reader's digest check.
        with db:
            db.execute('INSERT INTO closing_observations VALUES (?,?,?,?)', ('invalid-digest',c['snapshot_id'],c['candidate_id'],'{}'))
    with pytest.raises(ValueError, match='Closing payload changed'):
        observations(database)


def test_verified_quotes_cli_normalizes_books(strict_close_snapshot, tmp_path, capsys):
    import json
    from datetime import datetime, timedelta, timezone
    from scripts.capture_closing_lines import main
    from app_core.activation_closing import observations
    database, c = strict_close_snapshot
    now = datetime.now(timezone.utc)
    c = dict(c,game_start_utc=(now+timedelta(minutes=10)).isoformat())
    q = dict(game_id=c['game_id'], sport=c['sport'], market_type=c['market_type'],
        sportsbook='draftkings', provider_namespace=c['provider_namespace'], provider_event_id=c['provider_event_id'],
        quote_recorded_at=now.isoformat(), line=-3., price=-120)
    export = tmp_path/'export.csv'; quotes = tmp_path/'quotes.json'
    pd.DataFrame([c]).to_csv(export,index=False)
    quotes.write_text(json.dumps([q]),encoding='utf-8')
    assert main(['--export',str(export),'--verified-quotes',str(quotes),'--database',str(database)]) == 0
    assert 'Verified closing observations: 1; unavailable: 0' in capsys.readouterr().out
    assert observations(database)[0]['quote']['sportsbook'] == 'DraftKings'
    assert json.loads(quotes.read_text(encoding='utf-8')) == [q]
