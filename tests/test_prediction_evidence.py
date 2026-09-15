import hashlib
import json
import sqlite3

import pandas as pd
import pytest

from app_core import prediction_evidence as evidence
from core.selector_validation import build_report


def fixture_frames():
    quotes = json.dumps([
        {"book": "novig", "market_type": "total_over", "point": 8.5, "price": -110,
         "recorded_at": "2026-09-03T14:00:00Z"},
        {"book": "novig", "market_type": "total_under", "point": 8.5, "price": -110,
         "recorded_at": "2026-09-03T14:00:00Z"},
    ])
    audit = pd.DataFrame([
        {"matchup_id": "game-1", "league": "MLB", "home_team": "New York Yankees", "away_team": "Boston Red Sox",
         "game_date": "2026-09-03", "game_start_utc": "2026-09-03T23:00:00Z",
         "export_run_id": "20260903T140000Z", "market_type": kind, "best_pick": pick,
         "best_available_selected": selected, "best_available_candidate_count": 2,
         "odds_american": -110, "odds_source": "odds_api", "opposing_odds_source": "novig",
         "total_line": 8.5, "calibrated_probability": p, "market_probability": .5,
         "provider_quotes": quotes, "wager_approved": False}
        for kind, pick, selected, p in [("total_over", "Over 8.5", True, .6), ("total_under", "Under 8.5", False, .4)]
    ])
    final = audit[audit.best_available_selected].copy()
    final["wager_approved"] = True
    final["Kelly_Bet_Size"] = 12.0
    final["Pick_Status"] = "Actionable"
    return audit, final


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "model.py").write_text("version = 1\n")
    (root / "core").mkdir()
    (root / "core/selector_validation.py").write_bytes((evidence.ROOT / "core/selector_validation.py").read_bytes())
    db = tmp_path / "store" / "evidence.sqlite3"
    monkeypatch.setattr(evidence, "now_utc", lambda: "2026-09-01T12:00:00Z")
    context = evidence.begin_run({"use_ml": False}, path=db, root=root)
    monkeypatch.setattr(evidence, "now_utc", lambda: "2026-09-03T15:00:00.123456Z")
    return context, db, root


def test_capture_grade_report_loop_keeps_original_predictions(frozen):
    context, db, _ = frozen
    audit, final = fixture_frames()
    saved, card = evidence.capture_run(context, audit, final, audit, path=db)
    assert saved.loc[0, "wager_approved"]
    assert saved.loc[0, "odds_recorded_at"] == "2026-09-03T14:00:00+00:00"
    assert card.iloc[0].Kelly_Bet_Size == 12
    before = sqlite3.connect(db).execute("SELECT candidates,payload_hash FROM snapshots").fetchone()
    scores = card[["snapshot_id", "matchup_id"]].copy()
    scores["actual_home_score"], scores["actual_away_score"] = 6, 4
    assert evidence.record_scores(scores, path=db) == 1
    assert evidence.record_scores(scores, path=db) == 0
    graded, decisions = evidence.materialize(db)
    assert graded.candidate_outcome.tolist() == ["WIN", "LOSS"]
    r = build_report(graded, train_through="2026-09-02", selections=decisions)
    assert r["inventory"]["eligible_events"] == 1
    assert r["comparisons"]["qualified_wagers"]["selector"]["wins"] == 1
    assert before == sqlite3.connect(db).execute("SELECT candidates,payload_hash FROM snapshots").fetchone()
    assert "actual_home_score" not in pd.read_csv(__import__("io").StringIO(before[0]))


def test_bundle_reused_until_artifact_or_controls_change(frozen, monkeypatch):
    context, db, root = frozen
    again = evidence.begin_run({"use_ml": False}, path=db, root=root)
    assert again["model_version"] == context["model_version"]
    assert again["frozen_at"] == context["frozen_at"]
    (root / "model.py").write_text("version = 2\n")
    changed = evidence.begin_run({"use_ml": False}, path=db, root=root)
    assert changed["model_version"] != context["model_version"]
    a, f = fixture_frames()
    with pytest.raises(ValueError, match="changed during analysis"):
        evidence.capture_run(context, a, f, a, path=db)


def test_database_is_append_only_and_reusing_run_is_rejected(frozen):
    context, db, _ = frozen
    a, f = fixture_frames()
    evidence.capture_run(context, a, f, a, path=db)
    with pytest.raises(ValueError, match="immutable"):
        evidence.capture_run(context, a, f, a, path=db)
    with evidence.connect(db) as con, pytest.raises(sqlite3.IntegrityError, match="append-only"):
        con.execute("DELETE FROM snapshots")


def test_wrong_price_line_or_book_cannot_inherit_quote_time():
    a, _ = fixture_frames()
    for column, value in [("odds_american", -120), ("total_line", 9.5), ("opposing_odds_source", "draftkings")]:
        row = a.iloc[0].copy()
        row[column] = value
        assert not evidence.bind_quote(row)["quote_binding_verified"]


def test_provider_time_is_preserved_and_missing_timestamp_not_invented():
    game = {"home_team": "A", "away_team": "B", "bookmakers": [{"key": "novig_us", "last_update": "2026-09-03T12:00:00Z",
             "markets": [{"key": "spreads", "last_update": "2026-09-03T12:01:00Z", "outcomes": [{"name": "A", "point": -1.5, "price": -110}]}]}]}
    quote = json.loads(evidence.provider_quotes(game))[0]
    assert quote["recorded_at"] == "2026-09-03T12:01:00Z"
    assert quote["book"] == "novig" and quote["market_type"] == "spread_home"
    game["bookmakers"][0].pop("last_update")
    game["bookmakers"][0]["markets"][0].pop("last_update")
    assert json.loads(evidence.provider_quotes(game))[0]["recorded_at"] is None


def test_score_corrections_are_revisions_and_unknown_ids_do_nothing(frozen):
    context, db, _ = frozen
    a, f = fixture_frames()
    _, final = evidence.capture_run(context, a, f, a, path=db)
    scores = final[["snapshot_id", "matchup_id"]].copy()
    scores["actual_home_score"], scores["actual_away_score"] = 6, 4
    evidence.record_scores(scores, path=db)
    scores["actual_home_score"], scores["actual_away_score"] = 2, 1
    evidence.record_scores(scores, path=db)
    assert evidence.materialize(db)[0].candidate_outcome.tolist() == ["LOSS", "WIN"]
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM score_revisions").fetchone()[0] == 2
    scores["snapshot_id"] = "not-a-snapshot"
    assert evidence.record_scores(scores, path=db) == 0


def test_pending_scores_do_not_become_losses(frozen):
    context, db, _ = frozen
    a, f = fixture_frames()
    evidence.capture_run(context, a, f, a, path=db)
    graded, _ = evidence.materialize(db)
    assert graded.candidate_outcome.tolist() == ["N/A", "N/A"]


def test_integer_lines_keep_push_semantics_unverified(frozen):
    context, db, _ = frozen
    a, f = fixture_frames()
    a["total_line"], f["total_line"] = 8, 8
    saved, _ = evidence.capture_run(context, a, f, a, path=db)
    assert saved.probability_semantics.eq("push_semantics_unverified").all()


def test_versioned_results_reject_wrong_day_and_ambiguous_doubleheaders():
    from app_core.results_ingestion import attach_results
    a, f = fixture_frames()
    f["snapshot_id"] = "snap"
    results = pd.DataFrame([{"league": "MLB", "home_team": "New York Yankees", "away_team": "Boston Red Sox",
                             "date": "2026-09-02", "home_score": 6, "away_score": 4}])
    assert attach_results(f, results).actual_home_score.isna().all()
    results["date"] = "2026-09-03"
    doubled = pd.concat([results, results.assign(home_score=2)], ignore_index=True)
    assert attach_results(f, doubled).actual_home_score.isna().all()
    assert attach_results(f, results).actual_home_score.iloc[0] == 6


def test_live_app_captures_the_final_guarded_card(frozen, monkeypatch):
    import streamlit_app as app
    import core.streamlit_pipeline as pipeline
    context, db, _ = frozen
    a, f = fixture_frames()
    for frame in (a, f):
        frame["expected_value"] = .12
        frame["edge"] = .1
        frame["decimal_odds"] = 1.909
        frame["line_consistency_flag"] = True
        frame["line_event_identity_match_flag"] = True
        frame["market_line_source"] = "live"
        frame["line_provenance_warning"] = ""
    f["Pick_Status"] = "Below Threshold"
    f["Kelly_Bet_Size"] = 0.0
    f["market_line_used"] = 8.5
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(db.parent))
    monkeypatch.setattr(evidence, "begin_run", lambda controls: context)
    monkeypatch.setattr(app, "run_analysis_pipeline", lambda **kwargs: (a.copy(), pd.DataFrame(), {}))

    def build(analysis, diagnostics_out=None):
        diagnostics_out["candidate_audit_df"] = a.copy()
        diagnostics_out["candidate_authority_df"] = a.copy()
        return f.copy()

    monkeypatch.setattr(pipeline, "build_best_picks_df", build)
    monkeypatch.setattr(app, "optimize_portfolio_allocation", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app, "generate_parlays", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app, "run_bankroll_simulation", lambda *args, **kwargs: {})
    monkeypatch.setattr(app, "_enrich_with_kalshi_safe", lambda frame: (frame, None))
    monkeypatch.setattr(app, "_recompute_consensus_from_kalshi", lambda frame, require_ml=False: frame)
    controls = {"sports": ["MLB"], "use_ml": False, "use_gemini": False, "bankroll": 1000,
                "theover_spreads": None, "theover_totals": None}
    state, warnings, errors = app._run_pipeline(controls)
    assert state["diagnostics"].get("prediction_snapshot_saved"), warnings
    saved_audit, saved_final = evidence.materialize(db)
    assert saved_final.iloc[0].Kelly_Bet_Size == state["best_picks_df"].iloc[0].Kelly_Bet_Size
    assert saved_final.iloc[0].wager_approved == state["best_picks_df"].iloc[0].wager_approved
    assert saved_final.iloc[0].snapshot_id == context["snapshot_id"]


def test_reports_regenerated_from_frozen_bundle_after_grading(frozen):
    context, db, _ = frozen
    a, f = fixture_frames()
    _, final = evidence.capture_run(context, a, f, a, path=db)
    scores = final[["snapshot_id", "matchup_id"]].copy()
    scores["actual_home_score"], scores["actual_away_score"] = 6, 4
    evidence.record_scores(scores, path=db)
    paths = evidence.write_validation_reports(db)
    from pathlib import Path
    report = json.loads(Path(paths[0]).with_suffix(".json").read_text())
    assert report["configuration"]["train_through"] == "2026-09-01"
    assert report["inventory"]["eligible_events"] == 1
    assert report["evidence"]["preregistered"]
    assert report["reproducibility"]["snapshot_hashes"][0][0] == context["snapshot_id"]


def test_performance_refresh_records_only_settled_snapshot_results(frozen, monkeypatch):
    import app_core.performance_pipeline as performance
    context, db, _ = frozen
    a, f = fixture_frames()
    _, final = evidence.capture_run(context, a, f, a, path=db)
    results = pd.DataFrame([{"league": "MLB", "home_team": "New York Yankees", "away_team": "Boston Red Sox",
                             "date": "2026-09-03", "home_score": 6, "away_score": 4}])
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(db.parent))
    monkeypatch.setattr(performance, "fetch_yesterdays_results", lambda *args, **kwargs: results)
    graded = performance.grade_picks_with_live_results(final)
    assert graded.attrs["prediction_score_revisions_saved"] == 1
    assert graded.attrs["prediction_validation_reports"]
    assert evidence.materialize(db)[0].candidate_outcome.tolist() == ["WIN", "LOSS"]


def test_explicit_push_capture_settlement_and_conditional_scoring(frozen):
    context, db, _ = frozen
    a, f = fixture_frames()
    for frame in (a, f):
        frame["total_line"] = 8
        frame["best_pick"] = frame.market_type.map({"total_over": "Over 8.0", "total_under": "Under 8.0"})
        frame["provider_quotes"] = frame.provider_quotes.str.replace('8.5', '8.0', regex=False)
        frame["probability_semantics"] = "win_unconditional_with_push"
        frame["push_probability"] = .1
        frame["market_push_probability"] = .1
        frame["calibrated_probability"] = frame.market_type.map({"total_over": .54, "total_under": .36})
        frame["market_probability"] = .45
    saved, card = evidence.capture_run(context, a, f, a, path=db)
    assert saved.probability_semantics.eq("win_unconditional_with_push").all()
    scores = card[["snapshot_id", "matchup_id"]].copy()
    scores["actual_home_score"], scores["actual_away_score"] = 5, 3
    evidence.record_scores(scores, path=db)
    graded, decisions = evidence.materialize(db)
    assert graded.candidate_outcome.eq("PUSH").all()
    report, eligible = build_report(graded, train_through="2026-09-02", selections=decisions, return_eligible=True)
    assert report["inventory"]["eligible_events"] == 1
    assert eligible.iloc[0]._probability == pytest.approx(.6)
    assert eligible.iloc[0]._market == pytest.approx(.5)
    metrics = report["comparisons"]["qualified_wagers"]["selector"]
    assert metrics["pushes"] == 1 and metrics["wins"] == 0 and metrics["losses"] == 0


def test_final_rejected_line_preserved_despite_exact_quote(frozen):
    context, db, _ = frozen
    a, f = fixture_frames()
    for frame in (a, f):
        frame.loc[frame.best_available_selected, "best_pick"] = "Total line unresolved"
    f["market_line_source"] = "rejected_live"
    saved, _ = evidence.capture_run(context, a, f, a, path=db)
    selected = saved[saved.best_available_selected].iloc[0]
    assert selected.quote_binding_verified
    assert selected.final_line_rejected
    saved["candidate_outcome"] = "WIN"
    report = build_report(saved, train_through="2026-09-02")
    assert report["inventory"]["eligible_events"] == 0


def test_capture_marks_unselected_rejected_line(frozen):
    context, db, _ = frozen
    audit, final = fixture_frames()
    audit.loc[~audit.best_available_selected, "odds_source"] = "rejected_live_spread_price"
    captured, _ = evidence.capture_run(context, audit, final, audit, path=db)
    assert captured.loc[~captured.best_available_selected, "final_line_rejected"].all()
    assert not captured.loc[captured.best_available_selected, "final_line_rejected"].any()

def test_review_metadata_survives_capture_and_grading(frozen):
    context, db, _ = frozen
    audit, final = fixture_frames()
    final['gemini_agreement'] = 'agree'
    final['gemini_reviewed_at'] = '2026-09-03T14:30:00Z'
    final['gemini_review_model'] = 'test-model'
    final['gemini_review_input_hash'] = 'a' * 64
    saved, card = evidence.capture_run(context, audit, final, audit, path=db)
    assert saved.loc[0, 'gemini_agreement'] == 'agree'
    scores = card[['snapshot_id', 'matchup_id']].copy()
    scores['actual_home_score'], scores['actual_away_score'] = 6, 4
    evidence.record_scores(scores, path=db)
    graded, _ = evidence.materialize(db)
    from app_core.gemini_review_comparison import review_comparison
    report = review_comparison(graded)
    assert report.Selections.tolist() == [1, 1]
    assert report.Wins.tolist() == [1, 1]


@pytest.mark.parametrize('field,expected', [
    ('quote_binding_verified', True),
    ('probability_semantics', 'win_conditional_on_decision'),
    ('game_start_utc', '2026-09-03T23:00:00+00:00'),
])
def test_authoritative_binding_and_derived_fields_survive(frozen, field, expected):
    from app_core.candidate_evidence_schema import project
    context, db, _ = frozen
    a, f = fixture_frames()
    a['candidate_id'] = ['over', 'under']
    a['game_start_utc'] = None
    a['game_time_est'] = '2026-09-03 07:00 PM'
    a['probability_semantics'] = None
    prepared = project(a)
    f = prepared[prepared.best_available_selected].copy()
    saved, _ = evidence.capture_run(context, prepared, f, a, path=db, authoritative_candidates=True)
    assert saved.iloc[0][field] == expected


def binding_row():
    a, _ = fixture_frames()
    row = a.iloc[0].to_dict()
    row.update(book='novig', provider_quotes=json.dumps([dict(book='novig',
        market_type='total_over', point=8.5, price=-110,
        recorded_at='2026-09-03T14:00:00Z', provider_event_id='evt', provider_namespace='odds_api')]))
    return row


def test_preserve_complete_binding_without_rebinding():
    row = binding_row()
    row.update(quote_binding_verified=True, odds_recorded_at='2026-09-03T13:59:00Z',
        quote_bookmaker='novig', provider_event_id='original', provider_namespace='odds_api')
    assert evidence.ensure_authoritative_quote_binding(row) == dict(row, book='Novig', quote_bookmaker='Novig')


def test_bind_unique_exact_quote_uses_provider_time():
    row = binding_row()
    bound = evidence.ensure_authoritative_quote_binding(row)
    assert bound['quote_binding_verified'] is True
    assert bound['odds_recorded_at'] == '2026-09-03T14:00:00+00:00'
    assert bound['quote_bookmaker'] == 'Novig'
    assert bound['provider_event_id'] == 'evt'
    assert bound['provider_namespace'] == 'odds_api'
    assert evidence.ensure_authoritative_quote_binding(bound) == bound
    assert 'model_validated' not in bound and 'calibration_validated' not in bound
    assert bound['book'] == 'Novig'
    assert bound['provider_quotes'] == row['provider_quotes']
    for key in ('odds_american', 'total_line'):
        assert bound[key] == row[key]


@pytest.mark.parametrize('change', [
    {'total_line':9.5}, {'odds_american':-120}, {'book':'draftkings'},
    {'provider_quotes':'[]'}, {'provider_event_id':'wrong', 'provider_namespace':'odds_api'},
    {'quote_time':'2026-09-03T13:59:00Z'}, {'line':9.5},
])
def test_binding_mismatch_or_missing_fails_closed(change):
    row = dict(binding_row(), **change)
    row['exact_quote_verified'] = True  # Cannot bypass a failed canonical binding.
    bound = evidence.ensure_authoritative_quote_binding(row)
    assert bound['quote_binding_verified'] is False
    assert bound['exact_quote_verified'] is False


def test_ambiguous_quotes_do_not_choose_a_provider():
    row = binding_row()
    quotes = json.loads(row['provider_quotes'])
    quotes.append(dict(quotes[0], provider_event_id='second'))
    row['provider_quotes'] = json.dumps(quotes)
    bound = evidence.ensure_authoritative_quote_binding(row)
    assert bound['quote_binding_verified'] is False
    assert not bound.get('odds_recorded_at')
    assert not bound.get('provider_event_id')


@pytest.mark.parametrize('semantics,push', [('win_unconditional_with_push', .1), ('win_conditional_on_decision', 0)])
def test_authoritative_explicit_facts_and_capture_metadata(frozen, semantics, push):
    from app_core.candidate_evidence_schema import project
    context, db, _ = frozen
    a, _ = fixture_frames()
    a['candidate_id'] = ['over', 'under']
    a['probability_semantics'] = semantics
    a['push_probability'] = push
    a['market_push_probability'] = push
    a['model_validated'] = False
    a['critical_feature_error'] = False
    a['prior_clv_lower'] = 0
    for key in ('snapshot_id','export_run_id','created_process_id','decision_bundle_version'):
        a[key] = 'cannot_override_capture'
    prepared = project(a)
    f = prepared[prepared.best_available_selected].copy()
    saved, _ = evidence.capture_run(context, prepared, f, a, path=db, authoritative_candidates=True)
    row = saved.iloc[0]
    assert row['probability_semantics'] == semantics
    assert row['game_start_utc'] == '2026-09-03T23:00:00Z'
    assert not bool(row['model_validated']) and not bool(row['critical_feature_error'])
    assert row['prior_clv_lower'] == 0
    assert row['push_probability'] == push
    assert row['snapshot_id'] == context['snapshot_id']
    assert row['created_process_id'] == evidence.PROCESS_INSTANCE
    assert row['decision_bundle_version'] == context['model_version']
    assert row['export_run_id'] != 'cannot_override_capture'


@pytest.mark.parametrize('control', ['ambiguous', 'stale', 'validation', 'policy', 'moneyline', 'veto', 'legacy'])
def test_bound_quote_does_not_create_wager_authority(tmp_path, control):
    from activation_fixture import setup, NOW
    from dataclasses import replace
    from core.live_wager_contract import finalize_live_wagers
    row, policy, config = setup(tmp_path/'ledger.db')
    row.update(odds_source='DraftKings', provider_quotes=json.dumps([dict(book='draftkings',
        market_type='spread_home', point=-2.5, price=-110,
        recorded_at=row['quote_time'], provider_namespace='odds_api', provider_event_id='one')]))
    if control == 'ambiguous':
        row['provider_quotes'] = json.dumps(json.loads(row['provider_quotes'])*2)
    elif control == 'stale':
        row['quote_time'] = '2026-09-14T13:00:00+00:00'
        quotes=json.loads(row['provider_quotes']);quotes[0]['recorded_at']=row['quote_time'];row['provider_quotes']=json.dumps(quotes)
    elif control == 'validation': row['model_validated'] = False
    elif control == 'policy': policy=replace(policy,deployment_state='UNVALIDATED')
    elif control == 'moneyline': row['market_type']='moneyline_home'
    elif control == 'veto': row['gemini_review_status']='HARD_VETO'
    elif control == 'legacy': row.update(model_validated=False, Pick_Status='APPROVED', wager_approved=True)
    bound=evidence.ensure_authoritative_quote_binding(row)
    frame=pd.DataFrame([bound])
    result,_=finalize_live_wagers(frame,frame,1000,now=NOW,policies={'NFL':policy},config=config,reviews=frame)
    assert result.iloc[0]['production_bet_amount'] == 0


@pytest.mark.parametrize('null', [None, float('nan'), pd.NA, ''])
def test_projected_null_derivation_keeps_false_zero_and_binding(frozen, null):
    from app_core.candidate_evidence_schema import project
    context, db, _ = frozen
    a, _ = fixture_frames()
    a['candidate_id'] = ['over', 'under']
    a['game_start_utc'] = null
    a['game_time_est'] = '2026-09-03 07:00 PM'
    a['probability_semantics'] = null
    a['quote_binding_verified'] = False
    a['critical_feature_error'] = False
    a['push_probability'] = 0
    prepared = project(a)
    f=prepared[prepared.best_available_selected].copy()
    saved,_=evidence.capture_run(context,prepared,f,a,path=db,authoritative_candidates=True)
    assert bool(saved.iloc[0]['quote_verified'])
    assert bool(saved.iloc[0]['quote_binding_verified'])
    assert saved.iloc[0]['probability_semantics'] == 'win_conditional_on_decision'
    assert saved.iloc[0]['game_start_utc'] == '2026-09-03T23:00:00+00:00'
    assert not bool(saved.iloc[0]['critical_feature_error'])
    assert saved.iloc[0]['push_probability'] == 0


def test_binding_without_provider_timestamp_does_not_invent_time():
    row=binding_row()
    quotes=json.loads(row['provider_quotes']);quotes[0].pop('recorded_at')
    quotes[0]['observed_at']='2026-09-03T14:00:00Z'
    row['provider_quotes']=json.dumps(quotes)
    assert evidence.ensure_authoritative_quote_binding(row)['quote_binding_verified'] is False


def test_own_verified_book_is_not_replaced_by_opposing_price_source():
    row=binding_row()
    row.update(quote_binding_verified=True, odds_recorded_at='2026-09-03T14:00:00Z',
               quote_bookmaker='novig', opposing_odds_source='draftkings')
    assert evidence.ensure_authoritative_quote_binding(row) == dict(row, book='Novig', quote_bookmaker='Novig')
    row['quote_binding_verified']=False
    bound=evidence.ensure_authoritative_quote_binding(row)
    assert bound['quote_binding_verified'] is True
    assert bound['quote_bookmaker'] == 'Novig'
    assert bound['opposing_odds_source'] == 'draftkings'


@pytest.mark.parametrize('raw,label', [
    ('draftkings','DraftKings'), ('fanduel','FanDuel'), ('betmgm','BetMGM'),
    ('novig','Novig'), ('novig_us','Novig'), ('DraftKings','DraftKings'),
    ('FanDuel','FanDuel'), ('BetMGM','BetMGM'), ('Novig','Novig'),
    (' DRAFTKINGS ','DraftKings'), ('fAnDuEl','FanDuel'),
])
def test_raw_sportsbook_binding_policy_identity(raw, label):
    from app_core.public_quote_policy import canonical_book_label, supported_quote
    from core.live_wager_contract import adapt_candidate
    row=binding_row()
    row.pop('book')
    row.pop('opposing_odds_source')
    row['league']='NFL'
    quotes=json.loads(row['provider_quotes']);quotes[0]['book']=raw
    row['provider_quotes']=json.dumps(quotes)
    bound=evidence.ensure_authoritative_quote_binding(row)
    assert canonical_book_label(raw) == label
    assert bound['quote_binding_verified'] is True
    assert bound['quote_bookmaker'] == label
    assert bound['provider_quotes'] == row['provider_quotes']
    assert bound['provider_namespace'] == 'odds_api'
    assert bound['provider_event_id'] == 'evt'
    adapted=adapt_candidate(bound)
    assert adapted['book'] == label
    assert adapted['exact_quote_verified'] is True
    assert supported_quote({'sport':'NFL','quote_source':adapted['book']})


@pytest.mark.parametrize('raw', ['draftkings','fanduel','betmgm'])
@pytest.mark.parametrize('sport', ['MLB','NBA','WNBA','NHL','NCAAB'])
def test_canonicalization_does_not_expand_sport_fallback(raw, sport):
    from app_core.public_quote_policy import canonical_book_label, supported_quote
    assert not supported_quote({'sport':sport,'quote_source':raw})
    assert not supported_quote({'sport':sport,'quote_source':canonical_book_label(raw)})


@pytest.mark.parametrize('sport', ['NFL','NCAAF'])
@pytest.mark.parametrize('raw', ['draftkings','fanduel','betmgm','novig','novig_us'])
def test_existing_football_book_policy_handles_provider_keys(sport, raw):
    from app_core.public_quote_policy import supported_quote
    assert supported_quote({'sport':sport,'quote_source':raw})


@pytest.mark.parametrize('raw', ['DK','FD','MGM','Caesars','ESPN BET','odds_api','espn',None,''])
def test_unknown_books_and_provider_namespaces_do_not_authorize(raw):
    from app_core.public_quote_policy import canonical_book_label, supported_quote
    assert canonical_book_label(raw) not in {'DraftKings','FanDuel','BetMGM','Novig'}
    for sport in ('NFL','NCAAF','MLB'):
        assert not supported_quote({'sport':sport,'quote_source':raw})


@pytest.mark.parametrize('sport,book,basis,expected', [
    ('NCAAF','draftkings','espn_observed',True),
    ('NFL','draftkings','espn_observed',False),
    ('MLB','draftkings','espn_observed',False),
    ('NCAAF','fanduel','espn_observed',False),
    ('NCAAF','novig','espn_observed',False),
    ('NCAAF','draftkings','provider',False),
    ('NCAAF','draftkings',None,False),
])
def test_observed_quote_policy_retains_all_conditions(sport, book, basis, expected):
    from app_core.public_quote_policy import supported_quote
    assert supported_quote({'sport':sport,'quote_source':book,'quote_time_basis':basis}) is expected


@pytest.mark.parametrize('raw,label', [('draftkings','DraftKings'),('DraftKings','DraftKings'),('novig_us','Novig')])
def test_prebound_book_normalizes_without_changing_quote(raw, label):
    row=binding_row()
    row.pop('opposing_odds_source')
    row.update(book=raw, quote_bookmaker=raw, quote_binding_verified=True,
        odds_recorded_at='2026-09-03T12:00:00Z', provider_namespace='odds_api', provider_event_id='original')
    bound=evidence.ensure_authoritative_quote_binding(row)
    assert bound == dict(row, book=label, quote_bookmaker=label)


@pytest.mark.parametrize('value', ['absent', None, '', float('nan'), pd.NA, pd.NaT,
    '2026-09-03T14:30:00+00:00', 'not-a-timestamp'], ids=['absent','none','empty','nan','pdNA','NaT','producer','invalid'])
def test_capture_prediction_timestamp_missing_values(frozen, value):
    from core.run_readiness import build_readiness
    context, db, _ = frozen
    audit, final = fixture_frames()
    audit['candidate_id'] = ['over','under']
    final['candidate_id'] = ['over']
    absent = isinstance(value,str) and value == 'absent'
    if not absent:
        for frame in (audit, final):
            frame['prediction_generated_at'] = value
    saved, card = evidence.capture_run(context, audit, final, audit, path=db, authoritative_candidates=True)
    expected = '2026-09-03T15:00:00.123456Z' if absent or value is None or value is pd.NA or value is pd.NaT or (isinstance(value,float) and pd.isna(value)) or value == '' else value
    _, loaded, decisions = evidence.load_snapshots(db)[0]
    for frame in (saved, card, loaded, decisions):
        assert frame.prediction_generated_at.eq(expected).all()
    blockers = build_readiness(loaded, decisions)['games'][0]['evidence_blockers']
    assert 'model_provenance_missing' in blockers
    for col in ('model_version','model_trained_through','model_available_at'):
        assert col not in loaded or loaded[col].isna().all()
    if expected == 'not-a-timestamp':
        assert 'prediction_or_start_time_unverified' in blockers
        assert 'export_timing_unverified' in blockers
    else:
        assert 'prediction_or_start_time_unverified' not in blockers
        assert 'export_timing_unverified' not in blockers
        for _, row in loaded.iterrows():
            prediction = pd.Timestamp(row.prediction_generated_at)
            export = pd.to_datetime(row.export_run_id,format='%Y%m%dT%H%M%S.%fZ',utc=True)
            assert prediction <= export < pd.Timestamp(row.game_start_utc)
    with evidence.connect(db) as con:
        before = con.execute('SELECT candidates,decisions,payload_hash FROM snapshots').fetchall()
    with pytest.raises(ValueError,match='immutable snapshot'):
        evidence.capture_run(context,audit,final,audit,path=db,authoritative_candidates=True)
    with evidence.connect(db) as con:
        assert con.execute('SELECT candidates,decisions,payload_hash FROM snapshots').fetchall() == before


def test_capture_prediction_timestamp_mixed_rows(frozen):
    context, db, _ = frozen
    audit, final = fixture_frames()
    audit['prediction_generated_at'] = ['2026-09-03T14:30:00+00:00', None]
    final['prediction_generated_at'] = ['2026-09-03T14:30:00+00:00']
    saved, card = evidence.capture_run(context,audit,final,audit,path=db)
    assert saved.prediction_generated_at.tolist() == ['2026-09-03T14:30:00+00:00','2026-09-03T15:00:00.123456Z']
    assert card.prediction_generated_at.iloc[0] == '2026-09-03T14:30:00+00:00'
