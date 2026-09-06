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
