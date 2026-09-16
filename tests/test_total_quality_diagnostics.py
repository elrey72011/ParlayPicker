from copy import deepcopy
import pandas as pd
import pytest
from app_core.total_signal_quality import assess, attach, counters, VERSION
from app_core.price_value_display import display
from app_core.total_quality_report import read_candidates, summarize
from test_prediction_evidence import frozen, fixture_frames
from app_core import prediction_evidence as evidence


def complete(**changes):
    return dict(league="MLB", market_type="total_over", ml_target="total_over", ml_probability=.6,
                theover_probability=.55, market_probability=.52, kalshi_probability=.58,
                expected_value=-.05, edge=-.02, best_pick="Over 8.5", Play_Stake=0,
                **changes)


def test_complete_missing_stale_and_structured_only():
    row = complete()
    assert assess(row)["total_input_status"] == "COMPLETE"
    row.update(theover_probability=None, recent_regime_penalty_reason="stale_empirical_history")
    quality = assess(row)
    assert quality["total_input_status"] == "DEGRADED"
    assert quality["total_input_reason_codes"] == "missing_theover|stale_empirical_evidence"
    assert quality["total_input_signal_count"] == 3
    row["gemini_explanation"] = "Missing pitchers and weather. Approved premium wager."
    assert assess(row) == quality
    row["ml_target"] = "moneyline_home"
    assert assess(row)["total_input_status"] == "INCOMPLETE"
    assert "missing_target_model" in assess(row)["total_input_reason_codes"]
    assert assess({**row,"league":"NFL"}) == {}
    assert assess({**row,"market_type":"spread_home"}) == {}


def test_annotation_preserves_all_selection_and_authority_fields():
    original = pd.DataFrame([complete(), {**complete(),"theover_probability":None}])
    before = original.copy(deep=True)
    result = attach(original)
    pd.testing.assert_frame_equal(original,before)
    pd.testing.assert_frame_equal(result[original.columns],original)
    assert result.Play_Stake.tolist() == [0,0]
    assert counters(result)["mlb_total_candidate_count"] == 2


def test_price_value_is_not_probability_or_authority():
    result = display(.63,-180,-.02)
    assert result["model_win_probability"] == .63
    assert result["break_even_probability"] == pytest.approx(180/280)
    assert result["value_status"] == "NEGATIVE ESTIMATED VALUE"
    assert "production_eligible" not in result
    assert display(.63,-110,.05)["model_win_probability"] == .63
    assert display(.63,None,.05)["value_status"] == "VALUE UNAVAILABLE"


def test_prospective_label_is_immutable_after_grading(frozen):
    import sqlite3
    context, database, _ = frozen
    audit, final = fixture_frames()
    audit = attach(audit)
    saved, card = evidence.capture_run(context,audit,final,audit,path=database)
    with sqlite3.connect(database) as db:
        before = db.execute("SELECT candidates,payload_hash FROM snapshots").fetchone()
    scores = card[["snapshot_id","matchup_id"]].copy()
    scores["actual_home_score"],scores["actual_away_score"] = 6,4
    evidence.record_scores(scores,path=database)
    # Report reads existing file in mode=ro and never calls annotation again.
    rows, closes = read_candidates(database)
    assert len(rows) == 2 and all(r["total_input_status"] == "INCOMPLETE" for r in rows)
    result = summarize(rows,closes)
    cell = next(c for c in result["cohorts"] if c["status"] == "INCOMPLETE" and c["direction"] == "all")
    assert (cell["wins"],cell["losses"],cell["sample_size"]) == (1,1,2)
    assert cell["line_clv"] is None
    with sqlite3.connect(database) as db:
        assert before == db.execute("SELECT candidates,payload_hash FROM snapshots").fetchone()


def test_cohort_metrics_do_not_backfill_or_promote():
    row = dict(total_input_version=VERSION,total_input_status="DEGRADED",market_type="total_over",
               prediction_generated_at="2026-09-17T15:00:00Z",game_start_utc="2026-09-17T20:00:00Z",
               calibrated_probability=.6,odds_american=-110,candidate_outcome="WIN",snapshot_id="s",candidate_id="c")
    rows = [row,{**row,"candidate_outcome":"LOSS"},{**row,"candidate_outcome":"PUSH"},{**row,"total_input_version":""}]
    before = deepcopy(rows)
    cell = next(c for c in summarize(rows)["cohorts"] if c["status"] == "DEGRADED" and c["direction"] == "all")
    assert cell["sample_size"] == 3 and cell["win_rate"] == .5
    assert cell["brier"] == pytest.approx(.26)
    assert cell["roi"] == pytest.approx((100/110-1)/3)
    assert rows == before


def test_quality_survives_per_game_public_export_without_suppressing_picks():
    from test_per_game_boards import final, candidate
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import pick_record
    frame = pd.DataFrame([final(Bettable=False,Play_Stake=0,export_run_id='20260917T150000Z')])
    audit = attach(pd.DataFrame([candidate(export_run_id='20260917T150000Z')]))
    board = per_game_board(frame,audit,'totals')
    assert len(board) == 1 and board.iloc[0]['pick'] == 'Under 8.5'
    assert board.iloc[0].total_input_status == 'INCOMPLETE'
    row = board.iloc[0].to_dict();row['start']='2026-09-17T20:00:00Z'
    public = pick_record(row)
    assert public['total_input_status'] == 'INCOMPLETE'
    assert public['status'] == 'PASS'
    assert public['win_estimate'] == board.iloc[0].win_probability


def test_missing_or_proxy_closes_are_excluded():
    from app_core.total_quality_report import valid_close
    row = dict(game_start_utc='2026-09-17T20:00:00Z')
    assert not valid_close(row,dict(quote_verified=True,line_clv=1,price_clv=.1))
