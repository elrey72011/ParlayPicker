"""What each win estimate was built from: the price alone, or the price and something else (owner 2026-11-17).

Over nine graded nights (7 to 15 November 2026) the estimate was the sportsbook's own price and nothing
else for 451 of 525 games, and those picks went 233-217. ``app_core.rating_basis`` names the inputs the
blend recorded for each candidate, so the board and the exports can say which picks are the price alone.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from app_core import rating_basis as rb
from app_core import theover_model_totals as tmt


def _rows(*rows):
    return pd.DataFrame([dict(blend_in_market=market, blend_in_kalshi=kalshi, blend_in_ml=model,
                              blend_in_theover=theover, blend_in_theover_source=source)
                         for market, kalshi, model, theover, source in rows])


def test_a_rating_is_named_by_what_the_blend_recorded():
    frame = _rows((0.52, None, None, None, ""),                 # the book's price and nothing else
                  (0.50, 0.60, None, None, ""),                 # Kalshi at 55% or more is in the blend
                  (0.50, 0.54, None, None, ""),                 # below 55% the blend leaves Kalshi out
                  (0.48, None, 0.55, None, ""),                 # the model gave a probability
                  (0.50, None, None, 0.70, tmt.SOURCE),         # TheOver's model-backed totals pick
                  (0.50, None, None, 0.62, tmt.UPLOAD),         # a probability from the upload
                  (0.50, None, None, 0.50, tmt.UPLOAD),         # TheOver's "no prediction" value
                  (0.51, 0.70, 0.58, 0.70, tmt.SOURCE))
    assert list(rb.basis(frame)) == [
        "Price only", "Price + Kalshi", "Price only", "Price + model", "Price + TheOver model pick",
        "Price + TheOver probability", "Price only", "Price + Kalshi, model, TheOver model pick"]
    assert list(rb.price_only(rb.basis(frame))) == [True, False, True, False, False, False, True, False]
    assert rb.counts(rb.basis(frame)) == {"picks": 8, "price_only": 3}
    assert list(rb.inputs(frame))[-1] == (rb.KALSHI, rb.MODEL, rb.THEOVER_MODEL_PICK)


def test_a_row_with_no_price_says_so_and_unrecorded_inputs_claim_nothing():
    frame = _rows((None, None, None, None, ""), (None, 0.60, None, None, ""))
    assert list(rb.basis(frame)) == ["No price", "No price: Kalshi"]
    assert rb.counts(rb.basis(frame)) == {"picks": 2, "price_only": 0}
    # An older saved analysis never recorded the blend's inputs: nothing is said about it.
    bare = pd.DataFrame({"league": ["NCAAB", "NBA"], "calibrated_probability": [0.52, 0.5]})
    assert list(rb.basis(bare)) == ["", ""] and rb.counts(rb.basis(bare)) == {"picks": 0, "price_only": 0}
    assert rb.basis(frame.iloc[0:0]).empty and rb.basis(None).empty
    repeated = _rows((0.5, None, None, None, ""), (0.5, 0.6, None, None, ""))
    repeated.index = [7, 7]                                                  # a candidate frame's index can repeat
    assert list(rb.basis(repeated)) == ["Price only", "Price + Kalshi"]


def test_sentiment_counts_only_when_it_moves_more_than_the_blends_dead_band():
    frame = _rows((0.5, None, None, None, ""), (0.5, None, None, None, ""), (0.5, None, None, None, ""))
    frame["sentiment_diff"] = [0.0, 0.03, 0.10]            # the blend reads 0.5 + diff / 2: 0.500, 0.515, 0.550
    assert list(rb.basis(frame)) == ["Price only", "Price only", "Price + sentiment"]


def test_price_only_is_exactly_when_the_blend_returns_the_price():
    """The label follows the blend's own rules: checked against the blend itself, not a copy of its weights."""
    from core.streamlit_pipeline import compute_blended_probability
    rows = [dict(league=league, market_type=kind, blend_in_market=0.48, blend_in_kalshi=kalshi, blend_in_ml=model,
                 blend_in_theover=theover, blend_in_theover_source=source, sentiment_diff=diff)
            for league in ("NCAAB", "NBA", "NHL", "MLB", "NFL", "NCAAF", "WNBA")
            for kind in ("spread_home", "total_over")
            for kalshi in (None, 0.40, 0.5499, 0.55, 0.62)
            for model in (None, 0.57)
            for theover, source in ((None, ""), (0.5, tmt.UPLOAD), (0.64, tmt.UPLOAD), (0.70, tmt.SOURCE))
            for diff in (0.0, 0.2)]
    frame = pd.DataFrame(rows)
    blended = compute_blended_probability(
        p_market=frame["blend_in_market"], p_kalshi=frame["blend_in_kalshi"], p_ml=frame["blend_in_ml"],
        p_theover=frame["blend_in_theover"], p_sentiment=0.5 + frame["sentiment_diff"] * 0.5,
        league=frame["league"], market_type=frame["market_type"], theover_source=frame["blend_in_theover_source"])
    is_the_price = (blended - 0.48).abs() < 1e-12
    labelled = rb.price_only(rb.basis(frame))
    assert len(frame) == 1120 and is_the_price.sum() == 7 * 2 * 3 * 2          # no Kalshi, model, TheOver or sentiment
    assert list(labelled) == list(is_the_price)


# ---- in the pipeline: on the candidate audit and the best picks ----

@pytest.fixture
def _pregame(monkeypatch):
    monkeypatch.setattr("app_core.candidate_chronology.now_utc", lambda: pd.Timestamp("2026-11-16T21:00:00Z"))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)


def _game():
    """A college game as the pipeline hands it to selection: TheOver's model picked the Under; the sides are the price."""
    kinds = (("total_under", 0.495, 1.0), ("total_over", 0.505, 0.0), ("spread_home", 0.52, None), ("spread_away", 0.48, None))
    frame = pd.DataFrame([{"matchup_id": "g", "league": "NCAAB", "market_type": kind, "market_probability": market,
                           "theover_pick_agrees": picked, "theover_model_hit_rate": 0.73 if picked is not None else None}
                          for kind, market, picked in kinds])
    from core.streamlit_pipeline import compute_blended_probability
    empty = pd.Series(float("nan"), index=frame.index)
    theover, source = tmt.blend_input(frame)
    blended = compute_blended_probability(p_market=frame["market_probability"], p_kalshi=empty, p_ml=empty, p_theover=theover,
                                          p_sentiment=pd.Series(0.5, index=frame.index), league=frame["league"],
                                          market_type=frame["market_type"], theover_source=source)
    quotes = [{"book": "novig", "market_type": kind, "point": point, "price": -110, "recorded_at": "2026-11-16T20:55:00Z"}
              for kind, point in (("total_over", 145.5), ("total_under", 145.5), ("spread_home", -3.5), ("spread_away", 3.5))]
    rows = []
    for (kind, market, picked), probability, the, where in zip(kinds, blended, theover, source):
        total = kind.startswith("total")
        rows.append({
            "game_id": "rating-basis", "league": "NCAAB", "home_team": "Rice", "away_team": "Tulsa",
            "game_date": pd.Timestamp("2026-11-16", tz="UTC"), "game_start_utc": "2026-11-16T23:00:00Z",
            "odds_recorded_at": "2026-11-16T20:55:00Z", "market_type": kind, "total_line": 145.5 if total else pd.NA,
            "spread_line": pd.NA if total else (-3.5 if kind == "spread_home" else 3.5),
            "calibrated_probability": float(probability), "model_probability": float(probability), "ml_probability": pd.NA,
            "expected_value": round(float(probability) * (1 + 100 / 110) - 1, 4), "edge": round(float(probability) - market, 4),
            "market_probability": market, "kalshi_probability": pd.NA, "consensus_agreement": "No Kalshi",
            "odds_american": -110, "opposing_odds_american": -110, "opposing_odds_source": "novig", "odds_source": "odds_api",
            "line_source": "live", "market_line_source": "live", "live_total_line": 145.5, "is_live_data": True,
            "used_stale_features": False, "provider_quotes": json.dumps(quotes),
            "theover_pick_agrees": picked, "theover_model_hit_rate": 0.73 if picked is not None else None,
            # What the blend recorded, as run_analysis_pipeline stamps it.
            "blend_in_market": market, "blend_in_kalshi": float("nan"), "blend_in_ml": float("nan"),
            "blend_in_theover": the, tmt.SOURCE_COLUMN: where})
    return pd.DataFrame(rows)


def _run(frame):
    from core.streamlit_pipeline import build_best_picks_df
    diagnostics: dict = {}
    best = build_best_picks_df(frame, diagnostics_out=diagnostics)
    return best, diagnostics


def test_the_candidate_audit_and_the_pick_carry_what_the_estimate_was_built_from(_pregame):
    best, diagnostics = _run(_game())
    audit = diagnostics["candidate_audit_df"].set_index("market_type")
    assert audit.loc["total_under", "rating_basis"] == "Price + TheOver model pick"
    assert audit.loc["total_over", "rating_basis"] == "Price + TheOver model pick"     # the other side of the same pick
    assert audit.loc["spread_home", "rating_basis"] == audit.loc["spread_away", "rating_basis"] == "Price only"
    assert list(best["market_type"]) == ["total_under"] and list(best["rating_basis"]) == ["Price + TheOver model pick"]
    assert diagnostics["rating_basis"] == {"status": "OK", "picks": 1, "price_only": 0}


def test_marking_the_basis_changes_no_pick_estimate_or_wager_field(_pregame, monkeypatch):
    frame = _game()
    best, diagnostics = _run(frame)
    monkeypatch.setattr(rb, "basis", lambda pool: (_ for _ in ()).throw(RuntimeError("unreadable")))
    plain, quiet = _run(frame)                                             # the marker fails: Refresh still completes
    assert quiet["rating_basis"] == {"status": "ERROR", "error": "RuntimeError"}
    same = [c for c in best.columns if c != "rating_basis"]
    pd.testing.assert_frame_equal(best[same].reset_index(drop=True), plain[same].reset_index(drop=True))
    audit, bare = diagnostics["candidate_audit_df"], quiet["candidate_audit_df"]
    assert "rating_basis" not in bare.columns
    pd.testing.assert_frame_equal(audit.drop(columns=["rating_basis"]).reset_index(drop=True), bare.reset_index(drop=True))
    assert not audit["wager_approved"].any() and (best["production_bet_amount"] == 0).all()


def test_a_price_only_game_is_counted_as_one(_pregame):
    frame = _game()
    frame["theover_pick_agrees"] = None
    frame["theover_model_hit_rate"] = None
    frame["blend_in_theover"] = float("nan")
    frame[tmt.SOURCE_COLUMN] = ""
    frame["calibrated_probability"] = frame["market_probability"]
    frame["model_probability"] = frame["market_probability"]
    best, diagnostics = _run(frame)
    assert list(best["rating_basis"]) == ["Price only"]
    assert diagnostics["rating_basis"] == {"status": "OK", "picks": 1, "price_only": 1}


def test_the_basis_reaches_every_export_and_stays_out_of_the_hashed_predictions():
    from core.streamlit_pipeline import BEST_PICK_COLUMNS, REQUIRED_BEST_PICK_EXPORT_COLUMNS
    assert rb.COLUMN == "rating_basis"
    assert rb.COLUMN in BEST_PICK_COLUMNS and rb.COLUMN in REQUIRED_BEST_PICK_EXPORT_COLUMNS
    import streamlit_app as app
    assert app._COMPACT_EXPORT_COLUMNS[-1] == rb.COLUMN                     # last, so no earlier workbook column moves
    compact = app._build_compact_export_frame(pd.DataFrame([{
        "league": "NCAAB", "Home": "Rice", "Away": "Tulsa", "best_pick": "Under 145.5", "odds_american": -110,
        "odds_source": "odds_api", "Play_Stake": 0.0, "Kelly_Bet_Size": 0.0, "rating_basis": "Price only"}]))
    assert list(compact["rating_basis"]) == ["Price only"]
    from app_core import pinnacle_model as pm
    assert not {"basis", "inputs", "price_only", "counts"} & set(pm.PREDICTION_FUNCTIONS)


# ---- on the homepage board ----

def _board_and_candidates():
    result = pd.DataFrame([
        {"matchup_id": "g1", "market_type": "total_under", "pick": "Under 145.5", "odds": -110},
        {"matchup_id": "g2", "market_type": "spread_away", "pick": "Yale +2.5", "odds": -108},
        {"matchup_id": "g3", "market_type": "spread_home", "pick": "Duke -7.5", "odds": -110}])
    candidates = pd.DataFrame([
        {"matchup_id": "g1", "market_type": "total_under", "best_pick": "Under 145.5", "rating_basis": "Price + TheOver model pick"},
        {"matchup_id": "g1", "market_type": "total_over", "best_pick": "Over 145.5", "rating_basis": "Price + TheOver model pick"},
        {"matchup_id": "g2", "market_type": "spread_away", "best_pick": "Yale +2.5", "rating_basis": "Price only"}])
    return result, candidates


def test_the_board_reads_each_picks_basis_from_its_saved_candidate():
    from app.ui.daily_dashboard import rating_basis_labels
    result, candidates = _board_and_candidates()
    assert list(rating_basis_labels(result, candidates)) == ["Price + TheOver model pick", "Price only", ""]
    assert rating_basis_labels(result, candidates.drop(columns=["rating_basis"])) is None       # an older saved analysis
    assert rating_basis_labels(result, None) is None
    assert rating_basis_labels(result, candidates.assign(rating_basis="")) is None


def _board_app():
    import pandas as pd
    import streamlit as st
    from app.ui.daily_dashboard import render_daily_dashboard
    today, details = st.tabs(["Today", "Details"])
    base = {"league": "NCAAB", "production_edge": 0.0, "Play_Stake": 0, "production_eligible": False, "odds_american": -110}
    board = pd.DataFrame([
        dict(base, Home="Rice", Away="Tulsa", best_pick="Under 145.5", market_type="total_under", matchup_id="g1",
             production_win_probability=0.57, production_expected_value=0.088),
        dict(base, Home="Yale", Away="Brown", best_pick="Brown +2.5", market_type="spread_away", matchup_id="g2",
             production_win_probability=0.51, production_expected_value=-0.026)])
    candidates = pd.DataFrame([
        {"matchup_id": "g1", "market_type": "total_under", "best_pick": "Under 145.5", "rating_basis": "Price + TheOver model pick"},
        {"matchup_id": "g2", "market_type": "spread_away", "best_pick": "Brown +2.5", "rating_basis": "Price only"}])
    render_daily_dashboard(today.empty(), details.empty(), board, candidates)


def test_the_homepage_board_shows_the_basis_beside_the_win_estimate_and_counts_the_price_only_picks():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(_board_app).run()
    assert not app.exception
    frame = next(d.value for d in app.dataframe if "Wager status" in d.value.columns)
    assert list(frame.columns) == ["Game", "Best pick", "Odds", "Win estimate", "Built from", "EV estimate", "Wager status"]
    assert list(frame["Built from"]) == ["Price + TheOver model pick", "Price only"]
    assert list(frame["Wager status"]) == ["PASS", "PASS"]                              # naming an input approves nothing
    assert any(c.value.startswith("1 of 2 picks here are built from the price alone.") and rb.CAPTION in c.value
               for c in app.caption)


def test_the_full_pick_board_names_the_column_too():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    assert '"rating_basis": "Built From"' in text and '"Prob", "Built From", "ML Prob"' in text
