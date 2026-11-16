"""TheOver's model-backed totals pick as one of the blend's inputs (owner 2026-11-16).

The owner approved changing the prediction on 16 November 2026 and asked that TheOver be used
together with the rest of the data. That reverses, for this one signal, the 7 November rule that
TheOver's signals are kept per row and never used (roadmap D26). The uploaded
``theover_probability`` stays empty; the blend's TheOver input is filled from the pick.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from app_core import theover_model_totals as tmt
from app_core.weights_config import (FALLBACK_MARKET_WEIGHT, FALLBACK_THEOVER_WEIGHT, KALSHI_WEIGHT, MARKET_WEIGHT,
                                     THEOVER_WEIGHT)


def _rows(*rows, league="NCAAB"):
    return pd.DataFrame([{"matchup_id": game, "league": league, "market_type": kind, "market_probability": market,
                          "theover_pick_agrees": picked, "theover_model_hit_rate": hit}
                         for game, kind, market, picked, hit in rows])


def test_backed_is_a_total_theover_picked_with_a_hit_rate_listed():
    frame = _rows(("g", "total_under", 0.495, 1.0, 0.73), ("g", "total_over", 0.505, 0.0, 0.73),
                  ("g", "spread_home", 0.52, 1.0, 0.73),            # a side, not a total
                  ("h", "total_over", 0.50, 1.0, None),             # no hit rate listed
                  ("i", "total_over", 0.50, None, 0.80),            # no pick
                  ("j", "total_over", 0.50, 1.0, 1.0), ("k", "total_over", 0.50, 1.0, 0.0))   # not a rate
    assert list(tmt.backed(frame)) == [True, False, False, False, False, False, False]
    assert not tmt.backed(frame.drop(columns=["theover_model_hit_rate"])).any()


def test_the_picked_side_gets_the_listed_rate_capped_and_the_other_side_the_rest():
    frame = _rows(("g", "total_under", 0.495, 1.0, 0.73), ("g", "total_over", 0.505, 0.0, 0.73),
                  ("g", "spread_home", 0.52, None, None), ("g", "spread_away", 0.48, None, None),
                  ("h", "total_over", 0.50, 1.0, 0.62), ("h", "total_under", 0.50, 0.0, 0.62),
                  ("i", "total_over", 0.51, 0.0, None), ("i", "total_under", 0.49, 1.0, None))
    frame.index = [9, 4, 7, 1, 3, 8, 2, 6]
    values = tmt.derived(frame)
    assert list(values.index) == [9, 4, 7, 1, 3, 8, 2, 6]
    assert values.loc[9] == pytest.approx(0.70) and values.loc[4] == pytest.approx(0.30)     # 73% listed counts as 70%
    assert values.loc[3] == pytest.approx(0.62) and values.loc[8] == pytest.approx(0.38)     # under the cap: as listed
    assert values.loc[[7, 1, 2, 6]].isna().all()                                             # sides, and a pick with no rate
    assert tmt.CAP == 0.70


def test_games_with_nothing_usable_are_left_empty():
    both = _rows(("g", "total_over", 0.5, 1.0, 0.8), ("g", "total_under", 0.5, 1.0, 0.8))      # the upload names both sides
    assert tmt.derived(both).isna().all()
    alone = _rows(("g", "total_over", 0.5, 1.0, 0.8))                                          # the other side is not a candidate
    assert list(tmt.derived(alone)) == pytest.approx([0.70])
    two = _rows(("g", "total_under", 0.495, 1.0, 0.73), ("g", "total_over", 0.505, 0.0, 0.73))
    assert tmt.derived(two, cap=0).isna().all()                                                # the switch-off
    assert tmt.derived(two.drop(columns=["matchup_id"])).isna().all()
    assert tmt.derived(two.iloc[0:0]).empty
    assert tmt.derived(two.drop(columns=["theover_pick_agrees"])).isna().all()


def test_a_probability_from_the_upload_is_kept_and_the_pick_fills_the_rest():
    frame = _rows(("g", "total_under", 0.495, 1.0, 0.73), ("g", "total_over", 0.505, 0.0, 0.73),
                  ("h", "total_over", 0.50, 1.0, 0.80), ("h", "total_under", 0.50, 0.0, 0.80),
                  ("h", "spread_home", 0.52, None, None))
    uploaded = pd.Series([None, None, 0.61, 0.39, None], index=frame.index, dtype="float64")
    values, source = tmt.blend_input(frame, uploaded)
    assert list(values.iloc[:4]) == pytest.approx([0.70, 0.30, 0.61, 0.39]) and pd.isna(values.iloc[4])
    assert list(source) == [tmt.SOURCE, tmt.SOURCE, tmt.UPLOAD, tmt.UPLOAD, ""]
    plain, where = tmt.blend_input(frame)                    # nothing from the upload at all
    assert list(plain.iloc[:4]) == pytest.approx([0.70, 0.30, 0.70, 0.30]) and list(where.iloc[:4]) == [tmt.SOURCE] * 4


# ---- in the blend, with the rest of the data ----

def _blend(frame, kalshi=None):
    from core.streamlit_pipeline import compute_blended_probability
    empty = pd.Series(float("nan"), index=frame.index)
    theover, _ = tmt.blend_input(frame)
    return compute_blended_probability(p_market=frame["market_probability"], p_kalshi=empty if kalshi is None else kalshi,
                                       p_ml=empty, p_theover=theover, p_sentiment=pd.Series(0.5, index=frame.index),
                                       league=frame["league"], market_type=frame["market_type"])


def test_the_blend_weighs_theover_with_the_market():
    frame = _rows(("g", "total_under", 0.495, 1.0, 0.73), ("g", "total_over", 0.505, 0.0, 0.73),
                  ("g", "spread_home", 0.52, None, None))
    blended = _blend(frame)
    weight = FALLBACK_MARKET_WEIGHT + FALLBACK_THEOVER_WEIGHT
    assert blended.iloc[0] == pytest.approx((FALLBACK_MARKET_WEIGHT * 0.495 + FALLBACK_THEOVER_WEIGHT * 0.70) / weight)
    assert blended.iloc[1] == pytest.approx((FALLBACK_MARKET_WEIGHT * 0.505 + FALLBACK_THEOVER_WEIGHT * 0.30) / weight)
    assert blended.iloc[0] + blended.iloc[1] == pytest.approx(1.0)          # the two sides still sum to one
    assert 0.55 < blended.iloc[0] < 0.60                                    # moved toward TheOver, not handed to it
    assert blended.iloc[2] == pytest.approx(0.52)                           # a row TheOver says nothing about is unchanged


def test_a_kalshi_price_that_disagrees_pulls_the_estimate_back():
    frame = _rows(("g", "total_under", 0.495, 1.0, 0.73), ("g", "total_over", 0.505, 0.0, 0.73))
    alone = _blend(frame)
    kalshi = pd.Series([0.40, 0.60], index=frame.index)                     # Kalshi has the Over at 60%
    both = _blend(frame, kalshi)
    weight = KALSHI_WEIGHT + MARKET_WEIGHT + THEOVER_WEIGHT
    assert both.iloc[1] == pytest.approx((KALSHI_WEIGHT * 0.60 + MARKET_WEIGHT * 0.505 + THEOVER_WEIGHT * 0.30) / weight)
    assert both.iloc[1] > alone.iloc[1]                                     # the Over is rated higher than TheOver alone had it
    assert both.iloc[0] / (both.iloc[0] + both.iloc[1]) < alone.iloc[0]     # so TheOver's side ends lower once the pair is rescaled


def test_the_re_blend_after_kalshi_uses_the_same_input():
    import streamlit_app as app
    frame = pd.DataFrame([dict(matchup_id="g", league="NCAAB", home_team="Rice", away_team="Tulsa", market_type=kind,
                               market_probability=market, odds_american=-110, opposing_odds_american=-110,
                               total_line=145.5 if kind.startswith("total") else None,
                               spread_line=None if kind.startswith("total") else -3.5, theover_pick_agrees=picked,
                               theover_model_hit_rate=hit, theover_probability=None, ml_probability=None,
                               kalshi_probability=None, win_prob_source="not_provided")
                          for kind, market, picked, hit in (("total_under", 0.495, 1.0, 0.73), ("total_over", 0.505, 0.0, 0.73),
                                                            ("spread_home", 0.52, None, None))])
    out = app._recompute_consensus_from_kalshi(frame).set_index("market_type")
    weight = FALLBACK_MARKET_WEIGHT + FALLBACK_THEOVER_WEIGHT
    assert out.loc["total_under", "calibrated_probability"] == pytest.approx(
        (FALLBACK_MARKET_WEIGHT * 0.495 + FALLBACK_THEOVER_WEIGHT * 0.70) / weight)
    assert out.loc["total_under", "blend_in_theover"] == pytest.approx(0.70)
    assert out.loc["total_over", "blend_in_theover"] == pytest.approx(0.30)
    assert out.loc["total_under", "blend_in_theover_source"] == tmt.SOURCE
    assert out.loc["spread_home", "calibrated_probability"] == pytest.approx(0.52)
    assert pd.isna(out.loc["spread_home", "blend_in_theover"]) and out.loc["spread_home", "blend_in_theover_source"] == ""


def test_the_analysis_blend_uses_it_and_the_uploaded_column_stays_empty():
    from core.streamlit_pipeline import _apply_analysis_calculations
    frame = pd.DataFrame([dict(matchup_id="g", league="NCAAB", home_team="Rice", away_team="Tulsa", market_type=kind,
                               market_probability=0.5, odds_american=-110, opposing_odds_american=-110, total_line=145.5,
                               spread_line=None, theover_pick_agrees=picked, theover_model_hit_rate=0.73,
                               theover_probability=None, ml_probability=None, kalshi_probability=None,
                               win_prob_source="not_provided")
                          for kind, picked in (("total_under", 1.0), ("total_over", 0.0))])
    out = _apply_analysis_calculations(frame).set_index("market_type")
    assert out.loc["total_under", "calibrated_probability"] > 0.55 > 0.45 > out.loc["total_over", "calibrated_probability"]
    assert out["theover_probability"].isna().all()                          # nothing is written back as an upload


def test_all_three_blends_take_their_theover_input_from_the_same_place():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    pipeline = (root / "core" / "streamlit_pipeline.py").read_text(encoding="utf-8")
    app = (root / "streamlit_app.py").read_text(encoding="utf-8")
    assert pipeline.count("_theover_model_totals.blend_input(") == 1        # the production blend
    assert pipeline.count("_theover_model_blend_input(out, theover)") == 1  # the Analysis-tab blend
    assert app.count("_theover_model_totals.blend_input(out, theover_prob_blend)") == 1   # the re-blend after Kalshi


# ---- the pick, and what cannot follow from it ----

@pytest.fixture
def _pregame(monkeypatch):
    monkeypatch.setattr("app_core.candidate_chronology.now_utc", lambda: pd.Timestamp("2026-11-16T21:00:00Z"))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)


def _game(with_theover):
    """A college game as the pipeline hands it to selection, blended with or without TheOver's pick."""
    kinds = (("total_under", 0.495, 1.0), ("total_over", 0.505, 0.0), ("spread_home", 0.52, None), ("spread_away", 0.48, None))
    frame = _rows(*[("g", kind, market, picked, 0.73 if with_theover and picked is not None else None)
                    for kind, market, picked in kinds])
    blended = _blend(frame)
    quotes = [{"book": "novig", "market_type": kind, "point": point, "price": -110, "recorded_at": "2026-11-16T20:55:00Z"}
              for kind, point in (("total_over", 145.5), ("total_under", 145.5), ("spread_home", -3.5), ("spread_away", 3.5))]
    rows = []
    for (kind, market, picked), probability in zip(kinds, blended):
        total = kind.startswith("total")
        rows.append({
            "game_id": "theover-blend", "league": "NCAAB", "home_team": "Rice", "away_team": "Tulsa",
            "game_date": pd.Timestamp("2026-11-16", tz="UTC"), "game_start_utc": "2026-11-16T23:00:00Z",
            "odds_recorded_at": "2026-11-16T20:55:00Z", "market_type": kind, "total_line": 145.5 if total else pd.NA,
            "spread_line": pd.NA if total else (-3.5 if kind == "spread_home" else 3.5),
            "calibrated_probability": float(probability), "model_probability": float(probability), "ml_probability": pd.NA,
            "expected_value": round(float(probability) * (1 + 100 / 110) - 1, 4), "edge": round(float(probability) - market, 4),
            "market_probability": market, "kalshi_probability": pd.NA, "consensus_agreement": "No Kalshi",
            "odds_american": -110, "opposing_odds_american": -110, "opposing_odds_source": "novig", "odds_source": "odds_api",
            "line_source": "live", "market_line_source": "live", "live_total_line": 145.5, "is_live_data": True,
            "used_stale_features": False, "provider_quotes": json.dumps(quotes),
            "theover_pick_agrees": picked, "theover_model_hit_rate": 0.73 if with_theover and picked is not None else None})
    return pd.DataFrame(rows)


def _run(frame):
    from core.streamlit_pipeline import build_best_picks_df
    diagnostics: dict = {}
    best = build_best_picks_df(frame, diagnostics_out=diagnostics)
    return best, diagnostics["candidate_audit_df"].set_index("market_type")


def test_theovers_side_of_the_total_becomes_the_games_pick(_pregame):
    before, _ = _run(_game(False))
    after, audit = _run(_game(True))
    assert list(before["market_type"]) == ["spread_home"] and list(after["market_type"]) == ["total_under"]
    weight = FALLBACK_MARKET_WEIGHT + FALLBACK_THEOVER_WEIGHT
    estimate = (FALLBACK_MARKET_WEIGHT * 0.495 + FALLBACK_THEOVER_WEIGHT * 0.70) / weight
    assert audit.loc["total_under", "calibrated_probability"] == pytest.approx(estimate)
    assert audit.loc["total_under", "best_available_probability"] == pytest.approx(estimate)   # the pair already sums to one
    assert audit.loc["spread_home", "best_available_probability"] == pytest.approx(0.52)
    assert set(audit["best_available_selection_policy"]) == {"probability-first-v1"}


def test_a_qualified_lean_is_the_most_it_can_become_without_market_authority(_pregame):
    from app_core.lean_card import attach_play_stakes, score_best_picks_rows
    best, audit = _run(_game(True))
    pick = best.iloc[0]
    assert pick["expected_value"] > 0                               # the wager checks do see the blended estimate
    assert pick["Kelly_Bet_Size"] == 0 and pick["production_bet_amount"] == 0
    assert not bool(pick["production_eligible"]) and not bool(pick["wager_approved"])
    scored = attach_play_stakes(score_best_picks_rows(best), unit=1.0).iloc[0]
    assert scored["Play_Stake"] == 0 and str(scored["Bet_Decision"]).endswith("PASS")
    assert not audit["wager_approved"].any()


def test_the_source_reaches_the_audit_and_the_export_and_stays_out_of_the_hashed_predictions():
    from core.streamlit_pipeline import BEST_PICK_COLUMNS, REQUIRED_BEST_PICK_EXPORT_COLUMNS
    assert tmt.SOURCE_COLUMN in BEST_PICK_COLUMNS and tmt.SOURCE_COLUMN in REQUIRED_BEST_PICK_EXPORT_COLUMNS
    from app_core import pinnacle_model as pm
    assert not {"blend_input", "derived", "backed"} & set(pm.PREDICTION_FUNCTIONS)


# ---- NBA totals: the model pick takes the standard weight (owner 2026-11-16, after the first audit) ----

def _blend_as_the_pipeline_does(frame, kalshi=None, ml=None, uploaded=None):
    """The blend with the source named, as all three call sites pass it."""
    from core.streamlit_pipeline import compute_blended_probability
    empty = pd.Series(float("nan"), index=frame.index)
    theover, source = tmt.blend_input(frame, uploaded)
    return compute_blended_probability(p_market=frame["market_probability"], p_kalshi=empty if kalshi is None else kalshi,
                                       p_ml=empty if ml is None else ml, p_theover=theover,
                                       p_sentiment=pd.Series(0.5, index=frame.index), league=frame["league"],
                                       market_type=frame["market_type"], theover_source=source)


def test_an_nba_total_weighs_the_model_pick_like_any_other_league():
    from app_core.weights_config import NBA_TOTAL_FALLBACK_THEOVER_WEIGHT
    rows = (("g", "total_under", 0.488, 1.0, 0.875), ("g", "total_over", 0.512, 0.0, 0.875))
    nba = _blend_as_the_pipeline_does(_rows(*rows, league="NBA"))
    college = _blend_as_the_pipeline_does(_rows(*rows))
    weight = FALLBACK_MARKET_WEIGHT + FALLBACK_THEOVER_WEIGHT
    assert nba.iloc[0] == pytest.approx((FALLBACK_MARKET_WEIGHT * 0.488 + FALLBACK_THEOVER_WEIGHT * 0.70) / weight)
    assert list(nba) == pytest.approx(list(college))
    assert nba.iloc[0] + nba.iloc[1] == pytest.approx(1.0)
    # The first audit (16 November) had these NBA totals at 60%, above every college total, under the old weight.
    old = ((FALLBACK_MARKET_WEIGHT * 0.488 + NBA_TOTAL_FALLBACK_THEOVER_WEIGHT * 0.70)
           / (FALLBACK_MARKET_WEIGHT + NBA_TOTAL_FALLBACK_THEOVER_WEIGHT))
    assert old == pytest.approx(0.601, abs=0.001) and nba.iloc[0] == pytest.approx(0.565, abs=0.001)


def test_with_kalshi_and_a_model_the_nba_pick_still_takes_the_standard_weights():
    from app_core.weights_config import FALLBACK_ML_WEIGHT, ML_MODEL_WEIGHT
    frame = _rows(("g", "total_under", 0.488, 1.0, 0.875), ("g", "total_over", 0.512, 0.0, 0.875), league="NBA")
    kalshi = pd.Series([0.58, 0.42], index=frame.index)          # Kalshi has the Under at 58%: its weight counts on that row
    ml = pd.Series([0.53, 0.47], index=frame.index)
    blended = _blend_as_the_pipeline_does(frame, kalshi, ml)
    weight = KALSHI_WEIGHT + MARKET_WEIGHT + ML_MODEL_WEIGHT + THEOVER_WEIGHT
    assert blended.iloc[0] == pytest.approx(
        (KALSHI_WEIGHT * 0.58 + MARKET_WEIGHT * 0.488 + ML_MODEL_WEIGHT * 0.53 + THEOVER_WEIGHT * 0.70) / weight)
    fallback = FALLBACK_MARKET_WEIGHT + FALLBACK_ML_WEIGHT + FALLBACK_THEOVER_WEIGHT
    assert blended.iloc[1] == pytest.approx(
        (FALLBACK_MARKET_WEIGHT * 0.512 + FALLBACK_ML_WEIGHT * 0.47 + FALLBACK_THEOVER_WEIGHT * 0.30) / fallback)


def test_a_theover_probability_from_the_upload_keeps_the_nba_weight():
    from app_core.weights_config import NBA_TOTAL_FALLBACK_THEOVER_WEIGHT
    from core.streamlit_pipeline import compute_blended_probability
    frame = _rows(("g", "total_under", 0.488, None, None), ("g", "total_over", 0.512, None, None), league="NBA")
    uploaded = pd.Series([0.62, 0.38], index=frame.index)
    blended = _blend_as_the_pipeline_does(frame, uploaded=uploaded)
    weight = FALLBACK_MARKET_WEIGHT + NBA_TOTAL_FALLBACK_THEOVER_WEIGHT
    assert blended.iloc[0] == pytest.approx((FALLBACK_MARKET_WEIGHT * 0.488 + NBA_TOTAL_FALLBACK_THEOVER_WEIGHT * 0.62) / weight)
    # A caller that names no source gets what it always got, for the upload and for the pick alike.
    empty = pd.Series(float("nan"), index=frame.index)
    unnamed = compute_blended_probability(p_market=frame["market_probability"], p_kalshi=empty, p_ml=empty, p_theover=uploaded,
                                          p_sentiment=pd.Series(0.5, index=frame.index), league=frame["league"],
                                          market_type=frame["market_type"])
    assert list(unnamed) == pytest.approx(list(blended))


def test_other_leagues_and_nba_sides_are_untouched_by_the_source():
    from core.streamlit_pipeline import compute_blended_probability
    for league in ("NCAAB", "NHL", "NFL", "NCAAF", "MLB", "WNBA"):
        frame = _rows(("g", "total_under", 0.488, 1.0, 0.73), ("g", "total_over", 0.512, 0.0, 0.73), league=league)
        empty = pd.Series(float("nan"), index=frame.index)
        theover, source = tmt.blend_input(frame)
        kalshi = pd.Series([0.58, 0.42], index=frame.index)
        for k in (empty, kalshi):
            named = compute_blended_probability(frame["market_probability"], k, empty, theover, pd.Series(0.5, index=frame.index),
                                                league=frame["league"], market_type=frame["market_type"], theover_source=source)
            unnamed = compute_blended_probability(frame["market_probability"], k, empty, theover, pd.Series(0.5, index=frame.index),
                                                  league=frame["league"], market_type=frame["market_type"])
            assert list(named) == pytest.approx(list(unnamed)), league
    spread = _rows(("g", "spread_home", 0.52, None, None), ("g", "spread_away", 0.48, None, None), league="NBA")
    assert list(_blend_as_the_pipeline_does(spread)) == pytest.approx([0.52, 0.48])


def test_the_re_blend_and_the_analysis_blend_level_the_nba_too():
    import streamlit_app as app
    from core.streamlit_pipeline import _apply_analysis_calculations

    def game(league):
        return pd.DataFrame([dict(matchup_id="g", league=league, home_team="Boston", away_team="Orlando", market_type=kind,
                                  market_probability=0.5, odds_american=-110, opposing_odds_american=-110, total_line=212.5,
                                  spread_line=None, theover_pick_agrees=picked, theover_model_hit_rate=0.875,
                                  theover_probability=None, ml_probability=None, kalshi_probability=None,
                                  win_prob_source="not_provided")
                             for kind, picked in (("total_under", 1.0), ("total_over", 0.0))])

    estimate = (FALLBACK_MARKET_WEIGHT * 0.5 + FALLBACK_THEOVER_WEIGHT * 0.70) / (FALLBACK_MARKET_WEIGHT + FALLBACK_THEOVER_WEIGHT)
    again = app._recompute_consensus_from_kalshi(game("NBA")).set_index("market_type")
    assert again.loc["total_under", "calibrated_probability"] == pytest.approx(estimate)
    assert again.loc["total_under", "blend_in_theover_source"] == tmt.SOURCE
    nba = _apply_analysis_calculations(game("NBA")).set_index("market_type")
    college = _apply_analysis_calculations(game("NCAAB")).set_index("market_type")
    assert nba.loc["total_under", "calibrated_probability"] == pytest.approx(college.loc["total_under", "calibrated_probability"])
    assert nba.loc["total_under", "calibrated_probability"] == pytest.approx(estimate)


def test_all_three_blends_tell_the_blend_where_the_theover_input_came_from():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    pipeline = (root / "core" / "streamlit_pipeline.py").read_text(encoding="utf-8")
    app = (root / "streamlit_app.py").read_text(encoding="utf-8")
    assert pipeline.count("theover_source=merged[_theover_model_totals.SOURCE_COLUMN]") == 1     # the production blend
    assert pipeline.count("theover_source=_theover_for_blend_source") == 1                        # the Analysis-tab blend
    assert app.count("theover_source=out[_theover_model_totals.SOURCE_COLUMN]") == 1              # the re-blend after Kalshi
