"""Tests for the 11 Jun parlay-accuracy changes:

* isotonic (PAV) calibration fit/apply in core.probability_calibration
* Below Threshold legs eligible; Actionable no longer sort-prioritized
* MAX_PARLAY_LEGS=2 (no 3-leg combos)
* same (league, direction) totals capped at 1 leg per parlay
* calibration table applied to leg probabilities before floors/EV
* grade_slate excludes No Play / unresolved-line recap rows
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.probability_calibration import (
    apply_calibration,
    fit_isotonic_calibration,
    load_calibration,
    save_calibration,
)
from core.smart_parlay_engine import generate_smart_parlays


def _bets(**overrides) -> pd.DataFrame:
    base = {
        "best_pick": ["Over 8.5", "Under 7.5", "A -1.5"],
        "league": ["MLB", "NHL", "NBA"],
        "matchup_id": ["g1", "g2", "g3"],
        "edge": [0.07, 0.06, 0.05],
        "calibrated_probability": [0.66, 0.64, 0.62],
        "effective_win_probability": [0.64, 0.64, 0.62],
        "market_probability": [0.50, 0.50, 0.50],
        "decimal_odds": [2.0, 2.0, 2.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_isotonic_fit_is_monotone_and_maps_overconfidence_down():
    # Overconfident sample: predicted ~0.62 band realizes ~50%
    probs = [0.55] * 10 + [0.62] * 10 + [0.70] * 10
    outcomes = [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5 + [1] * 7 + [0] * 3
    knots = fit_isotonic_calibration(probs, outcomes)

    ys = [k[1] for k in knots]
    assert ys == sorted(ys)  # monotone

    mapped = apply_calibration(pd.Series([0.62]), knots).iloc[0]
    assert mapped < 0.62  # overconfidence pulled down
    assert 0.0 <= mapped <= 1.0


def test_apply_calibration_clamps_and_passes_through_without_table():
    knots = [[0.5, 0.45], [0.7, 0.60]]
    s = pd.Series([0.3, 0.6, 0.9])
    out = apply_calibration(s, knots)
    assert out.iloc[0] == 0.45  # below range clamps to first knot
    assert out.iloc[2] == 0.60  # above range clamps to last knot
    assert 0.45 < out.iloc[1] < 0.60  # interpolated

    untouched = apply_calibration(s, None)
    assert untouched.equals(s)


def test_calibration_save_load_roundtrip(tmp_path):
    knots = [[0.5, 0.45], [0.7, 0.60]]
    path = tmp_path / "cal.json"
    save_calibration(knots, path, meta={"n_graded": 2})
    assert load_calibration(path) == knots
    assert load_calibration(tmp_path / "missing.json") is None


def test_production_calibration_requires_chronological_promotion(tmp_path, monkeypatch):
    knots = [[0.5, 0.45], [0.7, 0.60]]
    path = tmp_path / "cal.json"
    monkeypatch.setattr(
        "core.probability_calibration.DEFAULT_CALIBRATION_PATH", path
    )

    save_calibration(knots, path, meta={"n_graded": 200})
    assert load_calibration() is None

    save_calibration(
        knots,
        path,
        meta={"validation": {"promotable": False}},
    )
    assert load_calibration() is None

    save_calibration(
        knots,
        path,
        meta={"validation": {"promotable": True}},
    )
    assert load_calibration() == knots


def test_below_threshold_legs_are_eligible():
    bets = _bets(Pick_Status=["Below Threshold", "Below Threshold", "Below Threshold"])
    parlays = generate_smart_parlays(bets)
    assert not parlays.empty


def test_no_three_leg_parlays_while_cap_is_two():
    bets = _bets()
    parlays = generate_smart_parlays(bets)
    assert not parlays.empty
    assert set(parlays["legs"].unique()) == {2}


def test_proven_losing_bucket_leg_is_excluded_from_parlays():
    # A leg the empirical overlay flagged as a proven loser must not become a parlay
    # leg even when its raw edge/prob clear the floors (the overlay tags it because
    # its own bucket has bled below break-even). Two clean legs + one flagged: the
    # flagged one must never appear.
    bets = _bets(
        best_pick=["Over 8.5", "Under 5.5", "Under 7.5"],
        league=["MLB", "NHL", "MLB"],
        matchup_id=["g1", "g2", "g3"],
        status_blocker_stage=["", "", "empirical_proven_losing_bucket"],
    )
    parlays = generate_smart_parlays(bets)
    assert not parlays.empty
    for legs in parlays["parlay_legs"]:
        assert "Under 7.5" not in legs


def test_same_league_same_direction_totals_cannot_pair():
    # Two MLB Overs in different games + one NHL Under: the MLB Over pair is the
    # correlated block that must not form; each Over may still pair with the Under.
    bets = _bets(
        best_pick=["Over 8.5", "Over 9.5", "Under 5.5"],
        league=["MLB", "MLB", "NHL"],
        effective_win_probability=[0.64, 0.64, 0.64],
    )
    parlays = generate_smart_parlays(bets)
    assert not parlays.empty
    for legs in parlays["parlay_legs"]:
        assert legs.lower().count("over") <= 1


def test_calibration_is_applied_to_leg_probabilities_and_ev():
    bets = _bets(
        best_pick=["Over 8.5", "Under 5.5"],
        league=["MLB", "NHL"],
        matchup_id=["g1", "g2"],
        edge=[0.07, 0.06],
        calibrated_probability=[0.66, 0.66],
        effective_win_probability=[0.65, 0.65],
        market_probability=[0.50, 0.50],
        decimal_odds=[2.0, 2.0],
    )
    raw = generate_smart_parlays(bets)
    assert np.isclose(raw.iloc[0]["combined_probability"], 0.65 * 0.65)

    # Identity-shaped table that maps everything to 0.60
    knots = [[0.40, 0.60], [0.90, 0.60]]
    calibrated = generate_smart_parlays(bets, calibration=knots)
    assert not calibrated.empty
    assert np.isclose(calibrated.iloc[0]["combined_probability"], 0.60 * 0.60)
    expected_ev = (0.36 * (4.0 - 1)) - (1 - 0.36)
    assert np.isclose(calibrated.iloc[0]["parlay_ev"], expected_ev)


def test_calibration_can_gate_out_legs_below_floor():
    # Mapping 0.65 predicted -> 0.50 realized drops legs under MIN_LEG_PROBABILITY
    bets = _bets(
        best_pick=["Over 8.5", "Under 5.5"],
        league=["MLB", "NHL"],
        matchup_id=["g1", "g2"],
        edge=[0.07, 0.06],
        calibrated_probability=[0.66, 0.66],
        effective_win_probability=[0.65, 0.65],
        market_probability=[0.50, 0.50],
        decimal_odds=[2.0, 2.0],
    )
    knots = [[0.40, 0.50], [0.90, 0.50]]
    parlays = generate_smart_parlays(bets, calibration=knots)
    assert parlays.empty


def test_sort_is_by_ev_not_actionable_anchor():
    # The Actionable-anchored combo has lower EV than the non-Actionable combo;
    # EV must win the sort now.
    bets = pd.DataFrame(
        {
            "best_pick": ["Over 8.5", "Under 5.5", "A -1.5", "B +2.5"],
            "league": ["MLB", "NHL", "NBA", "NBA"],
            "matchup_id": ["g1", "g2", "g3", "g4"],
            "edge": [0.07, 0.06, 0.05, 0.05],
            "calibrated_probability": [0.60, 0.60, 0.70, 0.70],
            "effective_win_probability": [0.60, 0.60, 0.70, 0.70],
            "market_probability": [0.50] * 4,
            "decimal_odds": [2.0] * 4,
            "Pick_Status": [
                "Actionable",
                "Actionable",
                "High Variance/Speculative",
                "High Variance/Speculative",
            ],
        }
    )
    parlays = generate_smart_parlays(bets)
    assert not parlays.empty
    top = parlays.iloc[0]
    assert not top["has_actionable_anchor"]
    assert parlays["parlay_ev"].is_monotonic_decreasing


def test_grade_slate_excludes_no_play_and_unresolved(tmp_path):
    from scripts.grade_slate import grade

    export = pd.DataFrame(
        {
            "league": ["MLB", "MLB", "NBA"],
            "Home": ["Tampa Bay", "Detroit", "New York"],
            "Away": ["Boston", "Minnesota", "San Antonio"],
            "best_pick": ["Under 7.5", "Over 9.5", "San Antonio line unresolved"],
            "WinProbability": [0.62, 0.61, 0.60],
            "effective_win_probability": [0.62, 0.61, 0.60],
            "effective_edge": [0.05, 0.05, 0.05],
            "consensus_agreement": ["Agrees", "Agrees", "Neutral"],
            "kalshi_probability": [0.6, 0.6, 0.6],
            "Kelly_Bet_Size": [10, 10, 10],
            "odds_american": [-110, -110, -110],
        }
    )
    recap = pd.DataFrame(
        {
            "Home": ["Tampa Bay", "Detroit", "New York"],
            "Away": ["Boston", "Minnesota", "San Antonio"],
            "Pick Taken": ["Under 7.5", "Over 9.5", "San Antonio line unresolved"],
            "Outcome": ["LOSS", "WIN", "LOSS"],
            "Status": ["Actionable", "Below Threshold", "No Play"],
        }
    )
    export_csv = tmp_path / "export.csv"
    recap_csv = tmp_path / "recap.csv"
    out_csv = tmp_path / "graded.csv"
    export.to_csv(export_csv, index=False)
    recap.to_csv(recap_csv, index=False)

    grade(export_csv, recap_csv, out_csv)
    graded = pd.read_csv(out_csv)
    assert len(graded) == 2
    assert not graded["best_pick"].str.contains("unresolved").any()


def test_floors_are_mapped_through_calibration():
    # MIN_LEG_PROBABILITY (0.58) is defined in RAW prob space. With a calibration
    # table that maps ~0.59 raw -> ~0.54, the leg must still pass because the floor
    # is transformed through the same table (0.58 -> ~0.53), not applied raw.
    bets = _bets(
        best_pick=["Under 5.5", "A -1.5"],
        league=["NHL", "NBA"],
        matchup_id=["g1", "g2"],
        edge=[0.07, 0.06],
        calibrated_probability=[0.59, 0.59],
        effective_win_probability=[0.59, 0.59],
        market_probability=[0.45, 0.45],
        decimal_odds=[2.2, 2.2],
    )
    knots = [[0.50, 0.45], [0.70, 0.65]]
    parlays = generate_smart_parlays(bets, calibration=knots)
    assert not parlays.empty
    # 0.59 -> 0.45 + (0.09/0.20)*0.20 = 0.54 per leg
    assert np.isclose(parlays.iloc[0]["min_leg_prob"], 0.54)


def test_agrees_same_direction_pair_allowed_and_flagged_correlated():
    # Two MLB Overs, both Kalshi-Agrees: allowed to pair under the Agrees
    # exception, but flagged is_high_correlation for stake downweighting.
    bets = _bets(
        best_pick=["Over 8.5", "Over 9.5", "Under 5.5"],
        league=["MLB", "MLB", "NHL"],
        consensus_agreement=["Agrees", "Agrees", "Agrees"],
        effective_win_probability=[0.64, 0.64, 0.64],
    )
    parlays = generate_smart_parlays(bets)
    over_pairs = parlays[
        parlays["parlay_legs"].str.lower().str.count("over").eq(2)
    ]
    assert not over_pairs.empty
    assert over_pairs["is_high_correlation"].all()

    # The same pair with Neutral consensus stays blocked.
    bets_neutral = bets.assign(consensus_agreement=["Neutral", "Neutral", "Neutral"])
    parlays_neutral = generate_smart_parlays(bets_neutral)
    if not parlays_neutral.empty:
        assert (
            parlays_neutral["parlay_legs"].str.lower().str.count("over").lt(2).all()
        )


def test_calibrated_ev_gate_supersedes_raw_mlb_floor():
    # A Below Threshold MLB total at 0.60 raw would fail the 0.62 raw-space floor,
    # but with calibration active the gate is calibrated prob vs the leg's own
    # break-even + margin: 0.55 calibrated vs 1/2.1 + 0.03 = 0.506 -> passes.
    bets = _bets(
        best_pick=["Over 8.5", "Under 5.5"],
        league=["MLB", "NHL"],
        matchup_id=["g1", "g2"],
        edge=[0.07, 0.06],
        calibrated_probability=[0.60, 0.60],
        effective_win_probability=[0.60, 0.60],
        market_probability=[0.48, 0.48],
        decimal_odds=[2.1, 2.1],
        Pick_Status=["Below Threshold", "Below Threshold"],
    )
    knots = [[0.40, 0.55], [0.90, 0.55]]
    parlays = generate_smart_parlays(bets, calibration=knots)
    assert not parlays.empty

    # And a leg below its break-even + margin is dropped: 0.50 calibrated vs
    # 1/2.0 + 0.03 = 0.53 -> no parlays.
    knots_low = [[0.40, 0.50], [0.90, 0.50]]
    bets_even = bets.assign(decimal_odds=[2.0, 2.0])
    assert generate_smart_parlays(bets_even, calibration=knots_low).empty


def test_downweight_correlated_parlay_kelly():
    from core.smart_parlay_engine import downweight_correlated_parlay_kelly

    parlays = pd.DataFrame(
        {
            "parlay_legs": ["a | b", "c | d"],
            "is_high_correlation": [True, False],
            "kelly_fraction": [0.04, 0.04],
            "recommended_bet": [10.0, 10.0],
        }
    )
    out = downweight_correlated_parlay_kelly(parlays)
    assert out.loc[0, "kelly_fraction"] == 0.02
    assert out.loc[0, "recommended_bet"] == 5.0
    assert out.loc[1, "kelly_fraction"] == 0.04
    assert out.loc[1, "recommended_bet"] == 10.0


def test_zero_zero_final_is_void_not_loss(tmp_path):
    # 0-0 finals don't exist in MLB/NBA/NHL — postponed game. 6 Jun NYY/BOS and
    # 11 Jun CWS/ATL were both graded LOSS at 0-0, poisoning calibration data.
    from app.ui.results_dashboard import determine_display_outcome
    from scripts.grade_slate import grade

    row = pd.Series(
        {"best_pick": "Over 8.5", "actual_home_score": 0, "actual_away_score": 0, "Outcome": pd.NA}
    )
    assert determine_display_outcome(row) == "N/A"

    export = pd.DataFrame(
        {
            "league": ["MLB"], "Home": ["Chicago White Sox"], "Away": ["Atlanta"],
            "best_pick": ["Over 8.5"], "WinProbability": [0.6],
            "effective_win_probability": [0.6], "effective_edge": [0.1],
            "consensus_agreement": ["Neutral"], "kalshi_probability": [0.6],
            "Kelly_Bet_Size": [10], "odds_american": [-110],
        }
    )
    recap = pd.DataFrame(
        {
            "Home": ["Chicago White Sox"], "Away": ["Atlanta"], "Pick Taken": ["Over 8.5"],
            "actual_home_score": [0], "actual_away_score": [0],
            "Outcome": ["LOSS"], "Status": ["Below Threshold"],
        }
    )
    export.to_csv(tmp_path / "e.csv", index=False)
    recap.to_csv(tmp_path / "r.csv", index=False)
    grade(tmp_path / "e.csv", tmp_path / "r.csv", tmp_path / "g.csv")
    assert len(pd.read_csv(tmp_path / "g.csv")) == 0


def test_line_drift_detection_and_exclusion(tmp_path, capsys):
    # The 11 Jun failure mode: recap graded 'Under 5.5' (stale morning card)
    # while the final export's pick was 'Under 10.5'. The drifted game must be
    # excluded from the graded output and reported, while clean rows grade.
    from scripts.grade_slate import find_line_drift, grade

    export = pd.DataFrame(
        {
            "league": ["MLB", "MLB"],
            "Home": ["New York Mets", "Baltimore"],
            "Away": ["Saint Louis", "Seattle"],
            "best_pick": ["Under 10.5", "Over 8.5"],
            "WinProbability": [0.58, 0.63],
            "effective_win_probability": [0.58, 0.63],
            "effective_edge": [0.1, 0.1],
            "consensus_agreement": ["Disagrees", "Agrees"],
            "kalshi_probability": [0.33, 0.6],
            "Kelly_Bet_Size": [7.77, 40.0],
            "odds_american": [104, -104],
            "export_run_id": ["20260611T191418Z", "20260611T191418Z"],
        }
    )
    recap = pd.DataFrame(
        {
            "Home": ["New York Mets", "Baltimore"],
            "Away": ["Saint Louis", "Seattle"],
            "Pick Taken": ["Under 5.5", "Over 8.5"],
            "Outcome": ["LOSS", "WIN"],
            "Status": ["No Play", "Below Threshold"],
            "export_run_id": ["20260611T080000Z", "20260611T080000Z"],
        }
    )

    drift = find_line_drift(export, recap)
    assert len(drift) == 1
    assert drift[0]["export_pick"] == "Under 10.5"
    assert drift[0]["recap_pick"] == "under 5.5"

    export.to_csv(tmp_path / "e.csv", index=False)
    recap.to_csv(tmp_path / "r.csv", index=False)
    grade(tmp_path / "e.csv", tmp_path / "r.csv", tmp_path / "g.csv")
    graded = pd.read_csv(tmp_path / "g.csv")
    # Only the clean Baltimore row grades; the drifted Mets line is excluded.
    assert len(graded) == 1
    assert graded.iloc[0]["Home"] == "Baltimore"
    err = capsys.readouterr().err
    assert "LINE DRIFT" in err
    assert "RUN ID MISMATCH" in err


def test_run_id_match_produces_no_warning():
    from scripts.grade_slate import _run_id_mismatch

    exp = pd.DataFrame({"export_run_id": ["A", "A"]})
    rec = pd.DataFrame({"export_run_id": ["A"]})
    assert _run_id_mismatch(exp, rec) is None
    rec2 = pd.DataFrame({"export_run_id": ["B"]})
    assert _run_id_mismatch(exp, rec2) == ("A", "B")
    # Missing column on either side -> no claim either way
    assert _run_id_mismatch(exp, pd.DataFrame({"x": [1]})) is None


def test_results_dashboard_grades_unresolved_pick_as_na():
    from app.ui.results_dashboard import determine_display_outcome

    # The exact 10 Jun recap row that was wrongly graded LOSS
    row = pd.Series(
        {
            "best_pick": "San Antonio line unresolved",
            "home_team": "New York",
            "away_team": "San Antonio",
            "actual_home_score": 107,
            "actual_away_score": 106,
            "Outcome": pd.NA,
        }
    )
    assert determine_display_outcome(row) == "N/A"

    # A normal total still grades from scores
    total_row = pd.Series(
        {
            "best_pick": "Under 7.5",
            "actual_home_score": 7,
            "actual_away_score": 5,
            "Outcome": pd.NA,
        }
    )
    assert determine_display_outcome(total_row) == "LOSS"
