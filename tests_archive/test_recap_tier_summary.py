"""Tiered recap summary — staked performance, not the all-rows blend.

Locks summarize_recap_tiers so the Performance Recap headline reflects the card that
is actually bet (Actionable, Actionable+HV) instead of a single all-rows hit rate that
includes unstaked Below Threshold / No Play picks. Built around the real 8 Jun recap,
where the all-rows number was a misleading 44% while the staked card went 3-1.
"""
import pandas as pd

from app_core.strategy_lab_realized import summarize_recap_tiers


def _jun8_recap():
    # (Status, Outcome) for the graded 8 Jun card.
    rows = [
        ("Actionable", "WIN"),                      # Toronto/Phil Under 7.5
        ("High Variance/Speculative", "WIN"),       # SD/Cin Over 7.5
        ("High Variance/Speculative", "WIN"),       # Cle/NYY Over 7.5
        ("High Variance/Speculative", "LOSS"),      # NBA NY/SA Under
        ("Below Threshold", "LOSS"),                # LAA Over 9.5
        ("Below Threshold", "LOSS"),                # Baltimore Over 9.5
        ("Below Threshold", "LOSS"),                # SF Over 8.5
        ("Below Threshold", "WIN"),                 # Athletics Over 10.5
        ("No Play", "LOSS"),                        # Boston -1.5
    ]
    return pd.DataFrame(rows, columns=["Status", "Outcome"])


def test_tiers_separate_staked_from_all_rows():
    t = summarize_recap_tiers(_jun8_recap()).set_index("Tier")

    # Actionable (production): 1-0
    assert (t.loc["Actionable (production)", ["Wins", "Losses", "Total"]].tolist()) == [1, 0, 1]
    assert t.loc["Actionable (production)", "Hit Rate"] == 1.0

    # Actionable + High Variance: 3-1 (the staked+speculative card)
    assert (t.loc["Actionable + High Variance", ["Wins", "Losses", "Total"]].tolist()) == [3, 1, 4]
    assert t.loc["Actionable + High Variance", "Hit Rate"] == 0.75

    # All graded rows: 4-5 — the misleading 44% headline
    allrow = t.loc["All graded rows (incl. unstaked)"]
    assert allrow[["Wins", "Losses", "Total"]].tolist() == [4, 5, 9]
    assert round(allrow["Hit Rate"], 3) == 0.444


def test_accepts_win_boolean_and_w_l_shorthand():
    df = pd.DataFrame({"Status": ["Actionable", "Actionable"], "Win": [True, False]})
    assert summarize_recap_tiers(df).set_index("Tier").loc["Actionable (production)", "Total"] == 2

    df2 = pd.DataFrame({"Status": ["Actionable", "High Variance/Speculative"], "Outcome": ["W", "L"]})
    t2 = summarize_recap_tiers(df2).set_index("Tier")
    assert t2.loc["Actionable + High Variance", ["Wins", "Losses"]].tolist() == [1, 1]


def test_empty_tier_is_zero_not_nan():
    df = pd.DataFrame({"Status": ["Below Threshold"], "Outcome": ["LOSS"]})
    t = summarize_recap_tiers(df).set_index("Tier")
    # No Actionable picks -> 0/0 must report 0.0, not NaN/crash.
    assert t.loc["Actionable (production)", "Total"] == 0
    assert t.loc["Actionable (production)", "Hit Rate"] == 0.0
