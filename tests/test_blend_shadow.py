"""Forward shadow: the app's blend vs the market price alone. SYNTHETIC picks."""
import pandas as pd
import pytest

from core import blend_shadow as bs


def _pick(day, blend, market, outcome, market_type="spread_home", sport="NCAAB"):
    return {"game_start_utc": f"{day}T23:00:00Z", "calibrated_probability": blend, "fair_market_probability": market,
            "candidate_outcome": outcome, "market_type": market_type, "sport": sport}


def test_only_picks_from_the_start_date_and_too_few_until_100():
    rows = [_pick("2026-11-05", 0.9, 0.5, "LOSS"), _pick("2026-11-06", 0.6, 0.5, "WIN"),
            _pick("2026-11-07", 0.6, 0.5, "LOSS", market_type="total_over")]
    report = bs.build_from_rows(rows)
    assert report["by_market"]["ALL"]["decided"] == 2                       # the 11-05 pick is before the start
    assert report["by_market"]["NCAAB SPREAD"]["decided"] == 1
    assert report["by_market"]["ALL"]["verdict"].startswith("TOO_FEW_DECIDED")
    assert "Blend vs market price: forward shadow" in bs.markdown(report)


def test_verdict_after_enough_picks_and_market_fallback():
    # The blend is overconfident (0.70) on coin flips the market prices at 0.50: the market wins.
    rows = [_pick("2026-11-06", 0.70, 0.50, "WIN" if i % 2 else "LOSS") for i in range(120)]
    s = bs.build_from_rows(rows)["by_market"]["ALL"]
    assert s["verdict"] == "MARKET_AS_GOOD_OR_BETTER" and s["log_loss_gain_vs_market"] < 0
    sharp = [_pick("2026-11-06", 0.8 if i % 2 else 0.2, 0.5, "WIN" if i % 2 else "LOSS") for i in range(120)]
    assert bs.build_from_rows(sharp)["by_market"]["ALL"]["verdict"] == "BLEND_BETTER"
    fallback = dict(_pick("2026-11-06", 0.6, None, "WIN"), market_probability=0.55)
    assert bs.build_from_rows([fallback])["by_market"]["ALL"]["market_log_loss"] == pytest.approx(-__import__("math").log(0.55))


def test_board_market_shadow_reads_the_saved_candidate():
    from app.ui.daily_dashboard import market_shadow
    result = pd.DataFrame({"candidate_id": ["a", "b", "c"], "odds": [-110, 150, None]})
    candidates = pd.DataFrame({"candidate_id": ["a", "b", "c"], "fair_market_probability": [0.52, None, 0.5],
                               "market_probability": [0.55, 0.41, 0.5]})
    probs, ev = market_shadow(result, candidates)
    assert list(probs.round(2)) == [0.52, 0.41, 0.5]                        # no-vig first, else market
    assert ev.iloc[0] == pytest.approx(0.52 * (1 + 100 / 110) - 1) and ev.iloc[1] == pytest.approx(0.41 * 2.5 - 1)
    assert pd.isna(ev.iloc[2])                                              # no odds: no EV
    assert market_shadow(result, None) is None
    assert market_shadow(result, pd.DataFrame({"candidate_id": ["a"]})) is None


def test_pinnacle_is_scored_on_the_picks_that_have_it():
    rows = [dict(_pick("2026-11-06", 0.6, 0.5, "WIN"), pinnacle_fair_probability=0.55),
            _pick("2026-11-06", 0.6, 0.5, "LOSS")]
    s = bs.build_from_rows(rows)["by_market"]["ALL"]
    assert s["decided"] == 2 and s["pinnacle"]["decided"] == 1
    assert s["pinnacle"]["pinnacle_log_loss"] == pytest.approx(-__import__("math").log(0.55))
    assert "On the picks that also have Pinnacle" in bs.markdown(bs.build_from_rows(rows))


def test_blend_study_fits_kalshi_and_pinnacle_together():
    from core.blend_weights import INPUTS, CURRENT
    assert INPUTS["pinnacle"] == "pinnacle_fair_probability" and "pinnacle" in CURRENT


def test_board_market_shadow_falls_back_to_game_market_and_pick():
    """Overall board rows come from the final export, which has no candidate_id (2026-11-06)."""
    from app.ui.daily_dashboard import market_shadow
    result = pd.DataFrame({"candidate_id": ["", ""], "matchup_id": ["g1", "g2"], "market_type": ["spread_home", "total_over"],
                           "pick": ["Home -3.5", "Over 140.5"], "odds": [-110, 100]})
    candidates = pd.DataFrame({"candidate_id": ["a", "b", "c"], "matchup_id": ["g1", "g2", "g2"],
                               "market_type": ["spread_home", "total_over", "total_over"],
                               "best_pick": ["Home -3.5", "Over 140.5", "Over 140.5"],
                               "market_probability": [0.52, 0.5, 0.49]})
    probs, ev = market_shadow(result, candidates)
    assert probs.tolist()[0] == 0.52 and abs(ev[0] - (0.52 * (1 + 100 / 110) - 1)) < 1e-9
    assert pd.isna(probs[1])                 # two candidates share the key: ambiguous, left empty
