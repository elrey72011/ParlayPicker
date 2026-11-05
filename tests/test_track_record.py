"""Paper track record summary. SYNTHETIC accepted rows; no network."""
from core import track_record as tr


def _row(sport, market_type, outcome, odds=-110, cid=None):
    return {"sport": sport, "market_type": market_type, "candidate_outcome": outcome,
            "odds_american": odds, "candidate_id": cid}


def test_summary_counts_units_roi_interval_and_clv():
    rows = [_row("NCAAB", "spread_home", "WIN", cid="a"), _row("NCAAB", "spread_away", "LOSS", cid="b"),
            _row("NCAAB", "spread_home", "PUSH"), _row("NBA", "total_over", "WIN", odds=120)]
    report = tr.summarize(rows, closes={"a": True, "b": False})
    spread = report["NCAAB SPREAD"]
    assert (spread["picks"], spread["wins"], spread["losses"], spread["pushes"]) == (3, 1, 1, 1)
    assert spread["hit_rate"] == 0.5 and abs(spread["units"] - (100 / 110 - 1)) < 1e-9
    assert spread["hit_rate_95_low"] < 0.5 < spread["hit_rate_95_high"]
    assert abs(spread["break_even_hit_rate"] - 110 / 210) < 1e-9
    assert spread["with_close"] == 2 and spread["beat_close_share"] == 0.5
    assert abs(report["NBA TOTAL"]["units"] - 1.2) < 1e-9
    assert report["ALL"]["picks"] == 4


def test_wilson_and_markdown():
    low, high = tr.wilson(55, 100)
    assert 0.45 < low < 0.46 and 0.64 < high < 0.65
    assert tr.wilson(0, 0) == (None, None)
    text = tr.markdown({"settled_picks": 1, "exclusions": {"unresolved_outcome": 3},
                        "by_market": tr.summarize([_row("NHL", "total_under", "WIN")])})
    assert "| NHL TOTAL | 1 | 1-0-0 |" in text and "no closes" in text and "unresolved_outcome 3" in text
