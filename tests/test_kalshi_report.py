"""Kalshi information report on settled saved picks. SYNTHETIC rows."""
import random

from core import kalshi_report as kr


def _row(k, m, outcome, market="spread_home", sport="NCAAB"):
    return {"sport": sport, "market_type": market, "kalshi_probability": k, "fair_market_probability": m,
            "candidate_outcome": outcome}


def test_rows_need_both_prices_and_a_decided_outcome():
    rows = kr._rows([_row(0.6, 0.5, "WIN"), _row(None, 0.5, "WIN"), _row(0.6, None, "WIN"),
                     _row(0.6, 0.5, "PUSH"), _row(0.6, 0.5, "WIN", market="moneyline_home")])
    assert len(rows) == 1 and rows[0] == {"group": "NCAAB SPREAD", "k": 0.6, "m": 0.5, "y": 1}


def test_noise_kalshi_is_not_more_accurate_than_an_informed_book():
    rng = random.Random(3)
    rows = []
    for _ in range(400):
        truth = rng.uniform(0.35, 0.65)
        outcome = "WIN" if rng.random() < truth else "LOSS"
        rows.append(_row(min(0.95, max(0.05, 0.5 + rng.gauss(0, 0.08))), truth, outcome))
    report = kr.build_from_rows(rows)
    s = report["by_market"]["ALL"]
    assert s["decided"] == 400 and s["brier_kalshi"] > s["brier_market"]
    assert s["verdict"] == "KALSHI_NOT_MORE_ACCURATE"
    assert s["tier1"]["picks"] == sum(1 for r in rows if r["kalshi_probability"] >= 0.55)
    text = kr.markdown(report)
    assert "Kalshi vs the book price" in text and "KALSHI_NOT_MORE_ACCURATE" in text


def test_small_samples_get_no_verdict():
    report = kr.build_from_rows([_row(0.6, 0.5, "WIN"), _row(0.4, 0.5, "LOSS")])
    assert report["by_market"]["ALL"]["verdict"].startswith("TOO_FEW_DECIDED")


def test_empty_report_renders():
    assert "no settled picks with a Kalshi price yet" in kr.markdown(kr.build_from_rows([]))
