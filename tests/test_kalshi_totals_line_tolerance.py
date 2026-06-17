"""MLB totals must match a Kalshi contract at (near) the SAME line.

Regression for the 17 Jun all-Over card: the generic MAX_LINE_TOLERANCE['MLB']
(8.5 runs) let an 'Over 7.5' pick match a Kalshi 'Over 5.5' contract — whose
P(over) is naturally high — systematically inflating Kalshi P(over) to ~0.55-0.62
on every game. Totals now use the tight MAX_TOTAL_LINE_TOLERANCE (1.0 for MLB), so
a far-off line is a miss (Kalshi absent) rather than a wrong-line proxy.
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core import kalshi_integrator as ki


class _Resp:
    def __init__(self, data, status_code=200):
        self._d = data
        self.status_code = status_code

    def json(self):
        return self._d


def _df():
    return pd.DataFrame(
        {
            "league": ["MLB"],
            "market_type": ["total_over"],
            "home_team": ["Seattle Mariners"],
            "away_team": ["Baltimore Orioles"],
            "game_date": ["2026-06-17T00:00:00Z"],
            "total_line": [7.5],
            "best_pick": ["Over 7.5"],
        }
    )


def _make_request_factory(kalshi_total_line: float, p_over: float):
    """Fake Kalshi API exposing a single MLB total contract at the given line."""
    yes = round(p_over, 2)

    def fake_make_request(url, **kwargs):
        if url.endswith("/events") and "series_ticker" in kwargs.get("params", {}):
            return _Resp({
                "events": [{
                    "event_ticker": "KXMLBTOTAL-26JUN17SEABAL",
                    "title": "Baltimore Orioles at Seattle Mariners: Total Runs",
                    "sub_title": "BAL at SEA (Jun 17)",
                    "close_time": "2026-06-17T23:00:00Z",
                }]
            })
        if "KXMLBTOTAL-26JUN17SEABAL" in url:
            return _Resp({"event": {"markets": [{
                "ticker": "KXMLBTOTAL-26JUN17SEABAL-T%s" % kalshi_total_line,
                "event_ticker": "KXMLBTOTAL-26JUN17SEABAL",
                "title": "Over %s runs scored" % kalshi_total_line,
                "yes_bid_dollars": yes - 0.02,
                "yes_ask_dollars": yes + 0.02,
            }]}})
        return _Resp({"error": "not found"}, 404)

    return fake_make_request


def _enrich(monkeypatch, kalshi_line, p_over):
    if hasattr(ki.enrich_with_kalshi_markets, "series_cache"):
        ki.enrich_with_kalshi_markets.series_cache.clear()
    monkeypatch.setattr(ki, "_make_kalshi_request", _make_request_factory(kalshi_line, p_over))
    return ki.enrich_with_kalshi_markets(_df())


def test_same_line_total_matches(monkeypatch):
    # Kalshi contract at the SAME 7.5 line -> matched, P(over) used as-is (~0.50).
    out = _enrich(monkeypatch, 7.5, 0.50)
    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert abs(float(out.loc[0, "kalshi_probability"]) - 0.50) < 0.05


def test_match_emits_diagnostic_columns(monkeypatch):
    # Instrumentation: a matched total exposes the contract line used, the line delta,
    # and the raw P(over) before orientation/decay, so a systematic bias is visible.
    out = _enrich(monkeypatch, 7.5, 0.50)
    assert abs(float(out.loc[0, "kalshi_matched_line"]) - 7.5) < 1e-9
    assert float(out.loc[0, "kalshi_line_diff"]) == 0.0
    assert abs(float(out.loc[0, "kalshi_raw_over_prob"]) - 0.50) < 0.05


def test_far_line_total_is_a_miss_not_an_inflated_proxy(monkeypatch):
    # Only a far 5.5 contract exists (delta 2.0 > MLB totals tolerance 1.0). Pre-fix
    # this proxied a ~0.70 P(over) down to ~0.62 and corrupted the blend; now it misses.
    out = _enrich(monkeypatch, 5.5, 0.70)
    assert out.loc[0, "kalshi_match_status"] != "matched"
    prob = out.loc[0, "kalshi_probability"]
    assert pd.isna(prob) or float(prob) == 0.0 or float(prob) < 0.55


def test_mlb_totals_tolerance_is_tight():
    assert ki.MAX_TOTAL_LINE_TOLERANCE["MLB"] <= 1.0
    assert ki.MAX_TOTAL_LINE_TOLERANCE["MLB"] < ki.MAX_LINE_TOLERANCE["MLB"]
