"""Kalshi extreme-price guard (21 Jun).

Books price totals and run lines near pick'em, so a de-vigged Kalshi price at near-
certainty is a settled/illiquid/mis-scraped market, not a read. 20 Jun: a "Total Runs
over 12.5" contract sat at bid 0.99 / ask 1.00 (mid 0.995) -- a tight spread that slipped
past the >0.40 illiquidity guard -- and inflated a blend to a fake 0.72 win prob behind
the $750 Cubs landmine. Moneyline is exempt (heavy favorites legitimately price extreme).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.kalshi_integrator import kalshi_total_spread_price_extreme


def test_extreme_total_price_is_flagged():
    assert kalshi_total_spread_price_extreme(0.995, "total_over") is True
    assert kalshi_total_spread_price_extreme(0.02, "total_under") is True


def test_extreme_runline_price_is_flagged():
    assert kalshi_total_spread_price_extreme(0.97, "spread_home") is True


def test_normal_total_price_is_trusted():
    # Real totals leans live well inside the band.
    assert kalshi_total_spread_price_extreme(0.52, "total_over") is False
    assert kalshi_total_spread_price_extreme(0.80, "total_under") is False


def test_moneyline_extremes_are_exempt():
    # A heavy favorite can legitimately price near-certainty on the moneyline.
    assert kalshi_total_spread_price_extreme(0.98, "moneyline") is False
    assert kalshi_total_spread_price_extreme(0.97, "h2h_home") is False


def test_missing_price_is_not_flagged():
    assert kalshi_total_spread_price_extreme(None, "total_over") is False
