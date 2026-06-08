"""Edge-based no-stake gates for MLB totals (graded 20 May-7 Jun, n=182).

Locks the two buckets that bled below the -110 breakeven and must never be staked:
  * Rule 1 — Neutral-consensus MLB totals (Over/Neutral 48.2%, n=56).
  * Rule 2 — mid-line MLB Overs, line in [8.0, 9.5) (46.5%, n=43).
Both demote a stakeable pick to Below Threshold; Agrees/Disagrees totals, sub-8.0 and
>=9.5 Overs, Unders, and non-MLB rows are untouched. See scripts/edge_by_bucket.py.
"""
from core.streamlit_pipeline import _edge_no_stake_demotion

KW = dict(neutral_no_stake=True, mid_line_no_stake=True, mid_min=8.0, mid_max=9.5)


def _demote(**over):
    base = dict(league="MLB", market_type="total_over", consensus="Agrees",
                total_line=9.5, status="Actionable", **KW)
    base.update(over)
    return _edge_no_stake_demotion(**base)


def test_rule1_neutral_total_is_held_below_stake():
    status, _reason, blocker = _demote(consensus="Neutral", market_type="total_over", total_line=10.5)
    assert status == "Below Threshold"
    assert blocker == "mlb_total_neutral_no_stake"
    # Applies to Unders too (any direction), at any line.
    status_u, _r, blocker_u = _demote(consensus="Neutral", market_type="total_under", total_line=7.5)
    assert status_u == "Below Threshold" and blocker_u == "mlb_total_neutral_no_stake"


def test_rule2_mid_line_over_is_held_below_stake():
    for line in (8.0, 8.5, 9.0, 9.49):
        status, _reason, blocker = _demote(consensus="Agrees", market_type="total_over", total_line=line)
        assert status == "Below Threshold", line
        assert blocker == "mlb_over_mid_line_no_stake", line


def test_high_and_low_line_agrees_overs_are_untouched():
    # >= 9.5 high-line over (65.0% bucket) keeps stake.
    assert _demote(consensus="Agrees", total_line=9.5) == (None, None, None)
    assert _demote(consensus="Agrees", total_line=10.5) == (None, None, None)
    # sub-8.0 over is handled by the separate low_line guardrail, not these gates.
    assert _demote(consensus="Agrees", total_line=7.5) == (None, None, None)


def test_agrees_and_disagrees_keep_edge():
    # Mid-line UNDER is the edge side (65.4%) — not demoted by the over rule.
    assert _demote(consensus="Agrees", market_type="total_under", total_line=8.5) == (None, None, None)
    # Disagrees over at a non-mid line keeps stake.
    assert _demote(consensus="Disagrees", total_line=10.5) == (None, None, None)


def test_only_demotes_stakeable_rows():
    # Already Below Threshold / No Play is left alone (never promotes, never re-labels).
    assert _demote(consensus="Neutral", status="Below Threshold", total_line=10.5) == (None, None, None)
    # High Variance Neutral total is demoted (it would otherwise be stakeable in HV mode).
    s, _r, b = _demote(consensus="Neutral", status="High Variance/Speculative", total_line=10.5)
    assert s == "Below Threshold" and b == "mlb_total_neutral_no_stake"


def test_non_mlb_rows_untouched():
    assert _demote(league="NBA", consensus="Neutral", total_line=8.5) == (None, None, None)
    assert _demote(league="NHL", market_type="total_over", consensus="Agrees", total_line=8.5) == (None, None, None)


def test_flags_disable_each_rule():
    off = dict(KW)
    off["neutral_no_stake"] = False
    assert _edge_no_stake_demotion(league="MLB", market_type="total_over", consensus="Neutral",
                                   total_line=10.5, status="Actionable", **off) == (None, None, None)
    off2 = dict(KW)
    off2["mid_line_no_stake"] = False
    assert _edge_no_stake_demotion(league="MLB", market_type="total_over", consensus="Agrees",
                                   total_line=8.5, status="Actionable", **off2) == (None, None, None)
