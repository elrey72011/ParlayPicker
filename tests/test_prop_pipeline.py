"""Strikeout-prop scoring/assembly (pure -- no network)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.prop_pipeline import score_strikeout_prop, evaluate_strikeout_props


# A high-strikeout ace: 11 K/9 over 6 IP -> ~7.3 expected Ks.
_ACE = {"k_per_9": 11.0, "avg_innings": 6.0, "n_games": 5}
# A low-strikeout starter: 6 K/9 over 5.5 IP -> ~3.7 expected Ks.
_SOFT = {"k_per_9": 6.0, "avg_innings": 5.5, "n_games": 5}


# Pick'em pricing on a mismatched line produces a large (>cap) synthetic edge; these
# direction tests disable the implausible-edge cap to isolate the over/under logic. The
# cap itself is covered by test_implausible_edge_is_suppressed.
_NO_CAP = dict(max_plausible_edge=1.0)


def test_ace_vs_low_line_recommends_over():
    # Ace projected ~7.3 Ks, line 5.5 priced near pick'em -> model loves the over.
    row = {"pitcher": "Ace", "line": 5.5, "over_odds": -110, "under_odds": -110}
    s = score_strikeout_prop(row, _ACE, **_NO_CAP)
    assert s["recommendation"] == "over"
    assert s["best_edge"] >= 0.04
    assert s["best_ev"] > 0


def test_soft_arm_high_line_recommends_under():
    # Soft starter ~3.7 Ks, line 6.5 priced near pick'em -> model loves the under.
    row = {"pitcher": "Soft", "line": 6.5, "over_odds": -110, "under_odds": -110}
    s = score_strikeout_prop(row, _SOFT, **_NO_CAP)
    assert s["recommendation"] == "under"


def test_priced_in_edge_is_no_play():
    # Ace model p_over ~0.72; market -260/+210 de-vigs to ~0.69 over -> only +3% edge,
    # below the 4% bar, so it's a no-play (the line already reflects the ace).
    row = {"pitcher": "Ace", "line": 5.5, "over_odds": -260, "under_odds": 210}
    s = score_strikeout_prop(row, _ACE)
    assert s["recommendation"] == "no_play"
    assert 0 < s["best_edge"] < 0.04


def test_missing_form_is_no_data():
    row = {"pitcher": "Unknown", "line": 6.5, "over_odds": -110, "under_odds": -110}
    s = score_strikeout_prop(row, None)
    assert s["recommendation"] == "no_data"
    assert s["best_edge"] is None


def test_over_under_edges_are_mirror_images():
    row = {"pitcher": "Ace", "line": 6.5, "over_odds": -110, "under_odds": -110}
    s = score_strikeout_prop(row, _ACE)
    # The chosen side's edge is the positive of the symmetric pair.
    assert s["best_edge"] >= 0


def test_evaluate_uses_lookups():
    props = [
        {"pitcher": "Ace", "line": 5.5, "over_odds": -110, "under_odds": -110},
        {"pitcher": "Soft", "line": 6.5, "over_odds": -110, "under_odds": -110},
    ]
    forms = {"Ace": _ACE, "Soft": _SOFT}
    scored = evaluate_strikeout_props(props, form_lookup=forms.get, max_plausible_edge=1.0)
    recs = {r["pitcher"]: r["recommendation"] for r in scored}
    assert recs["Ace"] == "over"
    assert recs["Soft"] == "under"


def test_high_k_opponent_raises_projection():
    row = {"pitcher": "Mid", "line": 5.5, "over_odds": -110, "under_odds": -110}
    mid = {"k_per_9": 8.0, "avg_innings": 5.5, "n_games": 5}  # ~4.9 Ks neutral
    base = score_strikeout_prop(row, mid)["expected_ks"]
    boosted = score_strikeout_prop(row, mid, opponent_k_rate=0.27)["expected_ks"]
    assert boosted > base


def test_insufficient_starter_sample_is_no_data():
    # Opener / thin sample: too few games or a reliever-shaped workload -> not projectable.
    row = {"pitcher": "Opener", "line": 4.5, "over_odds": -110, "under_odds": -110}
    thin = {"k_per_9": 9.0, "avg_innings": 1.2, "n_games": 2}  # ~1.2 Ks, opener
    s = score_strikeout_prop(row, thin)
    assert s["recommendation"] == "no_data"
    assert "insufficient starter sample" in s["reason"]
    assert s["best_edge"] is None


def test_low_games_count_is_no_data():
    row = {"pitcher": "Rookie", "line": 5.5, "over_odds": -110, "under_odds": -110}
    # Starter-length innings but only one logged start -> sample too thin to trust.
    one_start = {"k_per_9": 10.0, "avg_innings": 5.5, "n_games": 1}
    s = score_strikeout_prop(row, one_start)
    assert s["recommendation"] == "no_data"


def test_over_contradicting_projection_is_no_play():
    # Reproduces Kumar Rocker Over 4.5 (29 Jun): projection ~4.16 Ks sits UNDER the 4.5 line,
    # but a +178 over price is cheaper than the model's sub-50% over prob, so edge_over wins
    # the side selection. The projection-side gate must refuse to stake an Over the model's
    # own expected-Ks read points away from.
    row = {"pitcher": "Rocker", "line": 4.5, "over_odds": 178, "under_odds": -230}
    soft = {"k_per_9": 6.8, "avg_innings": 5.5, "n_games": 5}  # ~4.16 Ks, under the line
    s = score_strikeout_prop(row, soft, **_NO_CAP)
    assert s["best_side"] == "over"
    assert s["expected_ks"] < row["line"]
    assert s["recommendation"] == "no_play"
    assert "contradicts projection" in s["reason"]


def test_projection_consistent_side_still_stakes():
    # The gate must NOT touch picks where the side agrees with the projection: ace ~7.3 Ks,
    # line 5.5, model loves the over and the projection (7.3) is above the line.
    row = {"pitcher": "Ace", "line": 5.5, "over_odds": -110, "under_odds": -110}
    s = score_strikeout_prop(row, _ACE, **_NO_CAP)
    assert s["best_side"] == "over"
    assert s["expected_ks"] > row["line"]
    assert s["recommendation"] == "over"


def test_implausible_edge_is_suppressed():
    # Ace vs a pick'em-priced low line yields a >12% edge -> projection error, suppressed.
    row = {"pitcher": "Ace", "line": 5.5, "over_odds": -110, "under_odds": -110}
    s = score_strikeout_prop(row, _ACE)  # default cap (0.12) active
    assert s["best_edge"] > 0.12
    assert s["recommendation"] == "no_play"
    assert "plausible cap" in s["reason"]
