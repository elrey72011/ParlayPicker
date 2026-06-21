"""Assemble the pitcher-strikeout prop slice: odds + pitcher form -> scored card.

Ties together the three components (prop_odds_ingest, mlb_pitcher_stats, prop_model): for
each strikeout prop, project the pitcher's expected Ks, get the model's P(over/under),
de-vig the market price, and surface the edge / EV with the same no-edge-no-stake
discipline as the rest of the system. The scoring is pure and unit-tested; orchestration
(fetching odds + form) is a thin wrapper.

Delivered standalone first so it can be run + calibrated before any integration into the
main best-picks card.
"""
from __future__ import annotations

from app_core.prop_model import project_expected_strikeouts, strikeout_pick_probability

# Strikeouts are mildly overdispersed vs Poisson; NB with variance ~1.15x mean.
PROP_KS_DISPERSION = 1.15
# Discipline: no stake without a real edge over the de-vigged market price.
PROP_MIN_EDGE = 0.04


def _american_to_implied(odds: float) -> float:
    odds = float(odds)
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / ((-odds) + 100.0)


def _american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / (-odds))


def score_strikeout_prop(
    prop_row: dict,
    form: dict | None,
    opponent_k_rate: float | None = None,
    dispersion: float = PROP_KS_DISPERSION,
    min_edge: float = PROP_MIN_EDGE,
) -> dict:
    """Score one strikeout prop. Returns the row enriched with model/market/edge/EV and a
    recommendation. With no pitcher form (no projection) it's a 'no_data' no-play.
    """
    out = dict(prop_row)
    line = float(prop_row["line"])
    over_odds = prop_row["over_odds"]
    under_odds = prop_row["under_odds"]

    if not form or not form.get("k_per_9") or not form.get("avg_innings"):
        out.update(recommendation="no_data", reason="no pitcher form", best_edge=None, best_ev=None)
        return out

    lam = project_expected_strikeouts(form["k_per_9"], form["avg_innings"], opponent_k_rate)
    p_over = strikeout_pick_probability(lam, line, "over", dispersion)
    p_under = 1.0 - p_over

    io, iu = _american_to_implied(over_odds), _american_to_implied(under_odds)
    mkt_over = io / (io + iu)
    mkt_under = iu / (io + iu)

    edge_over = p_over - mkt_over
    edge_under = p_under - mkt_under  # == -edge_over (both pairs sum to 1)

    if edge_over >= edge_under:
        side, p_side, edge, odds = "over", p_over, edge_over, over_odds
    else:
        side, p_side, edge, odds = "under", p_under, edge_under, under_odds
    ev = p_side * (_american_to_decimal(odds) - 1.0) - (1.0 - p_side)

    rec = side if (edge >= min_edge and ev > 0) else "no_play"
    out.update(
        expected_ks=round(lam, 2),
        model_p_over=round(p_over, 4),
        market_p_over=round(mkt_over, 4),
        best_side=side,
        best_edge=round(edge, 4),
        best_ev=round(ev, 4),
        recommendation=rec,
        reason=(
            f"{side} edge {edge:+.1%} vs market; EV {ev:+.1%}"
            if rec != "no_play"
            else f"edge {edge:+.1%} below {min_edge:.0%} bar or EV<=0"
        ),
    )
    return out


def evaluate_strikeout_props(
    props: list[dict],
    form_lookup,
    opp_k_lookup=None,
    dispersion: float = PROP_KS_DISPERSION,
    min_edge: float = PROP_MIN_EDGE,
) -> list[dict]:
    """Score a list of parsed strikeout props.

    ``form_lookup(pitcher) -> form dict | None`` and optional ``opp_k_lookup(prop_row) ->
    float | None`` supply per-pitcher form and opponent strikeout rate (injected so this is
    testable without network).
    """
    scored = []
    for row in props:
        form = form_lookup(row.get("pitcher")) if form_lookup else None
        opp_k = opp_k_lookup(row) if opp_k_lookup else None
        scored.append(score_strikeout_prop(row, form, opp_k, dispersion, min_edge))
    return scored
