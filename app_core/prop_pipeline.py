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

from app_core.prop_model import (
    project_expected_outs,
    project_expected_strikeouts,
    project_expected_walks,
    strikeout_pick_probability,
)

# Strikeouts are mildly overdispersed vs Poisson; NB with variance ~1.15x mean.
PROP_KS_DISPERSION = 1.15

# ── Pitcher-prop market specs (4 Jul expansion: outs + walks) ──
# Each market is a count stat scored by the SAME engine (NB distribution, de-vig,
# min-edge/+EV gate, starter guards, projection-consistency gate, implausible-edge
# cap). Dispersion (variance/mean) is set conservatively per stat: wider variance
# shrinks model probabilities toward 50%, which UNDERSTATES edges until a graded
# record justifies tightening.
#   * outs: leash/manager-driven; recent avg innings is the direct signal. 1.30.
#   * walks: low-mean Poisson-ish count. 1.10.
# Labels feed pick text ("Over 16.5 Outs"); market_type = f"{key}_{side}".
PITCHER_PROP_SPECS = {
    "pitcher_strikeouts": {"label": "Ks", "dispersion": PROP_KS_DISPERSION},
    "pitcher_outs": {"label": "Outs", "dispersion": 1.30},
    "pitcher_walks": {"label": "BBs", "dispersion": 1.10},
}


def _project_for_market(market_key: str, form: dict, opponent_k_rate: float | None):
    """Expected count for one market from pitcher form, or None if unprojectable."""
    if market_key == "pitcher_outs":
        return project_expected_outs(form["avg_innings"])
    if market_key == "pitcher_walks":
        if not form.get("bb_per_9"):
            return None
        return project_expected_walks(form["bb_per_9"], form["avg_innings"])
    return project_expected_strikeouts(form["k_per_9"], form["avg_innings"], opponent_k_rate)
# Discipline: no stake without a real edge over the de-vigged market price.
PROP_MIN_EDGE = 0.04

# Owner preference (3 Jul): the card optimizes for WIN PROBABILITY, not ROI. The
# edge/EV gate had been admitting near-coin-flip plus-money picks (Rangel Over at
# 50.3% / +125, Sasaki Over at 53.9% / +130) whose value is in the PRICE, not the
# likelihood of cashing — mathematically sound, but not what the owner wants to
# bet. Every staked prop must now be a genuine favorite on the model's own number:
# p(win) >= this floor, and the card ranks by win probability (not edge). The
# min-edge/+EV gate above still applies — a favorite at a losing price stays
# unbettable — so this floor narrows the card to picks that are BOTH likely to
# win AND fairly priced.
PROP_MIN_WIN_PROBABILITY = 0.55

# Projection sanity guards (added after the first live slate produced implausible picks).
# The strikeout projection is only as good as its innings/role estimate. For openers,
# relievers, rookies, and pitchers with thin or relief-skewed game logs, avg_innings
# collapses, the expected-K projection collapses with it, and every Under looks like a
# lock — e.g. a starter the book prices at 4.5 Ks gets projected for 1.5 because the form
# sample has him at ~1 inning. Two guards keep these out of the card:
#   * MIN_STARTER_GAMES / MIN_STARTER_AVG_INNINGS — don't project a pitcher we can't
#     credibly model as a starter (the book line assumes a start). Conservative: skip when
#     the sample is too small or the workload looks like a reliever's.
#   * MAX_PLAUSIBLE_EDGE — in a liquid market a real edge is a few points; a 20-40% "edge"
#     is the model being wrong, not the market. Suppress anything above the cap until the
#     model has a graded record to justify trust. Set high to disable.
PROP_MIN_STARTER_GAMES = 3
PROP_MIN_STARTER_AVG_INNINGS = 4.0
PROP_MAX_PLAUSIBLE_EDGE = 0.12


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
    min_starter_games: int = PROP_MIN_STARTER_GAMES,
    min_starter_avg_innings: float = PROP_MIN_STARTER_AVG_INNINGS,
    max_plausible_edge: float = PROP_MAX_PLAUSIBLE_EDGE,
) -> dict:
    """Score one pitcher prop. Returns the row enriched with model/market/edge/EV and a
    recommendation. With no pitcher form (no projection) it's a 'no_data' no-play. Two sanity
    guards keep projection errors off the card: a starter-sample floor (skip pitchers we
    can't credibly model as a starter) and an implausible-edge cap (a 20-40% edge in this
    market is a model error, not real value).

    Multi-market (4 Jul): the row's ``market_key`` (default pitcher_strikeouts)
    selects the projection + dispersion from PITCHER_PROP_SPECS; the historic
    function/field names (score_strikeout_prop, expected_ks) are kept — expected_ks
    holds the projected count for whichever stat the row prices.
    """
    out = dict(prop_row)
    market_key = str(prop_row.get("market_key") or "pitcher_strikeouts")
    spec = PITCHER_PROP_SPECS.get(market_key, PITCHER_PROP_SPECS["pitcher_strikeouts"])
    if market_key != "pitcher_strikeouts":
        dispersion = float(spec["dispersion"])
    line = float(prop_row["line"])
    over_odds = prop_row["over_odds"]
    under_odds = prop_row["under_odds"]

    if not form or not form.get("k_per_9") or not form.get("avg_innings"):
        out.update(recommendation="no_data", reason="no pitcher form", best_edge=None, best_ev=None)
        return out

    # Starter-sample guard: the book line assumes a start, so a thin or reliever-shaped
    # sample (too few games, or avg innings below a starter's workload) means our projection
    # is modeling the wrong role — its "edge" is an artifact, not a read on the market.
    n_games = int(form.get("n_games") or 0)
    avg_innings = float(form.get("avg_innings") or 0.0)
    if n_games < min_starter_games or avg_innings < min_starter_avg_innings:
        out.update(
            recommendation="no_data",
            reason=(
                f"insufficient starter sample (n_games={n_games}, ip/g={avg_innings:.1f}); "
                f"projection unreliable for non-starter/thin workloads"
            ),
            best_edge=None,
            best_ev=None,
        )
        return out

    lam = _project_for_market(market_key, form, opponent_k_rate)
    if lam is None:
        out.update(recommendation="no_data", reason=f"no form rate for {market_key}", best_edge=None, best_ev=None)
        return out
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
    reason = (
        f"{side} edge {edge:+.1%} vs market; EV {ev:+.1%}"
        if rec != "no_play"
        else f"edge {edge:+.1%} below {min_edge:.0%} bar or EV<=0"
    )
    # Projection-side consistency gate (added 29 Jun after Kumar Rocker Over 4.5 staked while
    # the model's own projection, 4.15 Ks, sat UNDER the line). When the market prices a side
    # cheaper than the model's already-sub-50% probability for it, edge_over >= edge_under can
    # select the side the point projection contradicts -- a positive-EV longshot that loses
    # more often than it wins. Refuse to stake a side our own expected-Ks read points away
    # from: an Over needs lam > line, an Under needs lam < line. Exactly on the line is not a
    # contradiction (no side is favored), so it is left alone.
    if rec != "no_play":
        contradicts_projection = (side == "over" and lam < line) or (
            side == "under" and lam > line
        )
        if contradicts_projection:
            rec = "no_play"
            reason = (
                f"{side} contradicts projection (expected {lam:.2f} {spec['label']} vs line {line:g}); "
                f"+EV only from price, suppressed"
            )
    # Implausible-edge cap: a double-digit edge in a liquid prop market is the projection
    # being wrong, not the book. Suppress rather than stake until the model is calibrated.
    if rec != "no_play" and edge > max_plausible_edge:
        rec = "no_play"
        reason = (
            f"edge {edge:+.1%} exceeds plausible cap {max_plausible_edge:.0%} — "
            f"likely projection error, suppressed"
        )
    out.update(
        expected_ks=round(lam, 2),
        model_p_over=round(p_over, 4),
        market_p_over=round(mkt_over, 4),
        best_side=side,
        best_edge=round(edge, 4),
        best_ev=round(ev, 4),
        recommendation=rec,
        reason=reason,
    )
    return out


def evaluate_strikeout_props(
    props: list[dict],
    form_lookup,
    opp_k_lookup=None,
    dispersion: float = PROP_KS_DISPERSION,
    min_edge: float = PROP_MIN_EDGE,
    min_starter_games: int = PROP_MIN_STARTER_GAMES,
    min_starter_avg_innings: float = PROP_MIN_STARTER_AVG_INNINGS,
    max_plausible_edge: float = PROP_MAX_PLAUSIBLE_EDGE,
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
        scored.append(score_strikeout_prop(
            row, form, opp_k, dispersion, min_edge,
            min_starter_games, min_starter_avg_innings, max_plausible_edge,
        ))
    return scored
