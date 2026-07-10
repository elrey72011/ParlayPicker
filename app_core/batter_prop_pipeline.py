Exit code: 0
Wall time: 1 seconds
Output:
"""Conservative batter hits and total-bases prop scoring."""
from __future__ import annotations

from app_core.prop_model import strikeout_pick_probability

BATTER_PROP_SPECS = {
    "batter_hits": {
        "label": "Hits", "result_key": "hits", "rate_key": "hits_per_game", "dispersion": 1.10,
    },
    "batter_total_bases": {
        "label": "TB", "result_key": "total_bases", "rate_key": "total_bases_per_game", "dispersion": 1.45,
    },
}
BATTER_DEFAULT_ENABLED_MARKETS = tuple(BATTER_PROP_SPECS)
BATTER_MIN_GAMES = 20
BATTER_MIN_AVG_PA = 3.0
BATTER_MIN_EDGE = 0.04
BATTER_MAX_PLAUSIBLE_EDGE = 0.10


def _implied(odds: float) -> float:
    value = float(odds)
    return 100.0 / (value + 100.0) if value > 0 else (-value) / ((-value) + 100.0)


def _decimal(odds: float) -> float:
    value = float(odds)
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / -value)


def score_batter_prop(
    prop_row: dict,
    form: dict | None,
    *,
    min_edge: float = BATTER_MIN_EDGE,
    min_games: int = BATTER_MIN_GAMES,
    min_avg_pa: float = BATTER_MIN_AVG_PA,
    max_plausible_edge: float = BATTER_MAX_PLAUSIBLE_EDGE,
    enabled_markets: tuple[str, ...] | set[str] | None = None,
) -> dict:
    """Score one batter prop with strict sample, price, and plausibility guards."""
    out = dict(prop_row)
    market_key = str(prop_row.get("market_key") or "")
    spec = BATTER_PROP_SPECS.get(market_key)
    if spec is None:
        out.update(recommendation="no_play", reason="unsupported batter market")
        return out
    if enabled_markets is not None and market_key not in set(enabled_markets):
        out.update(recommendation="no_play", reason=f"{market_key} disabled pending calibration")
        return out
    if not form:
        out.update(recommendation="no_data", reason="no batter form", best_edge=None, best_ev=None)
        return out
    n_games = int(form.get("n_games") or 0)
    avg_pa = float(form.get("avg_plate_appearances") or 0.0)
    if n_games < int(min_games) or avg_pa < float(min_avg_pa):
        out.update(
            recommendation="no_data",
            reason=f"insufficient batter sample (games={n_games}, PA/g={avg_pa:.1f})",
            best_edge=None,
            best_ev=None,
        )
        return out

    expected = float(form.get(spec["rate_key"]) or 0.0)
    if expected <= 0:
        out.update(recommendation="no_data", reason="missing batter rate", best_edge=None, best_ev=None)
        return out
    line = float(prop_row["line"])
    over_odds, under_odds = float(prop_row["over_odds"]), float(prop_row["under_odds"])
    p_over = strikeout_pick_probability(expected, line, "over", float(spec["dispersion"]))
    p_under = 1.0 - p_over
    io, iu = _implied(over_odds), _implied(under_odds)
    mkt_over = io / (io + iu)
    mkt_under = 1.0 - mkt_over
    candidates = [
        ("over", p_over, p_over - mkt_over, over_odds),
        ("under", p_under, p_under - mkt_under, under_odds),
    ]
    side, p_side, edge, odds = max(candidates, key=lambda item: item[2])
    ev = p_side * _decimal(odds) - 1.0
    recommendation = side if edge >= float(min_edge) and ev > 0 else "no_play"
    reason = f"{side} edge {edge:+.1%}; EV {ev:+.1%}"
    if recommendation != "no_play" and edge > float(max_plausible_edge):
        recommendation = "no_play"
        reason = f"edge {edge:+.1%} exceeds plausible cap {max_plausible_edge:.0%}"

    result_key = spec["result_key"]
    out.update(
        expected_stat=result_key,
        expected_count=round(expected, 2),
        **{f"expected_{result_key}": round(expected, 2)},
        model_p_over=round(p_over, 4),
        market_p_over=round(mkt_over, 4),
        best_side=side,
        best_edge=round(edge, 4),
        best_ev=round(ev, 4),
        recommendation=recommendation,
        reason=reason,
    )
    return out


def evaluate_batter_props(props: list[dict], form_lookup, **score_kwargs) -> list[dict]:
    return [score_batter_prop(row, form_lookup(row.get("batter") or row.get("player")), **score_kwargs) for row in props]

