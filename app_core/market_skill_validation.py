"""Market-skill plan methodology (D19 option B, owner-chosen 2026-11-06).

The v1 template's absolute Brier <= 0.24 and log loss <= 0.68 cannot be met by a spread
or total market: each side is priced near 50%, so even a model that wins 55-56% against
the spread scores about 0.246 / 0.685 (``docs/roadmap/D19-PLAN-THRESHOLDS-2026-11-06.md``).
A plan frozen with ``market_skill_methodology`` is scored instead against the market:

* on the same games, the model's Brier and log loss must be **no worse than the
  de-vigged entry price** (the same book's two sides at the same line and time);
* at least ``min_market_reference_coverage`` of the scored games must have that
  two-sided reference, so the comparison is not cherry-picked;
* mean CLV against certified closes must be at least ``min_mean_clv``.

Everything else in the v1 template still applies unchanged (calibration error, coverage,
sample sizes, verified entry prices, certified-close coverage, paper ROI >= 0). The
absolute bounds are set to a coin flip (0.25, ln 2) as a sanity floor. Plans frozen
without this key are evaluated exactly as before.
"""
from __future__ import annotations

import math

KEY = "market_skill_methodology"
VERSION = "market-skill-v1"
COIN_FLIP = {"max_brier": 0.25, "max_log_loss": round(math.log(2), 6)}


def methodology():
    return {"version": VERSION, "probability_gate": "NO_WORSE_THAN_DEVIGGED_ENTRY_PRICE",
            "market_reference": "SAME_BOOK_OPPOSITE_SIDE_SAME_LINE_SAME_TIMESTAMP",
            "min_market_reference_coverage": 0.9, "min_mean_clv": 0.0,
            "absolute_bounds": "COIN_FLIP_SANITY_FLOOR", "decision_record": "D19 option B (owner 2026-11-06)"}


def template(base):
    """The v1 template with the market-skill gates: coin-flip absolute bounds plus ``KEY``."""
    spec = dict(base)
    spec["probability_thresholds"] = dict(base["probability_thresholds"], **COIN_FLIP)
    spec[KEY] = methodology()
    return spec


def market_probability(db, quote):
    """No-vig probability for ``quote``'s side from the same book's other side at the same line and time."""
    line = quote["line"] if quote["market_family"] == "TOTAL" else -quote["line"]
    rows = db.execute("""SELECT decimal_odds FROM prospective_quote WHERE event_id=? AND market_family=?
        AND sportsbook=? AND quote_timestamp=? AND selection<>? AND line=? AND quote_verified=1""",
                      (quote["event_id"], quote["market_family"], quote["sportsbook"],
                       quote["quote_timestamp"], quote["selection"], line)).fetchall()
    if len(rows) != 1:
        return None
    mine, other = 1 / quote["decimal_odds"], 1 / rows[0][0]
    return mine / (mine + other)


def _scores(pairs):
    clip = 1e-15
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    log_loss = -sum(y * math.log(max(p, clip)) + (1 - y) * math.log(max(1 - p, clip)) for p, y in pairs) / len(pairs)
    return brier, log_loss


def skill(db, scored):
    """Model vs market on the scored games that have a two-sided reference. ``scored``: [(p, y, quote)]."""
    paired = []
    for p, y, quote in scored:
        m = market_probability(db, quote)
        if m is not None:
            paired.append((p, m, y))
    out = {"scored": len(scored), "with_market_reference": len(paired),
           "market_reference_coverage": len(paired) / len(scored) if scored else 0.0,
           "model_brier": None, "market_brier": None, "model_log_loss": None, "market_log_loss": None}
    if paired:
        out["model_brier"], out["model_log_loss"] = _scores([(p, y) for p, _, y in paired])
        out["market_brier"], out["market_log_loss"] = _scores([(m, y) for _, m, y in paired])
    return out


def blockers(label, cohort, result, method):
    rules = method[KEY]
    found = []
    if result["market_reference_coverage"] < rules["min_market_reference_coverage"]:
        found.append(f"{label}_MARKET_REFERENCE_COVERAGE_TOO_LOW")
    if result["model_brier"] is None or result["model_brier"] > result["market_brier"]:
        found.append(f"{label}_BRIER_WORSE_THAN_MARKET")
    if result["model_log_loss"] is None or result["model_log_loss"] > result["market_log_loss"]:
        found.append(f"{label}_LOG_LOSS_WORSE_THAN_MARKET")
    if cohort["average_valid_clv"] is None or cohort["average_valid_clv"] < rules["min_mean_clv"]:
        found.append(f"{label}_MEAN_CLV_BELOW_MINIMUM")
    return found
