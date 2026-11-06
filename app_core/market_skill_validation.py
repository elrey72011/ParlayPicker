"""Market-skill plan methodology (D19 option B, owner-chosen 2026-11-06).

The v1 template's absolute Brier <= 0.24 and log loss <= 0.68 cannot be met by a spread
or total market: each side is priced near 50%, so even a model that wins 55-56% against
the spread scores about 0.246 / 0.685 (``docs/roadmap/D19-PLAN-THRESHOLDS-2026-11-06.md``).
A plan frozen with ``market_skill_methodology`` is scored instead against the market:

* on the same games, the model's Brier and log loss must be **no worse than the
  de-vigged entry price** (the same book's two sides at the same line and time);
* at least ``min_market_reference_coverage`` of the scored games must have that
  two-sided reference, so the comparison is not cherry-picked;
* the **bets** the model would make (either side, when its expected value at the entry
  price is at least ``min_edge``) must number at least ``min_bets``, return paper ROI
  >= 0, and beat certified closes on average (mean CLV >= ``min_mean_clv``, with a close
  for at least ``min_bet_close_coverage`` of them).

The v1 paper-ROI rule scores every game's first prediction, which for the research cycle
is always the home side or the Over: betting one side of every game at -110 loses the vig
whatever the model knows. So for these plans ROI and CLV are scored on the bets instead.

Everything else in the v1 template still applies unchanged (calibration error, coverage,
sample sizes, verified entry prices, certified-close coverage). The
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
            "bet_rule": "EITHER_SIDE_EV_AT_ENTRY_PRICE_AT_LEAST_MIN_EDGE", "min_edge": 0.02,
            "min_bets": 50, "min_bet_close_coverage": 0.8, "paper_roi_scope": "BETS", "min_bet_roi": 0.0,
            "absolute_bounds": "COIN_FLIP_SANITY_FLOOR", "decision_record": "D19 option B (owner 2026-11-06)"}


def template(base):
    """The v1 template with the market-skill gates: coin-flip absolute bounds plus ``KEY``."""
    spec = dict(base)
    spec["probability_thresholds"] = dict(base["probability_thresholds"], **COIN_FLIP)
    spec[KEY] = methodology()
    return spec


QUOTE_COLUMNS = "quote_id,event_id,market_family,selection,line,sportsbook,quote_timestamp,decimal_odds"


def opposite(db, quote):
    """The same book's other side at the same line and timestamp, or None."""
    line = quote["line"] if quote["market_family"] == "TOTAL" else -quote["line"]
    rows = db.execute(f"""SELECT {QUOTE_COLUMNS} FROM prospective_quote WHERE event_id=? AND market_family=?
        AND sportsbook=? AND quote_timestamp=? AND selection<>? AND line=? AND quote_verified=1""",
                      (quote["event_id"], quote["market_family"], quote["sportsbook"],
                       quote["quote_timestamp"], quote["selection"], line)).fetchall()
    return dict(zip(QUOTE_COLUMNS.split(","), rows[0])) if len(rows) == 1 else None


def market_probability(db, quote):
    """No-vig probability for ``quote``'s side from the same book's other side at the same line and time."""
    other = opposite(db, quote)
    if other is None:
        return None
    mine, theirs = 1 / quote["decimal_odds"], 1 / other["decimal_odds"]
    return mine / (mine + theirs)


def _scores(pairs):
    clip = 1e-15
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    log_loss = -sum(y * math.log(max(p, clip)) + (1 - y) * math.log(max(1 - p, clip)) for p, y in pairs) / len(pairs)
    return brier, log_loss


def skill(db, scored, *, as_of=None, rules=None):
    """Model vs market, and the model's bets, on the scored games. ``scored``: [(p, y, quote)].

    ``p`` is the decided probability for ``quote``'s side and ``y`` its WIN (1) / LOSS (0).
    """
    from app_core.prospective_evidence import _comparable_close
    rules = rules or methodology()
    paired, returns, clvs = [], [], []
    for p, y, quote in scored:
        other = opposite(db, quote)
        if other is None:
            continue
        mine, theirs = 1 / quote["decimal_odds"], 1 / other["decimal_odds"]
        paired.append((p, mine / (mine + theirs), y))
        sides = [(p * quote["decimal_odds"] - 1, quote, y), ((1 - p) * other["decimal_odds"] - 1, other, 1 - y)]
        ev, chosen, won = max(sides, key=lambda side: side[0])
        if ev < rules["min_edge"]:
            continue
        returns.append(chosen["decimal_odds"] - 1 if won else -1.0)
        close = _comparable_close(db, chosen, as_of) if as_of is not None else None
        if close is not None:
            clvs.append(chosen["decimal_odds"] / close["decimal_odds"] - 1)
    out = {"scored": len(scored), "with_market_reference": len(paired),
           "market_reference_coverage": len(paired) / len(scored) if scored else 0.0,
           "model_brier": None, "market_brier": None, "model_log_loss": None, "market_log_loss": None,
           "bets": len(returns), "bet_roi": sum(returns) / len(returns) if returns else None,
           "bets_with_close": len(clvs), "bet_close_coverage": len(clvs) / len(returns) if returns else 0.0,
           "bet_mean_clv": sum(clvs) / len(clvs) if clvs else None}
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
    if result["bets"] < rules["min_bets"]:
        found.append(f"{label}_NEED_{rules['min_bets'] - result['bets']}_MORE_BETS")
    if result["bet_roi"] is None or result["bet_roi"] < rules["min_bet_roi"]:
        found.append(f"{label}_BET_ROI_BELOW_MINIMUM")
    if result["bet_close_coverage"] < rules["min_bet_close_coverage"]:
        found.append(f"{label}_BET_CLOSE_COVERAGE_TOO_LOW")
    if result["bet_mean_clv"] is None or result["bet_mean_clv"] < rules["min_mean_clv"]:
        found.append(f"{label}_BET_MEAN_CLV_BELOW_MINIMUM")
    return found
