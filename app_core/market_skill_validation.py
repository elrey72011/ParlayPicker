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

Predictions by other models or calibrations in the windows (the research cycle writes an
uncalibrated one for every game) are reported, not blocking: each game is scored from its
first prediction by the plan's exact model and calibration. Under v1 they block, so no v1/v2
bound plan can pass while the research cycle runs.

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


DEFAULT_MIN_EDGE = 0.02
# The bet threshold is chosen when the plan is drafted and frozen (owner 2026-11-07; the
# bound plan draft's replay estimate shows the bet rate at each). It locks at the freeze.
ALLOWED_MIN_EDGES = (0.01, 0.02)

# Where a Pinnacle plan enters each game (``pinnacle_model.plan_quotes``). Plans frozen without
# ``entry_rule`` use the earliest capture where a US book posts Pinnacle's line, and bet on that
# prediction. ``FIRST_EDGE_CAPTURE`` (owner 2026-11-08) also predicts at the first later capture
# whose best side clears ``min_edge``, and bets on the game's first prediction that clears it;
# the probability gate still scores each game's first prediction.
EARLIEST_SHARED_LINE = "EARLIEST_SHARED_LINE"
FIRST_EDGE_CAPTURE = "FIRST_EDGE_CAPTURE"
ENTRY_RULES = (EARLIEST_SHARED_LINE, FIRST_EDGE_CAPTURE)
# How a bet that pushes counts (owner 2026-11-09). Plans frozen without ``push_rule`` leave a push
# out of the bets entirely; ``PUSH_ZERO_RETURN`` counts it as a bet that returned its stake (0), so
# whole-number lines do not flatter ROI. Recorded only when chosen, so frozen plans keep their hash.
PUSH_ZERO_RETURN = "PUSH_COUNTS_AS_ZERO_RETURN_BET"
# A first-edge entry must be bettable when it is predicted (audit 2026-11-09): its capture must be
# at most this old at the prediction time. Price gaps after news last about 10 to 30 minutes, so a
# capture found to clear hours later is no longer a price anyone could take.
FIRST_EDGE_MAX_ENTRY_AGE_MINUTES = 15


def methodology(min_edge=DEFAULT_MIN_EDGE, entry_books=None, entry_rule=None, push_rule=None):
    """The market-skill rules. ``entry_books`` (canonical book labels), when given, is the set of
    books the plan may enter at, frozen with the plan (a Pinnacle plan: the books a pick can be
    staked at, ``pinnacle_model.PLAN_BOOKS``). ``entry_rule`` is recorded only when it is not the
    default, so earlier plans keep their exact rules and hash."""
    if min_edge not in ALLOWED_MIN_EDGES:
        raise ValueError(f"min_edge must be one of {ALLOWED_MIN_EDGES}")
    if entry_rule is not None and entry_rule not in ENTRY_RULES:
        raise ValueError(f"entry_rule must be one of {ENTRY_RULES}")
    rules = {"version": VERSION, "probability_gate": "NO_WORSE_THAN_DEVIGGED_ENTRY_PRICE",
            "market_reference": "SAME_BOOK_OPPOSITE_SIDE_SAME_LINE_SAME_TIMESTAMP",
            "min_market_reference_coverage": 0.9, "min_mean_clv": 0.0,
            "bet_rule": "EITHER_SIDE_EV_AT_ENTRY_PRICE_AT_LEAST_MIN_EDGE", "min_edge": min_edge,
            "min_bets": 50, "min_bet_close_coverage": 0.8, "paper_roi_scope": "BETS", "min_bet_roi": 0.0,
            "absolute_bounds": "COIN_FLIP_SANITY_FLOOR",
            "other_model_predictions": "REPORTED_NOT_BLOCKING", "decision_record": "D19 option B (owner 2026-11-06)"}
    if entry_books:
        rules["entry_books"] = sorted(entry_books)
    if entry_rule == FIRST_EDGE_CAPTURE:
        rules["entry_rule"] = FIRST_EDGE_CAPTURE
    if push_rule is not None:
        if push_rule != PUSH_ZERO_RETURN:
            raise ValueError(f"push_rule must be {PUSH_ZERO_RETURN}")
        rules["push_rule"] = PUSH_ZERO_RETURN
    return rules


def template(base, min_edge=DEFAULT_MIN_EDGE, entry_books=None, entry_rule=None, push_rule=None):
    """The v1 template with the market-skill gates: coin-flip absolute bounds plus ``KEY``."""
    spec = dict(base)
    spec["probability_thresholds"] = dict(base["probability_thresholds"], **COIN_FLIP)
    spec[KEY] = methodology(min_edge, entry_books, entry_rule, push_rule)
    return spec


QUOTE_COLUMNS = "quote_id,event_id,market_family,selection,line,sportsbook,quote_timestamp,decimal_odds"


def response_key(raw):
    """When the provider response holding a quote was received, for quotes stored with their own
    response (``prospective_evidence.quote_source``); None for a first-capture quote."""
    from app_core import prospective_evidence as pe
    value = pe._source_json(raw)
    return value.get("response_received_at") if isinstance(value, dict) and set(value) == pe.QUOTE_SOURCE_KEYS else None


def opposite(db, quote):
    """The same book's other side at the same line and timestamp, or None.

    Later captures store the full snapshot each time (``decision_capture.store_quotes``), so an
    unchanged price can be stored more than once. Then the other side is taken from the same
    response as ``quote``. With one candidate (every quote stored before 2026-11-09) nothing changes.
    """
    line = quote["line"] if quote["market_family"] == "TOTAL" else -quote["line"]
    rows = db.execute(f"""SELECT {QUOTE_COLUMNS},raw_source FROM prospective_quote WHERE event_id=? AND market_family=?
        AND sportsbook=? AND quote_timestamp=? AND selection<>? AND line=? AND quote_verified=1""",
                      (quote["event_id"], quote["market_family"], quote["sportsbook"],
                       quote["quote_timestamp"], quote["selection"], line)).fetchall()
    if len(rows) > 1:
        mine = db.execute("SELECT raw_source FROM prospective_quote WHERE quote_id=?", (quote.get("quote_id"),)).fetchone()
        key = response_key(mine[0]) if mine else None
        rows = [row for row in rows if response_key(row[-1]) == key]
    return dict(zip(QUOTE_COLUMNS.split(","), rows[0][:-1])) if len(rows) == 1 else None


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


def _best_side(p, y, quote, other):
    sides = [(p * quote["decimal_odds"] - 1, quote, y), ((1 - p) * other["decimal_odds"] - 1, other, 1 - y)]
    return max(sides, key=lambda side: side[0])


def first_edge_bet(db, plan, quote, rules, as_of):
    """``FIRST_EDGE_CAPTURE``: the game's first prediction by the plan's exact model and calibration
    whose better side clears ``min_edge``, as (ev, side quote, won), or None.

    Only pregame predictions at verified quotes captured before the prediction count, and only
    when the prediction's own line is decided (scored from the stored final score at that line).
    """
    from app_core import prospective_evidence as pe
    event = pe._event(db, quote["event_id"])
    start = pe._time(event["scheduled_start"], "scheduled_start")
    # Decision capture's predictions (``decision-prediction:``) depend on a switch not frozen with the
    # plan, so they never enter the bet stream (audit 2026-11-09).
    rows = db.execute("""SELECT observation_id FROM prospective_prediction WHERE event_id=? AND market_family=?
        AND model_id=? AND calibration_id=? AND observation_id NOT LIKE 'decision-prediction:%'
        ORDER BY prediction_timestamp, observation_id""",
                      (quote["event_id"], quote["market_family"], plan["model_id"], plan["calibration_id"])).fetchall()
    for row in rows:
        prediction = pe._stored(db, "prospective_prediction", "observation_id", row[0])
        made = pe._time(prediction["prediction_timestamp"], "prediction_timestamp")
        if made >= start or (as_of is not None and made > as_of) or not prediction["quote_id"]:
            continue
        entry = pe._stored(db, "prospective_quote", "quote_id", prediction["quote_id"])
        mean, loss = prediction["mean_probability"], prediction["loss_probability"]
        if (entry is None or not entry["quote_verified"] or entry["line"] is None or entry["decimal_odds"] is None
                or mean is None or loss is None or mean + loss <= 0
                or pe._time(entry["quote_timestamp"], "quote_timestamp") > made):
            continue
        from datetime import timedelta
        from app_core.pinnacle_model import capture_time
        if made - capture_time(event, entry) > timedelta(minutes=FIRST_EDGE_MAX_ENTRY_AGE_MINUTES):
            continue  # predicted from a capture too old to still be bettable
        other = opposite(db, entry)
        if other is None:
            continue
        p = mean / (mean + loss)
        ev, side, _ = _best_side(p, 1, entry, other)
        if ev < rules["min_edge"]:
            continue
        result = pe._latest_result(db, entry["event_id"], entry["market_family"], prediction["selection"],
                                   as_of if as_of is not None else pe._clock())
        if result is None:
            continue  # this side is not settled: a later prediction of the game may still be the bet
        outcome = pe._outcome_at_quote_line(db, result, entry, result["outcome"])
        if outcome == "PUSH" and rules.get("push_rule") == PUSH_ZERO_RETURN:
            return ev, side, None  # a bet that returned its stake
        if outcome not in ("WIN", "LOSS"):
            return None
        won = int(outcome == "WIN")
        return ev, side, won if side is entry else 1 - won
    return None


def skill(db, scored, *, as_of=None, rules=None, plan=None):
    """Model vs market, and the model's bets, on the scored games. ``scored``: [(p, y, quote)].

    ``p`` is the decided probability for ``quote``'s side and ``y`` its WIN (1) / LOSS (0).
    Under ``FIRST_EDGE_CAPTURE`` the bets come from ``first_edge_bet`` (``plan`` names the model
    and calibration); the probability gate still uses each game's first prediction.
    """
    from app_core.prospective_evidence import _comparable_close
    rules = rules or methodology()
    first_edge = rules.get("entry_rule") == FIRST_EDGE_CAPTURE and plan is not None
    paired, returns, clvs = [], [], []
    for p, y, quote in scored:
        other = opposite(db, quote)
        if other is None:
            continue
        mine, theirs = 1 / quote["decimal_odds"], 1 / other["decimal_odds"]
        if y is not None:  # a push (``push_rule``) is a bet, never a probability-gate outcome
            paired.append((p, mine / (mine + theirs), y))
        if first_edge:
            bet = first_edge_bet(db, plan, quote, rules, as_of)
            if bet is None:
                continue
            ev, chosen, won = bet
        else:
            ev, chosen, won = _best_side(p, 0 if y is None else y, quote, other)
            won = None if y is None else won
        if ev < rules["min_edge"]:
            continue
        returns.append(0.0 if won is None else chosen["decimal_odds"] - 1 if won else -1.0)
        close = _comparable_close(db, chosen, as_of) if as_of is not None else None
        if close is not None:
            clvs.append(chosen["decimal_odds"] / close["decimal_odds"] - 1)
    out = {"scored": len(scored), "with_market_reference": len(paired),
           "market_reference_coverage": len(paired) / len(scored) if scored else 0.0,
           "model_brier": None, "market_brier": None, "model_log_loss": None, "market_log_loss": None,
           "bets": len(returns), "bet_roi": sum(returns) / len(returns) if returns else None,
           "bets_with_close": len(clvs), "bet_close_coverage": len(clvs) / len(returns) if returns else 0.0,
           "bet_mean_clv": sum(clvs) / len(clvs) if clvs else None,
           # Mean CLV minus 1.96 standard errors: the staking terms' prior-CLV bound (app_core.staking_terms).
           "bet_clv_lower_95": _lower_95(clvs)}
    if paired:
        out["model_brier"], out["model_log_loss"] = _scores([(p, y) for p, _, y in paired])
        out["market_brier"], out["market_log_loss"] = _scores([(m, y) for _, m, y in paired])
    return out


def _lower_95(values):
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return mean - 1.96 * sd / math.sqrt(len(values))


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
