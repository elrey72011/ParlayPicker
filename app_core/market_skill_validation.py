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

``market-skill-v2`` (owner D26, 2026-11-09; ``methodology_v2``) is for plans frozen from then on.
A simulation of the v1 gates (``scripts/plan_gate_power.py``) passes a true 2% edge about 1 time
in 4: Brier and log loss "no worse than the market" on 200 games is close to a coin flip, and
ROI >= 0 on 50 bets has a standard error near 13 points. Closing-line value is the one precise
measure, so v2 leads with it. In each window:

* at least ``min_bets`` bets, and their mean CLV at least ``min_clv_t`` standard errors above zero;
* a certified close for at least ``min_bet_close_coverage`` of the bets;
* bet ROI at or above a loss floor (``min_bet_roi``, negative: it catches a broken plan, not noise);
* the market reference for at least ``min_market_reference_coverage`` of scored games.

Brier and log loss against the market are reported, never blocking. The absolute probability
bounds are a coin flip plus room for sampling noise (``V2_THRESHOLDS``). The holdout bets' CLV
lower 95% bound is the staking terms' input, so a plan that passes can stake.

A v2 plan may pool its bets with named companion plans (owner D27: the Pinnacle/Novig lag only;
NBA and NCAAB, and NHL from 2026-11-09). A pool is one kind of market across sports: every scope's
total, or every scope's side market (spread, or the NHL puck line), never both. Every companion is frozen with the same rules and windows.
The bet gates above are then scored on all the pool's bets, each scope's own results are reported,
and each scope still needs ``min_scope_bets`` of its own. v1 plans never pool.
"""
from __future__ import annotations

import math

KEY = "market_skill_methodology"
VERSION = "market-skill-v1"
COIN_FLIP = {"max_brier": 0.25, "max_log_loss": round(math.log(2), 6)}


VERSION_V2 = "market-skill-v2"
# v2 absolute bounds: a coin flip (0.25, ln 2) plus room for sampling noise. A model that agrees
# with a market near 50% scores about 0.249 with a standard error near 0.002 on 200 games, so the
# exact coin-flip bound would fail a sound plan about 1 time in 4.
V2_THRESHOLDS = {"max_brier": 0.26, "max_log_loss": 0.715, "max_calibration_error": 0.15}
DEFAULT_MIN_BETS = 75
ALLOWED_MIN_BETS = (50, 60, 75, 100)        # chosen when the plan is drafted; locks at the freeze
MIN_CLV_T = 2.0
V2_MIN_BET_ROI = -0.25                      # a loss floor: 75 bets' ROI has a standard error near 11 points
MIN_SCOPE_BETS = 10
POOL_SPORTS = ("NBA", "NCAAB", "NHL")       # owner D27: the Pinnacle/Novig lag only; NHL added 2026-11-09
DEFAULT_POOL_SPORTS = ("NBA", "NCAAB")
SIDE_FAMILY = {"NBA": "SPREAD", "NCAAB": "SPREAD", "NHL": "PUCK_LINE"}
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
# A first-edge plan may also be frozen to enter only this close to the start (owner 2026-11-11).
# A bet needs a certified close at its own book and line for the 80% close gate, and a line taken
# hours early is often one the book later leaves: in the week of 2026-11-09, 64% of NCAAB spread
# bets at Novig had such a close at any entry time and 80% within the last 60 minutes. Recorded
# only when chosen, so plans frozen without it keep their hash.
ALLOWED_ENTRY_WITHIN_MINUTES = (60,)


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


def _valid_pool(scopes):
    """True for two or more scopes, one per sport, all totals or all side markets."""
    try:
        parts = [scope.split(":") for scope in scopes]
        sports = [p[0] for p in parts]
        if (len(parts) < 2 or any(len(p) != 2 for p in parts) or len(set(sports)) != len(sports)
                or any(s not in POOL_SPORTS for s in sports)):
            return False
        return all(f == "TOTAL" for _, f in parts) or all(f == SIDE_FAMILY[s] for s, f in parts)
    except (AttributeError, TypeError):
        return False


def pool_scopes(sport, family, sports=None):
    """The scopes a pooled v2 plan for ``sport`` / ``family`` counts bets across (owner D27).

    ``sports`` are the pool's sports (default NBA and NCAAB) and must include ``sport``; each brings
    its total, or its side market (spread, NHL puck line), to match ``family``.
    """
    chosen = tuple(dict.fromkeys(sports or DEFAULT_POOL_SPORTS))
    if sport not in chosen or family not in ("TOTAL", SIDE_FAMILY.get(sport)):
        raise ValueError("name the pool's sports, this one included; a pool is side markets or totals of NBA, NCAAB and NHL")
    scopes = sorted(f"{s}:{'TOTAL' if family == 'TOTAL' else SIDE_FAMILY.get(s)}" for s in chosen)
    if not _valid_pool(scopes):
        raise ValueError("a pool is two or more of NBA, NCAAB and NHL")
    return scopes


def methodology_v2(min_edge=DEFAULT_MIN_EDGE, entry_books=None, entry_rule=None, min_bets=DEFAULT_MIN_BETS,
                   max_entry_overround=None, pool=None, entry_within_minutes=None):
    """The ``market-skill-v2`` rules (owner D26). ``pool``: the scopes whose bets count together
    (``pool_scopes``; owner D27), or None for a plan that validates alone."""
    if min_edge not in ALLOWED_MIN_EDGES:
        raise ValueError(f"min_edge must be one of {ALLOWED_MIN_EDGES}")
    if entry_rule is not None and entry_rule not in ENTRY_RULES:
        raise ValueError(f"entry_rule must be one of {ENTRY_RULES}")
    if min_bets not in ALLOWED_MIN_BETS:
        raise ValueError(f"min_bets must be one of {ALLOWED_MIN_BETS}")
    rules = {"version": VERSION_V2, "primary_gate": "BET_MEAN_CLV_STANDARD_ERRORS_ABOVE_ZERO",
             "min_clv_t": MIN_CLV_T, "min_bets": min_bets, "min_bet_close_coverage": 0.8,
             "min_bet_roi": V2_MIN_BET_ROI, "paper_roi_scope": "BETS",
             "probability_gate": "MODEL_VS_MARKET_REPORTED_NOT_BLOCKING",
             "market_reference": "SAME_BOOK_OPPOSITE_SIDE_SAME_LINE_SAME_TIMESTAMP",
             "min_market_reference_coverage": 0.9,
             "bet_rule": "EITHER_SIDE_EV_AT_ENTRY_PRICE_AT_LEAST_MIN_EDGE", "min_edge": min_edge,
             "absolute_bounds": "COIN_FLIP_PLUS_SAMPLING_NOISE", "window_sizing": "EXPECTED_BETS",
             "staking_statistic": "HOLDOUT_BET_CLV_LOWER_95",
             "other_model_predictions": "REPORTED_NOT_BLOCKING", "push_rule": PUSH_ZERO_RETURN,
             "decision_record": "D26 (owner 2026-11-09)"}
    if entry_books:
        rules["entry_books"] = sorted(entry_books)
    if entry_rule == FIRST_EDGE_CAPTURE:
        rules["entry_rule"] = FIRST_EDGE_CAPTURE
    if max_entry_overround is not None:
        if not 1.0 < max_entry_overround <= 1.5:
            raise ValueError("max_entry_overround must be above 1 and at most 1.5")
        rules["max_entry_overround"] = max_entry_overround   # owner D29: junk prices never enter
    if entry_within_minutes is not None:
        if entry_within_minutes not in ALLOWED_ENTRY_WITHIN_MINUTES:
            raise ValueError(f"entry_within_minutes must be one of {ALLOWED_ENTRY_WITHIN_MINUTES}")
        if entry_rule != FIRST_EDGE_CAPTURE:
            raise ValueError("entry_within_minutes needs the FIRST_EDGE_CAPTURE entry rule")
        rules["entry_within_minutes"] = entry_within_minutes   # owner 2026-11-11: late entries only
    if pool is not None:
        scopes = sorted(pool)
        if not _valid_pool(scopes):
            raise ValueError("a pool is totals, or side markets, across two or more of NBA, NCAAB and NHL")
        rules["pool"] = {"scopes": scopes, "min_scope_bets": MIN_SCOPE_BETS,
                         "gate": "BET_GATES_ON_ALL_POOL_BETS_EACH_SCOPE_REPORTED", "decision_record": "D27 (owner 2026-11-09)"}
    return rules


def template_v2(base, **kwargs):
    """The v1 template with the ``market-skill-v2`` gates."""
    spec = dict(base)
    rules = methodology_v2(**kwargs)
    spec["probability_thresholds"] = dict(base["probability_thresholds"], **V2_THRESHOLDS)
    # Close coverage is gated on the bets (``min_bet_close_coverage``); over every game it is reported.
    spec["clv_policy"] = dict(base["clv_policy"], min_comparable_close_coverage=0.0)
    # Windows are sized by bets. Each scope still needs this many settled games of its own.
    sample = rules["pool"]["min_scope_bets"] if "pool" in rules else rules["min_bets"]
    spec["minimum_independent_sample"] = spec["minimum_effective_sample"] = sample
    if "pool" in rules:
        spec["model_scope"] = dict(base["model_scope"],
                                   cross_sport_or_market_pooling="BET_GATES_POOLED_ACROSS_NAMED_SCOPES_D27")
    spec[KEY] = rules
    return spec


def is_v2(method):
    return (method.get(KEY) or {}).get("version") == VERSION_V2


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
        if rules.get("entry_within_minutes") and start - capture_time(event, entry) > timedelta(
                minutes=rules["entry_within_minutes"]):
            continue  # the plan froze late entries only: this capture is too early to be its bet
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


def skill(db, scored, *, as_of=None, rules=None, plan=None, collect=None):
    """Model vs market, and the model's bets, on the scored games. ``scored``: [(p, y, quote)].

    ``collect``, when a dict, receives the bets' ``returns`` and ``clvs`` (for a pool).

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
    if rules.get("version") == VERSION_V2:
        out["bet_clv_t"] = _t_statistic(clvs)
    if collect is not None:
        collect.update(returns=returns, clvs=clvs)
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


def _t_statistic(values):
    """Mean over its standard error; None with fewer than 2 values or no spread."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return mean / (sd / math.sqrt(len(values))) if sd > 0 else None


def _bet_summary(returns, clvs):
    return {"bets": len(returns), "bet_roi": sum(returns) / len(returns) if returns else None,
            "bets_with_close": len(clvs), "bet_close_coverage": len(clvs) / len(returns) if returns else 0.0,
            "bet_mean_clv": sum(clvs) / len(clvs) if clvs else None, "bet_clv_t": _t_statistic(clvs),
            "bet_clv_lower_95": _lower_95(clvs)}


WINDOW_COLUMNS = ("validation_start", "validation_end", "holdout_start", "holdout_end")


def pool_results(db, plan, method, as_of, cohort, own):
    """{window: pooled bet summary} for a pooled v2 plan, or None when the plan does not pool.

    ``own``: {window: {"returns", "clvs"}} already collected for ``plan``. Each companion is the
    latest plan of its scope and must be frozen with the same rules and windows and bound to a
    model and calibration; otherwise the pool is ``complete: False`` and blocks. ``cohort`` is
    ``prospective_evidence._cohort``.
    """
    from app_core import prospective_evidence as pe
    rules = method[KEY]
    pool = rules.get("pool") if rules.get("version") == VERSION_V2 else None
    if not pool:
        return None
    mine = f"{plan['sport']}:{plan['market_family']}"
    collected, problems = {mine: own}, []
    if mine not in pool["scopes"]:
        problems.append("OWN_SCOPE_NOT_IN_POOL")
    for scope in pool["scopes"]:
        if scope == mine:
            continue
        sport, family = scope.split(":")
        row = db.execute("""SELECT validation_plan_id FROM prospective_validation_plan WHERE sport=? AND market_family=?
            ORDER BY version DESC LIMIT 1""", (sport, family)).fetchone()
        other = pe._stored(db, "prospective_validation_plan", "validation_plan_id", row[0]) if row else None
        if other is None:
            problems.append(f"{scope}_NO_PLAN")
            continue
        if (json_rules(other) != rules or any(other[c] != plan[c] for c in WINDOW_COLUMNS)
                or other["model_id"] is None or other["calibration_id"] is None or not pe.plan_integrity(other)):
            problems.append(f"{scope}_PLAN_DIFFERS")
            continue
        collected[scope] = {}
        for window in ("validation", "holdout"):
            pairs, bets = [], {}
            cohort(db, other, window, as_of, pairs)
            skill(db, pairs, as_of=as_of, rules=rules, plan=other, collect=bets)
            collected[scope][window] = bets
    out = {}
    for window in ("validation", "holdout"):
        scopes = {scope: _bet_summary(data[window]["returns"], data[window]["clvs"])
                  for scope, data in sorted(collected.items())}
        out[window] = dict(_bet_summary([r for d in collected.values() for r in d[window]["returns"]],
                                        [c for d in collected.values() for c in d[window]["clvs"]]),
                           complete=not problems, problems=problems, scopes=scopes)
    return out


def json_rules(plan):
    import json
    try:
        return json.loads(plan["payload"]).get(KEY)
    except (TypeError, ValueError, KeyError):
        return None


def _v2_blockers(label, result, rules, pool):
    found = []
    if result["market_reference_coverage"] < rules["min_market_reference_coverage"]:
        found.append(f"{label}_MARKET_REFERENCE_COVERAGE_TOO_LOW")
    bets = result
    if "pool" in rules:
        if pool is None or not pool["complete"]:
            return found + [f"{label}_POOL_INCOMPLETE"]
        if result["bets"] < rules["pool"]["min_scope_bets"]:
            found.append(f"{label}_NEED_{rules['pool']['min_scope_bets'] - result['bets']}_MORE_BETS_IN_THIS_SCOPE")
        bets = pool
    if bets["bets"] < rules["min_bets"]:
        found.append(f"{label}_NEED_{rules['min_bets'] - bets['bets']}_MORE_BETS")
    if bets["bet_roi"] is None or bets["bet_roi"] < rules["min_bet_roi"]:
        found.append(f"{label}_BET_ROI_BELOW_LOSS_FLOOR")
    if bets["bet_close_coverage"] < rules["min_bet_close_coverage"]:
        found.append(f"{label}_BET_CLOSE_COVERAGE_TOO_LOW")
    if bets["bet_clv_t"] is None or bets["bet_clv_t"] < rules["min_clv_t"]:
        found.append(f"{label}_BET_CLV_NOT_{rules['min_clv_t']:g}_STANDARD_ERRORS_ABOVE_ZERO")
    return found


def blockers(label, cohort, result, method, pool=None):
    rules = method[KEY]
    if rules.get("version") == VERSION_V2:
        return _v2_blockers(label, result, rules, pool)
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
