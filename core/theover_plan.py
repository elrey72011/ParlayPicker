"""TheOver's college totals plan: the tally against the rules the owner approved on 2026-11-19.

Read-only, and it grants nothing: no rating, pick, stake or approval reads it. The rules are in
``docs/roadmap/theover-college-totals-plan-v1.json`` and are never edited (a changed rule is a new
file); every report carries that file's SHA-256, so a change cannot pass unnoticed. The plan itself is
``docs/roadmap/THEOVER-COLLEGE-TOTALS-PLAN-2026-11-19.md``.

**A bet.** For each game, the last run the app saved before the game started. If that run holds
TheOver's model pick for the game's total and the pick fed the blend
(``theover_model_totals.backed`` and ``in_blend``), TheOver's side is the bet: one unit at the board's
line and its best price at that line (``best_line_price``, else the pick's own price). A game whose last
run holds no such pick has no bet, even if an earlier run did.

**Its close.** The latest verified closing observation for the same provider event, the same side and
the book the pick was bound to (``quote_bookmaker``). The closing-capture job records a close against
one saved run's candidate; a close read for an earlier run's candidate is the same market at the same
moment, so it is matched by event, side and book, and the move is counted from the bet's own line:
points the total moved toward the pick (``core.clv.line_clv``).

**Windows.** A game belongs to the Eastern date of its start, or to the night before when it starts
before 5 AM Eastern. A window that ends short of its minimum bets is extended a night at a time, and
the holdout then begins the night after. A window is judged only when all its nights are past and every
bet in it has a result.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

RULES_PATH = Path(__file__).resolve().parents[1] / "docs" / "roadmap" / "theover-college-totals-plan-v1.json"
EASTERN = ZoneInfo("America/New_York")
NIGHT_ENDS_HOUR = 5                 # a start before 5 AM Eastern belongs to the night before
WINDOWS = ("validation", "holdout")
SCHEMA = "theover-college-totals-plan-report-v1"


def load_rules(path=RULES_PATH):
    """``(rules, sha256 of the file's bytes)``."""
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _decimal(american):
    american = _number(american)
    if american is None or abs(american) < 100:
        return None
    return 1 + (american / 100 if american > 0 else 100 / -american)


def slate_date(start):
    """The Eastern date a game's night is counted under."""
    return (start.astimezone(EASTERN) - timedelta(hours=NIGHT_ENDS_HOUR)).date()


def bets(rows, sports=("NCAAB", "NCAAF")):
    """One bet a game from saved candidate rows (``core.research_performance.saved_rows``), oldest game first.

    Two passes, so the store's every row is never held as one table: first each game's last run saved
    before its start (by any of its candidates), then that run's totals rows for the game.
    """
    import pandas as pd
    from app_core import theover_model_totals as theover
    from app_core.public_quote_policy import canonical_book_label
    from core.wager_decisions import aware
    last, held = {}, {}
    for row in rows:
        sport = str(row.get("sport") or row.get("league") or "").upper()
        start, saved = aware(row.get("game_start_utc")), aware(row.get("capture_recorded_at"))
        if sport not in sports or start is None or saved is None or not saved < start or not row.get("matchup_id"):
            continue
        key, run = (sport, row["matchup_id"]), (saved, str(row.get("snapshot_id")))
        if key not in last or run > last[key]:
            last[key], held[key] = run, []
        if run == last[key] and str(row.get("market_type")) in theover.OTHER_SIDE:
            held[key].append(dict(row, _sport=sport, _start=start))
    totals = [row for rows_of_game in held.values() for row in rows_of_game]
    if not totals:
        return []
    frame = pd.DataFrame(totals)
    frame = frame[(theover.backed(frame) & theover.in_blend(frame)).to_numpy()]
    out = []
    for _, side in frame.groupby(["_sport", "matchup_id"], sort=False):
        if len(side) != 1:
            continue        # TheOver's side is one row; anything else is not a pick this tally can name
        row = side.iloc[0]
        start = row["_start"]
        out.append({
            "sport": row["_sport"], "game": row["matchup_id"], "start": start.isoformat(), "slate": slate_date(start).isoformat(),
            "snapshot_id": row.get("snapshot_id"), "candidate_id": row.get("candidate_id"),
            "market_type": str(row.get("market_type")), "pick": row.get("best_pick"),
            "line": _number(row.get("total_line")), "sheet_line": _number(row.get("theover_pick_line")),
            "decimal": _decimal(row.get("best_line_price")) or _decimal(row.get("odds_american")),
            "book": canonical_book_label(row.get("quote_bookmaker")) if isinstance(row.get("quote_bookmaker"), str) and row.get("quote_bookmaker") else None,
            "event": row.get("provider_event_id") if isinstance(row.get("provider_event_id"), str) else None,
            "rating": _number(row.get("calibrated_probability")),
            "outcome": row.get("candidate_outcome") if row.get("candidate_outcome") in ("WIN", "LOSS", "PUSH") else None})
    return sorted(out, key=lambda bet: (bet["start"], str(bet["game"])))


def with_closes(plan_bets, observations):
    """Each bet with ``close_line`` and ``move`` (points toward the pick), or ``None`` where no close was read."""
    from app_core.public_quote_policy import canonical_book_label
    from core.clv import line_clv
    latest = {}
    for observation in observations:
        quote = observation.get("quote") or {}
        if observation.get("quote_verified") is not True:
            continue
        key = (quote.get("provider_event_id"), quote.get("market_type"), canonical_book_label(quote.get("sportsbook")))
        if key not in latest or str(observation.get("closing_capture_at")) > str(latest[key].get("closing_capture_at")):
            latest[key] = observation
    out = []
    for bet in plan_bets:
        close = latest.get((bet["event"], bet["market_type"], bet["book"])) if bet["event"] and bet["book"] else None
        line = _number((close or {}).get("quote", {}).get("line"))
        move = line_clv(bet["market_type"], bet["line"], line) if close else None
        out.append(dict(bet, close_line=line if move is not None else None, move=move))
    return out


def _wilson(wins, decided, z=1.96):
    if not decided:
        return None, None
    p = wins / decided
    denominator = 1 + z * z / decided
    centre = (p + z * z / (2 * decided)) / denominator
    half = z * math.sqrt(p * (1 - p) / decided + z * z / (4 * decided * decided)) / denominator
    return centre - half, centre + half


def summarize(plan_bets):
    """Counts and measures over a set of bets. Nothing here decides anything."""
    wins = sum(bet["outcome"] == "WIN" for bet in plan_bets)
    losses = sum(bet["outcome"] == "LOSS" for bet in plan_bets)
    pushes = sum(bet["outcome"] == "PUSH" for bet in plan_bets)
    priced = [bet for bet in plan_bets if bet["outcome"] and bet["decimal"]]
    units = sum(bet["decimal"] - 1 if bet["outcome"] == "WIN" else -1 if bet["outcome"] == "LOSS" else 0.0 for bet in priced)
    moves = [bet["move"] for bet in plan_bets if bet.get("move") is not None]
    mean = sum(moves) / len(moves) if moves else None
    error = (math.sqrt(sum((move - mean) ** 2 for move in moves) / (len(moves) - 1)) / math.sqrt(len(moves))
             if len(moves) > 1 else None)
    rated = [(bet["rating"], bet["outcome"] == "WIN") for bet in plan_bets
             if bet["outcome"] in ("WIN", "LOSS") and bet["rating"] is not None and 0 < bet["rating"] < 1]
    low, high = _wilson(wins, wins + losses)
    decimals = [bet["decimal"] for bet in priced]
    return {
        "bets": len(plan_bets), "wins": wins, "losses": losses, "pushes": pushes,
        "waiting_for_a_result": len(plan_bets) - wins - losses - pushes,
        "hit_rate": wins / (wins + losses) if wins + losses else None, "hit_rate_95_low": low, "hit_rate_95_high": high,
        "break_even_hit_rate": len(decimals) / sum(decimals) if decimals else None,
        "units": units if priced else None, "units_per_bet": units / len(priced) if priced else None,
        "closes": len(moves), "close_share": len(moves) / len(plan_bets) if plan_bets else None,
        "mean_move": mean, "move_standard_error": error,
        "move_t": (mean / error if error else (0.0 if mean == 0 else None)) if mean is not None and len(moves) > 1 else None,
        "brier": sum((p - won) ** 2 for p, won in rated) / len(rated) if rated else None,
        "calibration_error": (abs(sum(p for p, _ in rated) / len(rated) - sum(won for _, won in rated) / len(rated))
                              if rated else None)}


def failures(summary, bar):
    """The rules of the bar a complete window does not meet, in the rules file's own words."""
    checks = (("min_bets", summary["bets"], lambda value, limit: value >= limit),
              ("min_share_of_bets_with_a_close", summary["close_share"], lambda value, limit: value >= limit),
              ("min_standard_errors_of_mean_line_move_above_zero", summary["move_t"], lambda value, limit: value >= limit),
              ("min_units_per_bet", summary["units_per_bet"], lambda value, limit: value >= limit),
              ("max_brier", summary["brier"], lambda value, limit: value <= limit),
              ("max_calibration_error", summary["calibration_error"], lambda value, limit: value <= limit))
    return [name for name, value, met in checks if value is None or not met(value, bar[name])]


def _window(name, first, nominal_last, plan_bets, minimum, today):
    """One window's nights, bets and state. Extended a night at a time while it is short and a past night remains."""
    last = nominal_last

    def inside(until):
        return [bet for bet in plan_bets if first.isoformat() <= bet["slate"] <= until.isoformat()]
    while len(inside(last)) < minimum and last < today - timedelta(days=1):
        last += timedelta(days=1)
    mine = inside(last)
    summary = summarize(mine)
    if today < first:
        state = "not started"
    elif last >= today:
        state = "collecting"
    elif summary["waiting_for_a_result"]:
        state = "waiting for results"
    elif summary["bets"] < minimum:
        state = "extending: short of its bets"
    else:
        state = "complete"
    return {"window": name, "first_night": first.isoformat(), "last_night": last.isoformat(),
            "extended_nights": (last - nominal_last).days, "state": state, **summary}


def report(rows, observations, rules, fingerprint, now=None):
    """The whole tally as one JSON-ready dict."""
    now = now or datetime.now(timezone.utc)
    today = slate_date(now)
    everything = with_closes(bets(rows), observations)
    college = [bet for bet in everything if bet["sport"] == rules["scope"]["sport"]]
    bar, dates = rules["bar_in_each_window"], rules["windows"]["slate_dates_eastern"]
    windows, first = {}, date.fromisoformat(dates["validation"][0])
    for name in WINDOWS:
        nominal_first, nominal_last = (date.fromisoformat(day) for day in dates[name])
        first = max(first, nominal_first)
        last = first + (nominal_last - nominal_first)
        window = _window(name, first, last, college, bar["min_bets"], today)
        window["fails"] = failures(window, bar) if window["state"] == "complete" else None
        windows[name] = window
        first = date.fromisoformat(window["last_night"]) + timedelta(days=1)
    judged = [windows[name] for name in WINDOWS if windows[name]["state"] == "complete"]
    if any(window["fails"] for window in judged):
        verdict = "FAILED"
    elif len(judged) == len(WINDOWS):
        verdict = "PASSED"
    else:
        verdict = "COLLECTING"
    start = dates["validation"][0]
    sides = {side: summarize([bet for bet in college if bet["slate"] >= start and bet["market_type"] == market])
             for side, market in (("unders", "total_under"), ("overs", "total_over"))}
    gap = {}
    for label, test in (("board_line_worse_for_the_pick", lambda d: d < -0.25), ("same_line", lambda d: abs(d) <= 0.25),
                        ("board_line_better_for_the_pick", lambda d: d > 0.25)):
        chosen = [bet for bet in college if bet["slate"] >= start and bet["line"] is not None and bet["sheet_line"] is not None
                  and test((bet["line"] - bet["sheet_line"]) * (1 if bet["market_type"] == "total_under" else -1))]
        gap[label] = summarize(chosen)
    return {"schema": SCHEMA, "plan_id": rules["plan_id"], "rules_sha256": fingerprint, "as_of_night": today.isoformat(),
            "verdict": verdict, "windows": windows, "bar": bar,
            "before_the_windows": summarize([bet for bet in college if bet["slate"] < start]),
            "by_side": sides, "line_against_theovers_own": gap,
            "reported_without_a_bar": {
                entry.split()[0]: summarize([bet for bet in everything if bet["sport"] == entry.split()[0] and bet["slate"] >= start])
                for entry in rules["scope"].get("also_reported_without_a_bar", [])},
            "changes_nothing": True}


def build(database, now=None, rules_path=RULES_PATH):
    """The tally from an evidence database (verified saved runs, their results and the closes read)."""
    from app_core.activation_closing import observations
    from core.research_performance import saved_rows
    rules, fingerprint = load_rules(rules_path)
    return report(saved_rows(database), observations(database), rules, fingerprint, now=now)


def markdown(tally):
    def pct(value):
        return "" if value is None else f"{value:.1%}"

    def signed(value, digits=2):
        return "" if value is None else f"{value:+.{digits}f}"

    def record(s):
        text = f"{s['wins']}-{s['losses']}" + (f"-{s['pushes']}" if s["pushes"] else "")
        return text + (f" ({pct(s['hit_rate'])}, 95% {pct(s['hit_rate_95_low'])} to {pct(s['hit_rate_95_high'])})"
                       if s["hit_rate"] is not None else "")

    def closes(s):
        return f"{s['closes']} of {s['bets']}" + (f" ({pct(s['close_share'])})" if s["bets"] else "")

    def move(s):
        if s["mean_move"] is None:
            return "no close yet"
        return f"{signed(s['mean_move'])} pts" + (f" (t {s['move_t']:.1f})" if s["move_t"] is not None else "")
    bar = tally["bar"]
    lines = ["## TheOver's college totals plan", "",
             f"Plan `{tally['plan_id']}`, rules fingerprint `{tally['rules_sha256'][:16]}`. As of the night of "
             f"{tally['as_of_night']}: **{tally['verdict']}**. A tally only: it changes no rating, pick or stake.", "",
             f"The bar in each window: at least {bar['min_bets']} bets, a close for {bar['min_share_of_bets_with_a_close']:.0%} of them, "
             f"the total's average move toward the pick at least {bar['min_standard_errors_of_mean_line_move_above_zero']:g} standard errors "
             f"above zero, units per bet no worse than {bar['min_units_per_bet']:+.2f}.", "",
             "| window | nights | state | bets | record | units | closes read | move toward the pick | fails |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in WINDOWS:
        w = tally["windows"][name]
        nights = f"{w['first_night']} to {w['last_night']}" + (f" (+{w['extended_nights']})" if w["extended_nights"] else "")
        fails = "" if w["fails"] is None else (", ".join(w["fails"]) or "none")
        lines.append(f"| {name} | {nights} | {w['state']} | {w['bets']} | {record(w)} | {signed(w['units'])} | {closes(w)} | {move(w)} | {fails} |")
    lines += ["", "Reported, never deciding:", ""]
    extra = [("before the windows", tally["before_the_windows"])]
    extra += [(side, summary) for side, summary in tally["by_side"].items()]
    extra += [(label.replace("_", " "), summary) for label, summary in tally["line_against_theovers_own"].items()]
    extra += [(f"{sport} totals (no bar)", summary) for sport, summary in tally["reported_without_a_bar"].items()]
    lines += ["| group | bets | record | units | closes read | move toward the pick |", "|---|---|---|---|---|---|"]
    lines += [f"| {label} | {s['bets']} | {record(s)} | {signed(s['units'])} | {closes(s)} | {move(s)} |" for label, s in extra]
    return "\n".join(lines) + "\n"
