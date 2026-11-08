"""Pinnacle's no-vig price as a registered model (C5; D4 option A; D19 option B).

Team form alone does not beat the line (NCAAF D18, NCAAB line screen), and the Pinnacle
value board needs a plan to qualify. This puts Pinnacle's price through the same chain
as any model: register it, calibrate it, bind it to a reviewed plan (``market_skill``
methodology: no worse than the US book's own no-vig price, and positive ROI and CLV on
the bets), then the gate, the owner's activation and Gemini.

* **Model** (``register``): ``prospective_model`` with an artifact of kind
  ``KIND``. It has no training games (it is a price, not a fit), so it is never the
  research cycle's pinned candidate (``prospective_calibration.candidate_model`` needs
  120 training games).
* **Probability** (``fair``): for a US-book quote, Pinnacle's no-vig probability for the
  same side at the **same home-perspective line**, from Pinnacle's two sides in the **same
  capture response** as the US quote (``capture_time``), received no later than the
  prediction time. Pairing by timestamp could mix two captures a few minutes apart (decision
  capture runs every 15 minutes) while the price moved between them. No line translation:
  a game without Pinnacle at that line gets no prediction.
* **Calibration** (``fit_calibration``): Platt, fitted from games settled after the model
  was registered, at the research cycle's reference quote (each game's first canonical US
  quote; ``settle_reference_games`` settles them, since early in a season the research
  cycle has no model and settles nothing), using stored quotes only. It records **no predictions**, so it never enters,
  or blocks, a plan bound to another model.
* **Predictions** (``predict``): only through ``prospective_calibration._predict`` when a
  bound plan names this model (``predict_bound``, decision capture), so a market holds
  one plan's predictions at a time.

Research only until a plan binds it and the owner activates the market; no stake.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from app_core import prospective_evidence as evidence
from app_core.market_skill_validation import EARLIEST_SHARED_LINE, FIRST_EDGE_CAPTURE

KIND = "pinnacle_no_vig_v1"
FEATURE_VERSION = "pinnacle-no-vig-v1"
REFERENCE_BOOK = "pinnacle"
FAMILIES = {"NBA": ("SPREAD", "TOTAL"), "NCAAB": ("SPREAD", "TOTAL")}


def _t(value):
    return evidence._time(value, "timestamp")


def artifact(sport, family):
    return {"kind": KIND, "sport": sport, "market_family": family, "reference_book": REFERENCE_BOOK,
            "line_rule": "SAME_HOME_PERSPECTIVE_LINE", "pairing": "SAME_CAPTURE_RESPONSE",
            "probability": "NO_VIG_TWO_SIDED", "training": "NONE_PRICE_NOT_FIT"}


def is_pinnacle(model):
    import json
    try:
        return (json.loads(model["payload"]).get("model_artifact") or {}).get("kind") == KIND
    except (TypeError, ValueError, KeyError):
        return False


def registered(path, sport, family):
    """The registered Pinnacle model with this code's exact definition (``artifact``), or None."""
    digest = evidence._sha(evidence._json(artifact(sport, family)))
    found = [m for m in evidence.read_records(path, "prospective_model", sport=sport, market_family=family)
             if is_pinnacle(m) and m["artifact_hash"] == digest]
    return min(found, key=lambda m: (m["available_at"], m["model_id"])) if found else None


def register(path, sport, family, *, now=None, commit="UNAVAILABLE"):
    """Register the Pinnacle model once per market. Returns the model record."""
    if family not in FAMILIES.get(sport, ()):
        raise ValueError("Pinnacle model is for NBA/NCAAB spreads and totals")
    existing = registered(path, sport, family)
    if existing is not None:
        return existing
    now = now or datetime.now(timezone.utc)
    body = artifact(sport, family)
    digest = evidence._sha(evidence._json(body))
    model_id = f"pinnacle-model:{sport}:{family}:{digest[:16]}"
    evidence.insert_model(path, {
        "model_id": model_id, "sport": sport, "market_family": family, "model_version": KIND,
        "training_start": (now - timedelta(seconds=1)).isoformat(), "training_cutoff": now.isoformat(),
        "training_observation_count": 0, "independent_event_count": 0, "feature_version": FEATURE_VERSION,
        "training_code_commit": commit, "created_at": now.isoformat(), "available_at": now.isoformat(),
        "artifact_hash": digest, "model_artifact": body, "training_result_ids": []})
    return evidence.load_record(path, "prospective_model", model_id)


def _home_line(event, quote):
    if quote["market_family"] == "TOTAL" or quote["selection"] == event["home_team"]:
        return round(float(quote["line"]), 2)
    return round(-float(quote["line"]), 2)


def _canonical(event, family):
    return event["home_team"] if family != "TOTAL" else "Over"


def capture_time(event, quote):
    """When the response holding ``quote`` was received: its own stored response
    (``prospective_evidence.quote_source``, later captures) or the event's first capture."""
    raw = evidence._source_json(quote.get("raw_source"))
    if isinstance(raw, dict) and set(raw) == evidence.QUOTE_SOURCE_KEYS:
        return _t(raw["response_received_at"])
    return _t(event["observed_at"])


def pairs(event, quotes, family):
    """[(capture_time, home_line, p_canonical)] for Pinnacle's two-sided quotes on this event."""
    grouped = {}
    for q in quotes:
        if (q["event_id"] != event["event_id"] or q["sportsbook"] != REFERENCE_BOOK or q["market_family"] != family
                or not q["quote_verified"] or q["line"] is None):
            continue
        key = (capture_time(event, q), q["quote_timestamp"], _home_line(event, q))
        grouped.setdefault(key, {})[q["selection"]] = q
    out = []
    canonical = _canonical(event, family)
    for (captured, _, line), sides in grouped.items():
        if len(sides) != 2 or canonical not in sides:
            continue
        mine = 1 / sides[canonical]["decimal_odds"]
        other = 1 / next(q for s, q in sides.items() if s != canonical)["decimal_odds"]
        out.append((captured, line, mine / (mine + other)))
    return out


def fair(event, quote, quotes, *, now):
    """Pinnacle's no-vig probability for ``quote``'s side at its exact line, same capture, or None."""
    line, captured = _home_line(event, quote), capture_time(event, quote)
    if captured > now:
        return None
    found = [p for when, home_line, p in pairs(event, quotes, quote["market_family"])
             if when == captured and home_line == line]
    if len(found) != 1:
        return None
    return found[0] if quote["selection"] == _canonical(event, quote["market_family"]) else 1 - found[0]


def _other_side(quote, quotes):
    line = quote["line"] if quote["market_family"] == "TOTAL" else -quote["line"]
    found = [q for q in quotes if q["event_id"] == quote["event_id"] and q["sportsbook"] == quote["sportsbook"]
             and q["quote_timestamp"] == quote["quote_timestamp"] and q["selection"] != quote["selection"]
             and q["line"] == line and q["quote_verified"]]
    return found[0] if len(found) == 1 else None


# The books a qualified NBA/NCAAB pick can be staked at (core.sport_market_gate: Novig, or DraftKings,
# FanDuel and BetMGM with their capture replayed). A plan must be measured where it can bet: a
# price at another book would pass validation and then be blocked at the stake (owner 2026-11-07).
PLAN_BOOKS = frozenset({"Novig", "DraftKings", "FanDuel", "BetMGM"})


def _plan_book(quote, books):
    if books is None:
        return True
    from app_core.public_quote_policy import canonical_book_label
    return canonical_book_label(quote["sportsbook"]) in books


def plan_quotes(events, quotes, now, *, books=None, entry_rule=None, min_edge=None, adjust=None):
    """The plan's entry quote per game (owner 2026-11-07): the best US price at Pinnacle's line.

    A fixed rule, using only prices known before tip-off (never results):

    * the canonical side (home team, or Over), so each game is settled once per side
      (``prospective_result`` keeps one result per game, market and side);
    * the **earliest capture** in which Pinnacle and at least one US book post the same line;
    * among that capture's US books at that line, the one whose better side has the
      highest EV against Pinnacle's no-vig price (ties: higher price, then book name).
      Both of the book's sides must be posted, since the market-skill gate compares the
      model with that book's own two-sided price;
    * quoted before the scheduled start, verified, half or whole line, never Pinnacle itself;
    * at one of ``books`` (canonical labels). A plan drafted by the Bound plan workflow freezes
      ``PLAN_BOOKS`` (the books the stake can use) as its ``entry_books``; its predictions, window
      rate and replay estimate use them. ``None`` means every US book.

    ``entry_rule`` ``FIRST_EDGE_CAPTURE`` (owner 2026-11-08, frozen with a plan) enters instead at
    the **first capture whose best side clears ``min_edge``**, and at the earliest shared-line
    capture while none has. US books match Pinnacle's line at the same price when both first
    post, and lag when Pinnacle moves, so the earliest capture almost never clears a 1-2% edge.
    Each capture is judged only on prices known at that capture. ``adjust`` maps Pinnacle's
    no-vig probability to the plan's (its calibration), so the entry and the bet rule agree.

    The first canonical US quote (``prospective_research_models._reference_quotes``) has
    Pinnacle at its line only about 22% of the time (NCAAB, 2026-11-07) and is rarely a good
    price, which left too few qualifying bets for a plan to pass.
    """
    from app_core.research_api_budget import REFERENCE_BOOKS
    by_event = {}
    for q in quotes:
        event = events.get(q["event_id"])
        if (event is None or q["sportsbook"] in REFERENCE_BOOKS or not _plan_book(q, books)
                or not q["quote_verified"] or q["line"] is None
                or q["selection"] != _canonical(event, q["market_family"])
                or abs(2 * q["line"] - round(2 * q["line"])) > 1e-8
                or _t(q["quote_timestamp"]) > now or _t(q["quote_timestamp"]) >= _t(event["scheduled_start"])):
            continue
        by_event.setdefault(q["event_id"], []).append(q)
    chosen = {}
    for event_id, candidates in by_event.items():
        event = events[event_id]
        family = candidates[0]["market_family"]
        priced = {(when, line): p for when, line, p in pairs(event, quotes, family)}
        best = {}
        for q in candidates:
            captured = capture_time(event, q)
            p = priced.get((captured, _home_line(event, q)))
            other = _other_side(q, quotes) if p is not None and captured <= now else None
            if other is None:
                continue
            if adjust is not None:
                p = adjust(p)
            ev = max(p * q["decimal_odds"] - 1, (1 - p) * other["decimal_odds"] - 1)
            key = (ev, q["decimal_odds"], q["sportsbook"])
            if captured not in best or key > best[captured][0]:
                best[captured] = (key, q)
        if best:
            when = min(best)
            if entry_rule == FIRST_EDGE_CAPTURE:
                clearing = [c for c, (key, _) in best.items() if key[0] >= min_edge]
                when = min(clearing) if clearing else when
            chosen[event_id] = best[when][1]
    return chosen


def plan_games_per_day(path, sport, family, *, now, lookback_days=14, books=None):
    """Games per day the plan can predict (``plan_quotes``), over the days actually observed.

    Sizing a plan's windows (``prospective_calibration.window_days``) by every game over a
    fixed 14 days undercounted a new season (NCAAB capture began about 2026-11-04) and
    counted games a Pinnacle plan cannot predict.
    """
    from app_core import prospective_research_models as research
    events, quotes, _ = research._read_scope(path, sport, family)
    start = now - timedelta(days=lookback_days)
    started = [e for e in events.values() if start <= _t(e["scheduled_start"]) < now]
    if not started:
        return 0.0
    covered = plan_quotes({e["event_id"]: e for e in started}, quotes, now, books=books)
    first = min(_t(e["scheduled_start"]) for e in started)
    days = max(1.0, (now - first).total_seconds() / 86400)
    return len(covered) / days


REPLAY_EDGES = (0.01, 0.02)


def replay_estimate(path, sport, family, *, now, lookback_days=14, edges=REPLAY_EDGES,
                    min_sample=200, min_bets=50, books=None, calibration=None):
    """How fast a Pinnacle plan would fill, replayed over games already captured (read-only).

    For games started in the last ``lookback_days`` (counted over the days actually
    observed), applies the plan's entry rule (``plan_quotes``) and counts, per day:
    games the plan could predict, and games whose better side's EV against Pinnacle's
    no-vig price clears each ``edge`` (the market-skill bet rule). Then the days each
    window would need for ``min_sample`` games and ``min_bets`` bets.

    ``by_edge`` is the earliest-capture rule on Pinnacle's raw no-vig price. ``variants``
    adds each entry rule (``EARLIEST_SHARED_LINE``, ``FIRST_EDGE_CAPTURE``) on the raw price
    and, given the plan's ``calibration`` artifact, on the calibrated probability the plan
    actually bets with (owner 2026-11-08). Both rules predict every plan game, so the game
    rate is the same; only the bets differ.

    It never reads results or scores, so the bet threshold is chosen before seeing which
    bets won.
    """
    from app_core import prospective_research_models as research
    events, quotes, _ = research._read_scope(path, sport, family)
    start = now - timedelta(days=lookback_days)
    started = {e["event_id"]: e for e in events.values() if start <= _t(e["scheduled_start"]) < now}
    if not started:
        return {"sport": sport, "market_family": family, "games_started": 0, "status": "NO_GAMES"}
    chosen = plan_quotes(started, quotes, now, books=books)
    first = min(_t(e["scheduled_start"]) for e in started.values())
    days = max(1.0, (now - first).total_seconds() / 86400)
    games_per_day = len(chosen) / days
    out = {"sport": sport, "market_family": family, "status": "ESTIMATED", "days_observed": round(days, 1),
           "games_started": len(started), "plan_games": len(chosen),
           "plan_share": round(len(chosen) / len(started), 3), "plan_games_per_day": round(games_per_day, 2),
           "days_for_min_sample": (round(min_sample / games_per_day, 1) if games_per_day else None),
           "min_sample": min_sample, "min_bets": min_bets}

    def row(bets):
        per_day = bets / days
        days_bets = round(min_bets / per_day, 1) if per_day else None
        window = (max(out["days_for_min_sample"], days_bets)
                  if out["days_for_min_sample"] is not None and days_bets is not None else None)
        return {"bets": bets, "bets_per_day": round(per_day, 2), "days_for_min_bets": days_bets,
                "days_per_window": window}

    def bets(picked, edge, adjust):
        count = 0
        for event_id, quote in picked.items():
            p = fair(started[event_id], quote, quotes, now=now)
            other = _other_side(quote, quotes)
            if p is None or other is None:
                continue
            p = adjust(p) if adjust else p
            count += max(p * quote["decimal_odds"] - 1, (1 - p) * other["decimal_odds"] - 1) >= edge
        return count

    out["by_edge"] = {f"{edge:.0%}": row(bets(chosen, edge, None)) for edge in edges}
    probabilities = [("PINNACLE_NO_VIG", None)]
    if calibration:
        from app_core.prospective_calibration import calibrated
        probabilities.append(("CALIBRATED", lambda p: calibrated({"win": p, "push": 0.0, "loss": 1 - p},
                                                                 calibration)["win"]))
        out["calibration"] = {k: calibration.get(k) for k in ("a", "b", "decided")}
    out["variants"] = []
    for label, adjust in probabilities:
        for rule in (EARLIEST_SHARED_LINE, FIRST_EDGE_CAPTURE):
            by_edge = {}
            for edge in edges:
                picked = (chosen if rule == EARLIEST_SHARED_LINE else
                          plan_quotes(started, quotes, now, books=books, entry_rule=rule, min_edge=edge,
                                      adjust=adjust))
                by_edge[f"{edge:.0%}"] = row(bets(picked, edge, adjust))
            out["variants"].append({"entry_rule": rule, "probability": label, "by_edge": by_edge})
    return out


def _settled_at(result, quote):
    """True when ``result`` was settled at exactly ``quote`` (its stored settlement rule names it)."""
    rule = evidence._source_json(result.get("raw_source"))
    return isinstance(rule, dict) and rule.get("quote_id") == quote["quote_id"]


def settle_reference_games(path, sport, family, *, now=None):
    """Settle each finished game at the research cycle's reference quote, where Pinnacle priced that line.

    The research cycle settles only games it predicted, and early in a season it has no
    model yet. Settlement needs only a replay-verified final score and the verified quote
    (``prospective_evidence.insert_score_settlement``), so this settles the same quote the
    research cycle would (``_reference_quotes``: each game's first canonical US quote).
    When the research cycle later settles that quote, it gets this same result row.
    """
    from app_core import prospective_calibration as calibration
    from app_core import prospective_research_models as research
    now = now or datetime.now(timezone.utc)
    events, quotes, scores = research._read_scope(path, sport, family)
    latest = calibration._latest_scores(scores)
    settled = calibration._market_result(path, sport, family)
    report = {"settled": 0, "already_settled": 0, "no_pinnacle_at_line": 0, "awaiting_score": 0, "errors": 0}
    for event_id, quote in sorted(research._reference_quotes(events, quotes, now).items()):
        event = events[event_id]
        if (event_id, quote["selection"]) in settled:
            report["already_settled"] += 1
            continue
        if fair(event, quote, quotes, now=_t(event["scheduled_start"])) is None:
            report["no_pinnacle_at_line"] += 1
            continue
        score = latest.get(event_id)
        if score is None:
            report["awaiting_score"] += 1
            continue
        try:
            evidence.insert_score_settlement(path, score_result_id=score["result_id"], quote_id=quote["quote_id"])
            report["settled"] += 1
        except (ValueError, evidence.EvidenceConflict):
            report["errors"] += 1
    return report


def settled_games(path, sport, family, *, since=None, now=None):
    """Decided games settled at the research cycle's reference quote, where Pinnacle priced
    that line: [{event_id, result_id, y, pinnacle, quote, other}].

    ``other`` is the same US book's opposite side at that time (None when not stored).
    """
    from app_core import prospective_calibration as calibration
    from app_core import prospective_research_models as research
    now = now or datetime.now(timezone.utc)
    events, quotes, _ = research._read_scope(path, sport, family)
    results = calibration._market_result(path, sport, family)
    out = []
    for event_id, quote in sorted(research._reference_quotes(events, quotes, now).items()):
        result, event = results.get((event_id, quote["selection"])), events[event_id]
        if (result is None or result["outcome"] not in ("WIN", "LOSS") or not _settled_at(result, quote)
                or _t(result["available_at"]) > now or (since is not None and _t(result["available_at"]) < since)):
            continue
        p = fair(event, quote, quotes, now=_t(event["scheduled_start"]))
        if p is None:
            continue
        out.append({"event_id": event_id, "result_id": result["result_id"], "y": int(result["outcome"] == "WIN"),
                    "pinnacle": p, "quote": quote, "other": _other_side(quote, quotes)})
    return out


def preview(path, sport, family, *, now=None, min_edge=0.02):
    """What a market-skill plan bound to the Pinnacle model would see so far (research only).

    * readiness: model registered, calibration fitted or games still needed;
    * coverage: of the games the research cycle settled, the share where Pinnacle had a price
      at the settled quote's exact line (a game without one gets no Pinnacle prediction);
    * replay on all settled games with a Pinnacle price (uncalibrated): Pinnacle's Brier and log loss
      vs the US book's own no-vig price, and the bets (either side, EV >= ``min_edge`` against
      Pinnacle) with their record and ROI. Not the plan's verdict: no certified closes or windows.
    """
    from app_core import prospective_calibration as calibration
    now = now or datetime.now(timezone.utc)
    model = registered(path, sport, family)
    out = {"sport": sport, "market_family": family, "model_id": model["model_id"] if model else None,
           "calibration": None, "settled_games": 0, "with_pinnacle_at_line": 0, "coverage": None}
    if model is not None:
        cals = [c for c in evidence.read_records(path, "prospective_calibration", sport=sport, market_family=family)
                if c["model_id"] == model["model_id"]]
        since = settled_games(path, sport, family, since=_t(model["available_at"]), now=now)
        out["calibration"] = ({"status": "FITTED", "calibration_id": max(cals, key=lambda c: c["available_at"])["calibration_id"]}
                              if cals else {"status": "WAITING", "decided": len(since),
                                            "required": calibration.MIN_CALIBRATION_DECIDED})
    from app_core import prospective_research_models as research
    events, quotes, _ = research._read_scope(path, sport, family)
    results = calibration._market_result(path, sport, family)
    decided = [e for e, q in research._reference_quotes(events, quotes, now).items()
               if (e, q["selection"]) in results and results[(e, q["selection"])]["outcome"] in ("WIN", "LOSS")
               and _settled_at(results[(e, q["selection"])], q)]
    games = settled_games(path, sport, family, now=now)
    out.update(settled_games=len(decided), with_pinnacle_at_line=len(games),
               coverage=len(games) / len(decided) if decided else None)
    paired = [g for g in games if g["other"] is not None]
    if paired:
        def scores(rows):
            brier = sum((p - y) ** 2 for p, y in rows) / len(rows)
            loss = -sum(y * math.log(max(p, 1e-15)) + (1 - y) * math.log(max(1 - p, 1e-15)) for p, y in rows) / len(rows)
            return brier, loss
        us = [((1 / g["quote"]["decimal_odds"]) / (1 / g["quote"]["decimal_odds"] + 1 / g["other"]["decimal_odds"]), g["y"])
              for g in paired]
        out["pinnacle_brier"], out["pinnacle_log_loss"] = scores([(g["pinnacle"], g["y"]) for g in paired])
        out["us_brier"], out["us_log_loss"] = scores(us)
    bets = []
    for g in paired:
        sides = [(g["pinnacle"] * g["quote"]["decimal_odds"] - 1, g["quote"]["decimal_odds"], g["y"]),
                 ((1 - g["pinnacle"]) * g["other"]["decimal_odds"] - 1, g["other"]["decimal_odds"], 1 - g["y"])]
        ev, decimal, won = max(sides)
        if ev >= min_edge:
            bets.append(decimal - 1 if won else -1.0)
    out.update(compared=len(paired), bets=len(bets), bet_wins=sum(1 for r in bets if r > 0),
               bet_roi=sum(bets) / len(bets) if bets else None)
    return out


def fit_calibration(path, sport, family, model, *, now=None, min_decided=None):
    """A Platt calibration for the Pinnacle model from settled games, recording no predictions."""
    from app_core import prospective_calibration as calibration
    from app_core import prospective_research_models as research
    now = now or datetime.now(timezone.utc)
    min_decided = calibration.MIN_CALIBRATION_DECIDED if min_decided is None else min_decided
    existing = calibration.current_calibrations(evidence.read_records(path, "prospective_calibration", sport=sport,
                                                                      market_family=family), model["model_id"])
    if existing:
        return {"status": "PRESENT", "calibration_id": max(existing, key=lambda c: c["available_at"])["calibration_id"]}
    fit_start = _t(model["available_at"])
    settled = settled_games(path, sport, family, since=fit_start, now=now)
    pairs_ = [(g["pinnacle"], g["y"]) for g in settled]
    result_ids = [g["result_id"] for g in settled]
    if len(pairs_) < min_decided:
        return {"status": "INSUFFICIENT_EVIDENCE", "calibration_id": None, "decided": len(pairs_),
                "required_decided": min_decided}
    a, b, fallback = calibration.guarded_fit(pairs_)
    fit_end = max(_t(evidence.load_record(path, "prospective_result", rid)["available_at"]) for rid in result_ids)
    body = {"method": calibration.CALIBRATION_VERSION, "a": a, "b": b, "decided": len(pairs_),
            "model_id": model["model_id"], "sport": sport, "market_family": family,
            "fit_start": fit_start.isoformat(), "fit_end": fit_end.isoformat(), "fit_result_ids": result_ids,
            "applies_to": "decided_win_probability", "source": "STORED_PINNACLE_QUOTES_AT_SETTLED_QUOTE",
            **({"fallback": fallback} if fallback else {})}
    digest = research._digest(body)
    calibration_id = f"platt-calibration:{sport}:{family}:{digest[:24]}"
    created = max(now, fit_end).isoformat()
    evidence.insert_calibration(path, {
        "calibration_id": calibration_id, "sport": sport, "market_family": family,
        "calibration_version": calibration.CALIBRATION_VERSION, "model_id": model["model_id"],
        "fit_start": fit_start.isoformat(), "fit_end": fit_end.isoformat(), "method": calibration.CALIBRATION_VERSION,
        "created_at": created, "available_at": created, "artifact_hash": digest,
        "calibration_artifact": body, "fit_result_ids": result_ids})
    return {"status": "FITTED", "calibration_id": calibration_id, "decided": len(pairs_), "a": a, "b": b,
            **({"fallback": fallback["reason"]} if fallback else {})}


def predict(path, sport, family, model, calibration, *, now, policy_version, tag, select=None):
    """Bound predictions from Pinnacle's price (called by ``prospective_calibration._predict``)."""
    import json
    from app_core import prospective_calibration as cal
    from app_core import prospective_research_models as research
    cal_artifact = json.loads(calibration["payload"]).get("calibration_artifact") if calibration else None
    calibration_id = calibration["calibration_id"] if calibration else None
    events, quotes, _ = research._read_scope(path, sport, family)
    all_quotes = evidence.read_records(path, "prospective_quote", sport=sport, market_family=family)
    done = {(r["event_id"], r["quote_id"]) for r in evidence.read_records(
        path, "prospective_prediction", sport=sport, market_family=family)
        if r["model_id"] == model["model_id"] and r["calibration_id"] == calibration_id}
    created, no_reference = 0, 0
    commit = research.source_commit()
    for _, quote in sorted((select or plan_quotes)(events, quotes, now).items()):
        event = events[quote["event_id"]]
        if now >= _t(event["scheduled_start"]) or (event["event_id"], quote["quote_id"]) in done:
            continue
        p = fair(event, quote, all_quotes, now=now)
        if p is None:
            no_reference += 1
            continue
        raw = {"win": p, "push": 0.0, "loss": 1 - p}
        probabilities = cal.calibrated(raw, cal_artifact) if cal_artifact else raw
        n = max(1, cal_artifact["decided"] if cal_artifact else 1)
        win = probabilities["win"]
        conservative = max(0.0, win - 1.645 * math.sqrt(win * (1 - win) / n))
        snapshot = {"protocol": FEATURE_VERSION, "sport": sport, "market_family": family,
                    "event_id": event["event_id"], "quote_id": quote["quote_id"], "as_of": now.isoformat(),
                    "pinnacle_probability": p, "exact_line": quote["line"], "american_odds": quote["american_odds"],
                    "decimal_odds": quote["decimal_odds"], "sportsbook": quote["sportsbook"],
                    "provider_event_hash": event["source_hash"], "provider_quote_hash": quote["source_hash"],
                    "raw_probabilities": raw, "calibration_id": calibration_id, "policy_version": policy_version}
        snapshot_id = evidence._sha(evidence._json(snapshot))
        record = {
            "observation_id": f"{tag.lower()}-prediction:" + research._digest([model["model_id"], calibration_id,
                                                                              snapshot_id])[:32],
            "event_id": event["event_id"], "sport": sport, "market_family": family, "selection": quote["selection"],
            "quote_id": quote["quote_id"], "model_id": model["model_id"], "model_version": model["model_version"],
            "feature_version": FEATURE_VERSION, "feature_snapshot_id": snapshot_id, "feature_snapshot": snapshot,
            "feature_frozen_at": now.isoformat(), "prediction_timestamp": now.isoformat(),
            "mean_probability": win, "conservative_probability": conservative,
            "push_probability": probabilities["push"], "loss_probability": probabilities["loss"],
            "probability_semantics": "WIN_PUSH_LOSS", "policy_version": policy_version,
            "evidence_snapshot_id": snapshot_id,
            "evidence_hash": research._digest([event["source_hash"], quote["source_hash"], p]),
            "runtime_hash": evidence._sha(evidence._json(artifact(sport, family))), "source_commit": commit,
            "uncertainty": {"method": "one_sided_normal_approximation_v1", "calibration_id": calibration_id,
                            "sample": n, "research_only": True}}
        if calibration is not None:
            record.update(calibration_id=calibration_id, calibration_version=calibration["calibration_version"])
        evidence.insert_prediction(path, record)
        created += 1
    return {"predictions": created, "skipped_no_pinnacle_at_line": no_reference, "blocker": None}
