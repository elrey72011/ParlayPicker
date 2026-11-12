"""Live price-gap candidates, research only (owner 2026-11-10).

A candidate is a stake book's price that is better than Pinnacle's no-vig fair price for the
same side at the same line, in the newest capture of a game that has not started. It is the
thing the Pinnacle replay counts after the fact, shown while it is live.

**It is not a bet with a proven edge.** On 2026-11-09 the replay found 22 such gaps and only 4
could be checked against the closing price. Nothing here is a wager approval: no stake, no
plan evaluation and no pick reads it. The app shows it marked as unproven.

The price watch computes the list after each capture (it holds the store) and uploads it as
one small object under ``PREFIX``, outside the evidence. Drive objects are create-only and a
prefix listing walks the whole folder, so the name is the capture's 5-minute slot and the app
asks for exact names, newest first (the same pattern as ``readiness_report_remote``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

PREFIX = "parlaypicker/price-gap-candidates-v1/"
MIN_EDGE = 0.01
WINDOW = timedelta(hours=3)
MAX_CAPTURE_AGE = timedelta(minutes=10)     # an older capture is not a live price
SLOT_MINUTES = 5
LOOKBACK_SLOTS = 12                          # the app looks back one hour
STALE_AFTER = timedelta(minutes=15)
MAX_BYTES = 2_000_000
MOVEMENT_WINDOW = timedelta(hours=12)       # movement is shown for games starting this soon
MISSING = "Remote evidence object is missing"
NOTICE = ("Research only. These gaps are unproven: they have not been shown to beat the closing price. "
          "Not approved to bet; no stake is attached.")


def _utcnow():
    return datetime.now(timezone.utc)


def _american(probability):
    """The American price with no margin at ``probability`` (the fair price)."""
    if not 0 < probability < 1:
        return None
    return round(-100 * probability / (1 - probability)) if probability >= 0.5 else round(100 * (1 - probability) / probability)


def _label(quote):
    line = quote["line"]
    return f"{quote['selection']} {line:.1f}" if quote["market_family"] == "TOTAL" else f"{quote['selection']} {line:+.1f}"


def current(path, now, *, sports, min_edge=MIN_EDGE, books=None, read_scope=None):
    """Stake-book quotes in each upcoming game's newest capture whose EV against Pinnacle is ``min_edge`` or more."""
    from app_core import news_capture, pinnacle_model as pm
    from app_core.public_quote_policy import canonical_book_label
    if read_scope is None:
        from app_core.prospective_research_models import _read_scope as read_scope
    books = pm.PLAN_BOOKS if books is None else books
    out = []
    for sport in sports:
        for family in news_capture.families(sport):
            events, quotes, _ = read_scope(path, sport, family)
            upcoming = {e["event_id"]: e for e in events.values()
                        if now < pm._t(e["scheduled_start"]) <= now + WINDOW}
            kept, _ = pm.without_junk(pm.slim_sources([q for q in quotes if q["event_id"] in upcoming]))
            for event_id, found in sorted(pm.by_event(kept).items()):
                event = upcoming[event_id]
                newest = max(pm.capture_time(event, q) for q in found)
                if not timedelta(0) <= now - newest <= MAX_CAPTURE_AGE:
                    continue
                live = [q for q in found if pm.capture_time(event, q) == newest]
                for quote in live:
                    # Stored names are provider keys ("draftkings"); the stake books are named by label.
                    if not pm._plan_book(quote, books) or not quote["quote_verified"] or quote["line"] is None:
                        continue
                    fair = pm.fair(event, quote, live, now=now)
                    if fair is None:
                        continue
                    edge = fair * quote["decimal_odds"] - 1
                    if edge < min_edge:
                        continue
                    out.append({"sport": sport, "market_family": family, "away_team": event["away_team"],
                                "home_team": event["home_team"], "scheduled_start": event["scheduled_start"],
                                "sportsbook": canonical_book_label(quote["sportsbook"]), "pick": _label(quote),
                                "american_odds": quote.get("american_odds"), "fair_probability": round(fair, 4),
                                "fair_american_odds": _american(fair), "edge": round(edge, 4),
                                "captured_at": newest.isoformat(timespec="seconds")})
    return sorted(out, key=lambda row: (-row["edge"], row["scheduled_start"], row["sportsbook"], row["pick"]))


def _toward(event, family, line_move, shift):
    """Which side Pinnacle moved toward: the line when it moved, else the price."""
    home, away = (("Over", "Under") if family == "TOTAL" else (event["home_team"], event["away_team"]))
    if line_move:
        # A total going up favours the Over; a home line going down (more points to give) favours the home side.
        return home if (line_move > 0) == (family == "TOTAL") else away
    if shift and abs(shift) >= 0.005:
        return home if shift > 0 else away
    return None


def movement(path, now, *, sports, books=None, read_scope=None, window=MOVEMENT_WINDOW):
    """Per upcoming game: where Pinnacle's main line and no-vig price started, where they are now,
    and which stake books quote Pinnacle's current line (owner 2026-11-12; research only).

    The line is the home line for a side market and the total for a total; the probability is the
    home side's or the Over's. The first capture is the earliest one stored, which can be hours
    after the market opened, so this is movement since the watch first saw the game, not since
    the open. Results are never read and nothing here approves a bet.
    """
    from app_core import news_capture, pinnacle_model as pm
    from app_core.public_quote_policy import canonical_book_label
    if read_scope is None:
        from app_core.prospective_research_models import _read_scope as read_scope
    books = pm.PLAN_BOOKS if books is None else books
    out = []
    for sport in sports:
        for family in news_capture.families(sport):
            events, quotes, _ = read_scope(path, sport, family)
            upcoming = {e["event_id"]: e for e in events.values()
                        if now < pm._t(e["scheduled_start"]) <= now + window}
            kept = pm.slim_sources([q for q in quotes if q["event_id"] in upcoming])
            for event_id, found in sorted(pm.by_event(kept).items()):
                event = upcoming[event_id]
                track = [row for row in pm._main_lines(event, found, family) if row[0] <= now]
                if not track:
                    continue
                (first_at, first_line, first_p), (last_at, last_line, last_p) = track[0], track[-1]
                seen = {}
                for q in found:
                    when = pm.capture_time(event, q)
                    # Stored names are provider keys ("novig"); the stake books are named by label.
                    if pm._plan_book(q, books) and q["quote_verified"] and q["line"] is not None and when <= now:
                        seen.setdefault(canonical_book_label(q["sportsbook"]), {}).setdefault(when, set()).add(
                            pm._home_line(event, q))
                newest = {label: by_time[max(by_time)] for label, by_time in seen.items()}
                line_move, shift = last_line - first_line, last_p - first_p
                out.append({
                    "sport": sport, "market_family": family, "away_team": event["away_team"],
                    "home_team": event["home_team"], "scheduled_start": event["scheduled_start"],
                    "pinnacle_first_line": first_line, "pinnacle_line": last_line, "line_move": line_move,
                    "pinnacle_first_probability": round(first_p, 4), "pinnacle_probability": round(last_p, 4),
                    "probability_shift": round(shift, 4), "moved_toward": _toward(event, family, line_move, shift),
                    "captures": len(track), "first_capture_at": first_at.isoformat(timespec="seconds"),
                    "last_capture_at": last_at.isoformat(timespec="seconds"),
                    "stake_books_at_line": sorted(b for b, lines in newest.items() if last_line in lines),
                    "stake_books_off_line": sorted(b for b, lines in newest.items() if last_line not in lines)})
    return sorted(out, key=lambda row: (-abs(row["line_move"]), -abs(row["probability_shift"]),
                                        row["scheduled_start"], row["home_team"], row["market_family"]))


def key_for(when) -> str:
    when = when.astimezone(timezone.utc)
    return PREFIX + when.strftime("%Y%m%dT%H") + f"{when.minute - when.minute % SLOT_MINUTES:02d}.json"


def publish(client, folder, rows, *, now, sports=(), movement=None):
    """Upload the list (an empty one too: "no gap right now" is information) as this slot's object.
    ``movement`` (``movement()``'s rows) rides in the same object when given."""
    from app_core.evidence_drive import AlreadyExists
    raw = json.dumps({"generated_at": now.isoformat(timespec="seconds"), "min_edge": MIN_EDGE, "sports": list(sports),
                      "notice": NOTICE, "candidates": rows,
                      **({"movement": movement} if movement is not None else {})},
                     sort_keys=True, default=str).encode()
    if len(raw) > MAX_BYTES:
        return {"status": "TOO_LARGE", "bytes": len(raw)}
    key = key_for(now)
    try:
        client.put_object(Bucket=folder, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
    except AlreadyExists:
        return {"status": "ALREADY_PUBLISHED_THIS_SLOT", "key": key}
    return {"status": "PUBLISHED", "key": key, "candidates": len(rows)}


def latest(client, folder, *, now=None, slots=LOOKBACK_SLOTS):
    """The newest published list within ``slots`` 5-minute slots, with ``stale`` set; or None."""
    from app_core.evidence_config import EvidenceStorageError
    now = now or _utcnow()
    for back in range(slots + 1):
        try:
            with client.get_object(Bucket=folder, Key=key_for(now - timedelta(minutes=SLOT_MINUTES * back)))["Body"] as body:
                raw = body.read(MAX_BYTES + 1)
        except EvidenceStorageError as exc:
            if str(exc) == MISSING:
                continue
            raise
        if len(raw) > MAX_BYTES:
            raise ValueError("price_gap_candidates_too_large")
        found = json.loads(raw)
        if not isinstance(found, dict) or not isinstance(found.get("candidates"), list):
            raise ValueError("price_gap_candidates_shape")
        generated = datetime.fromisoformat(str(found.get("generated_at")))
        found["age_minutes"] = round((now - generated).total_seconds() / 60, 1)
        found["stale"] = now - generated > STALE_AFTER
        return found
    return None
