"""Do the stake books lag Pinnacle after NBA injury news? One-time study (owner 2026-11-08).

Step 2 of ``docs/roadmap/NEWS-EDGE-PLAN-2026-11-08.md``. The owner approved a one-time budget
of **5,000 Odds API credits**.

1. **Which news.** From the archived official injury reports (``nba_injury_reports``), each
   (report slot, game) with a material status flip (a listed player moving into or out of
   Out/Doubtful), plus first listings as Out/Doubtful within 3 hours of tip-off. Only pregame
   news is used (slot at least 10 minutes before the start). Flips come first, newest first.
2. **Prices.** For each, the Odds API historical event odds (spreads; Pinnacle, DraftKings,
   FanDuel, BetMGM, Novig: one 10-credit request per snapshot) at ``OFFSETS`` minutes from
   the slot (30 before as the control, then at, 10 after and 30 after). The event id comes
   from the historical events list at the slot (1 credit). Snapshots at or after the start
   are skipped.
3. **Measure.** At each snapshot: Pinnacle's line and no-vig home probability, and each stake
   book's best-side EV against it **at Pinnacle's line**. A lag shows as a stake book 1% or 2%
   better than Pinnacle after the news, when Pinnacle moved and the book had not.

**Spending is capped three ways:** ``max_credits`` (the owner's 5,000), a floor on the
account's remaining monthly credits (``min_remaining``, so routine capture is never starved),
and a write-once start marker per ``label`` on Drive, so the budget cannot be spent twice by
re-running. Every raw response is kept, so the analysis replays. Research only: no picks, no stake.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from app_core import nba_injury_reports as reports

PREFIX = "parlaypicker/news-lag-study-v1"
EVENTS_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events"
ODDS_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/events/{event_id}/odds"
BOOKS = ("pinnacle", "draftkings", "fanduel", "betmgm", "novig")
STAKE_BOOKS = BOOKS[1:]
OFFSETS = (-30, 0, 10, 30)
CREDITS_EVENTS = 1
CREDITS_SNAPSHOT = 10                 # one market (spreads), at most 10 bookmakers
FIRST_LISTING_HOURS = 3
MIN_LEAD = timedelta(minutes=10)
NICKNAMES = {"ATL": "Hawks", "BKN": "Nets", "BOS": "Celtics", "CHA": "Hornets", "CHI": "Bulls", "CLE": "Cavaliers",
             "DAL": "Mavericks", "DEN": "Nuggets", "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets",
             "IND": "Pacers", "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat", "MIL": "Bucks",
             "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks", "OKC": "Thunder", "ORL": "Magic",
             "PHI": "76ers", "PHX": "Suns", "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
             "UTA": "Jazz", "WAS": "Wizards"}


def approx_start(game_date, game_time):
    """The report's game time ("07:30 (ET)", no AM/PM: NBA games are afternoon or evening)."""
    month, day, year = (int(x) for x in game_date.split("/"))
    hour, minute = (int(x) for x in game_time.split()[0].split(":"))
    hour = hour if hour == 12 else hour + 12
    return datetime(year, month, day, hour, minute, tzinfo=reports.ET)


def candidates(manifests) -> list[dict]:
    """News groups (slot, game) from day manifests, flips first then first listings, newest first."""
    groups = {}
    for manifest in manifests:
        for entry in manifest.get("reports", []):
            if not entry.get("changes"):
                continue
            slot = datetime.fromisoformat(entry["slot"])
            times = {(r["game_date"], r["matchup"]): r["game_time"] for r in entry.get("rows", [])
                     if r.get("game_time")}
            for change in entry["changes"]:
                if not change["material"] or (change["game_date"], change["matchup"]) not in times:
                    continue
                start = approx_start(change["game_date"], times[(change["game_date"], change["matchup"])])
                if slot + MIN_LEAD > start:
                    continue
                flip = change["previous"] is not None
                if not flip and (change["status"] not in reports.UNAVAILABLE
                                 or start - slot > timedelta(hours=FIRST_LISTING_HOURS)):
                    continue
                key = (entry["slot"], change["game_date"], change["matchup"])
                group = groups.setdefault(key, {"slot": entry["slot"], "game_date": change["game_date"],
                                                "matchup": change["matchup"], "approx_start": start.isoformat(),
                                                "flip": False, "changes": []})
                group["flip"] = group["flip"] or flip
                group["changes"].append(change)
    return sorted(groups.values(), key=lambda g: (not g["flip"], -datetime.fromisoformat(g["slot"]).timestamp()))


def _iso(value):
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def match_event(events, matchup):
    away, home = matchup.split("@")
    found = [e for e in events if e.get("home_team", "").endswith(NICKNAMES.get(home, "?"))
             and e.get("away_team", "").endswith(NICKNAMES.get(away, "?"))]
    return found[0] if len(found) == 1 else None


def _decimal(american):
    return 1 + american / 100 if american > 0 else 1 + 100 / abs(american)


def _spread(book, home, away):
    for market in book.get("markets") or []:
        if market.get("key") != "spreads":
            continue
        sides = {o.get("name"): o for o in market.get("outcomes") or []}
        if home in sides and away in sides and sides[home].get("point") is not None:
            return float(sides[home]["point"]), _decimal(sides[home]["price"]), _decimal(sides[away]["price"])
    return None


def snapshot_view(event, response) -> dict:
    """Pinnacle's line and no-vig home probability, and each stake book's EV against it at the same line."""
    home, away = event["home_team"], event["away_team"]
    books = {b.get("key"): b for b in ((response or {}).get("data") or {}).get("bookmakers") or []}
    view = {"timestamp": (response or {}).get("timestamp"), "pinnacle_line": None, "pinnacle_home": None,
            "books": {}, "best_ev": None}
    pin = _spread(books["pinnacle"], home, away) if "pinnacle" in books else None
    if pin:
        line, home_dec, away_dec = pin
        p = (1 / home_dec) / (1 / home_dec + 1 / away_dec)
        view["pinnacle_line"], view["pinnacle_home"] = line, round(p, 4)
    for key in STAKE_BOOKS:
        quoted = _spread(books[key], home, away) if key in books else None
        if not quoted:
            continue
        line, home_dec, away_dec = quoted
        ev = None
        if pin and abs(line - view["pinnacle_line"]) < 1e-9:
            p = view["pinnacle_home"]
            ev = round(max(p * home_dec - 1, (1 - p) * away_dec - 1), 4)
        view["books"][key] = {"line": line, "ev": ev}
    evs = [b["ev"] for b in view["books"].values() if b["ev"] is not None]
    view["best_ev"] = max(evs) if evs else None
    return view


def analyze(group) -> dict:
    snaps = [s for s in group.get("snapshots", []) if s.get("view")]
    before = [s["view"] for s in snaps if s["offset"] < 0]
    after = [s["view"] for s in snaps if s["offset"] >= 0]
    lines = [v["pinnacle_line"] for v in (before + after) if v["pinnacle_line"] is not None]
    probs = [v["pinnacle_home"] for v in (before + after) if v["pinnacle_home"] is not None]
    best = lambda views: max((v["best_ev"] for v in views if v["best_ev"] is not None), default=None)  # noqa: E731
    return {"pinnacle_line_move": round(abs(lines[-1] - lines[0]), 2) if len(lines) > 1 else None,
            "pinnacle_prob_move": round(abs(probs[-1] - probs[0]), 4) if len(probs) > 1 else None,
            "best_ev_before": best(before), "best_ev_after": best(after)}


def summary(groups) -> dict:
    studied = [g for g in groups if g.get("analysis")]
    moved = [g for g in studied if (g["analysis"]["pinnacle_line_move"] or 0) >= 0.5
             or (g["analysis"]["pinnacle_prob_move"] or 0) >= 0.02]

    def share(items, key, edge):
        values = [g["analysis"][key] for g in items if g["analysis"][key] is not None]
        return {"with_price": len(values), "at_least": sum(v >= edge for v in values)}
    return {"news_studied": len(studied), "pinnacle_moved": len(moved),
            "after_news_1pct": share(moved, "best_ev_after", 0.01), "after_news_2pct": share(moved, "best_ev_after", 0.02),
            "control_before_1pct": share(studied, "best_ev_before", 0.01),
            "all_after_1pct": share(studied, "best_ev_after", 0.01)}


class Stop(Exception):
    pass


def run(groups, get, *, api_key, max_credits, min_remaining, log=None) -> dict:
    """Spend at most ``max_credits``; stop when the account's remaining credits reach ``min_remaining``.

    ``get(url, params) -> (status, headers, json)``. Returns {spent, stopped, groups}.
    """
    spent, stopped, done = 0, None, []
    books = list(BOOKS)

    def call(url, params, reserve):
        nonlocal spent
        if spent + reserve > max_credits:
            raise Stop("BUDGET_CAP")
        status, headers, body = get(url, dict(params, apiKey=api_key))
        charged = int(headers.get("x-requests-last") or (reserve if status == 200 else 0))
        spent += charged
        remaining = headers.get("x-requests-remaining")
        if remaining is not None and float(remaining) < min_remaining:
            raise Stop("MONTHLY_FLOOR")
        return body if status == 200 else status

    per_group = CREDITS_EVENTS + CREDITS_SNAPSHOT * len(OFFSETS)
    for group in groups:
        if spent + per_group > max_credits:
            stopped = "BUDGET_CAP"
            break
        slot = datetime.fromisoformat(group["slot"])
        item = dict(group, snapshots=[])
        try:
            events = call(EVENTS_URL, {"date": _iso(slot)}, CREDITS_EVENTS)
            event = match_event((events if isinstance(events, dict) else {}).get("data") or [], group["matchup"])
            if event is None:
                item["status"] = "NO_EVENT"
                done.append(item)
                continue
            item["event"] = {k: event.get(k) for k in ("id", "home_team", "away_team", "commence_time")}
            start = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
            for offset in OFFSETS:
                at = slot + timedelta(minutes=offset)
                if at >= start:
                    continue
                params = {"markets": "spreads", "oddsFormat": "american", "date": _iso(at)}
                body = call(ODDS_URL.format(event_id=event["id"]), dict(params, bookmakers=",".join(books)),
                            CREDITS_SNAPSHOT)
                if body == 422 and "novig" in books:
                    books.remove("novig")  # a bookmaker key the endpoint does not know: drop it once, retry
                    body = call(ODDS_URL.format(event_id=event["id"]), dict(params, bookmakers=",".join(books)),
                                CREDITS_SNAPSHOT)
                body = body if isinstance(body, dict) else None
                item["snapshots"].append({"offset": offset, "requested": _iso(at), "response": body,
                                          "view": snapshot_view(item["event"], body) if body else None})
            item["status"] = "STUDIED"
            item["analysis"] = analyze(item)
        except Stop as exc:
            stopped = str(exc)
            if item["snapshots"]:
                item["status"], item["analysis"] = "PARTIAL", analyze(item)
                done.append(item)
            break
        done.append(item)
        if log:
            log(f"{group['slot']} {group['matchup']}: {item.get('status')} spent {spent}")
    return {"spent": spent, "stopped": stopped, "books": books, "groups": done, "summary": summary(done)}


def start_key(label):
    return f"{PREFIX}/{label}/started.json"


def result_key(label):
    return f"{PREFIX}/{label}/study.json"


def claim(client, label, *, max_credits, now):
    """Write the label's start marker once; refuses when this label already ran (no double spend)."""
    try:
        client.get_object(Key=start_key(label))
    except Exception:
        body = json.dumps({"label": label, "max_credits": max_credits, "started_at": now.isoformat()}).encode()
        client.put_object(Key=start_key(label), Body=body, IfNoneMatch="*")
        return True
    return False
